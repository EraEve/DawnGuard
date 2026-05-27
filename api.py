"""FastAPI entrypoint for the MAPFM medical AI ecosystem.

Supports multimodal input: CSV (vital signs, lab results), TXT (clinical text),
JPG/PNG (medical images).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

from mapfm_ecosystem_repaired import EcosystemConfig, HeterogeneousMultiAgentEcosystem, load_medical_dataset
from mapfm_utils.multimodal_utils import MultimodalPreprocessor

app = FastAPI(title="MAPFM Medical AI API", version="2.0.0")

_ecosystem: Optional[HeterogeneousMultiAgentEcosystem] = None
_preprocessor = MultimodalPreprocessor()


def _get_ecosystem() -> HeterogeneousMultiAgentEcosystem:
    """Lazy-initialize the ecosystem on first request so import never crashes."""
    global _ecosystem
    if _ecosystem is not None:
        return _ecosystem
    config = EcosystemConfig()
    csv_path = Path("medNo.22.csv")
    dataset = load_medical_dataset(csv_path if csv_path.exists() else None, seed=config.random_seed)
    _ecosystem = HeterogeneousMultiAgentEcosystem(
        config=config, dataset=dataset, base_model=config.ollama_model_name, verbose=False,
    )
    return _ecosystem


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Single diagnosis request (text-only or with pre-processed multimodal input)."""

    patient_query: str = Field(..., min_length=1)
    multimodal_input: Optional[Dict[str, Any]] = None
    true_label: Optional[str] = None
    force_high_risk: bool = False


class BatchRequest(BaseModel):
    """Batch diagnosis request for thread-pooled concurrent execution."""

    requests: List[QueryRequest]


# ---------------------------------------------------------------------------
# Standard endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Return API health status (does not force ecosystem init)."""
    if _ecosystem is not None:
        return {"status": "ok", "document_count": _ecosystem.online_kb.get_stats()["document_count"]}
    return {"status": "ready", "detail": "Ecosystem not initialized — first request will trigger warm-up."}


@app.post("/diagnose")
def diagnose(request: QueryRequest) -> Dict[str, Any]:
    """Run a single multi-agent diagnosis request (text + optional multimodal dict)."""
    return _get_ecosystem().run_collaborative_task(**request.model_dump())


@app.post("/diagnose/batch")
def diagnose_batch(request: BatchRequest) -> List[Dict[str, Any]]:
    """Run concurrent diagnosis requests without changing single-task logic."""
    return _get_ecosystem().run_concurrent_tasks([item.model_dump() for item in request.requests])


# ---------------------------------------------------------------------------
# Multimodal data upload endpoints
# ---------------------------------------------------------------------------

@app.post("/diagnose/multimodal")
async def diagnose_multimodal(
    patient_query: str = Form(..., min_length=1),
    true_label: Optional[str] = Form(None),
    force_high_risk: bool = Form(False),
    vital_signs_csv: Optional[UploadFile] = File(None),
    lab_results_csv: Optional[UploadFile] = File(None),
    clinical_text_txt: Optional[UploadFile] = File(None),
    medical_images: Optional[List[UploadFile]] = File(None),
) -> Dict[str, Any]:
    """Full multimodal diagnosis with file uploads.

    Accepts up to 4 modalities simultaneously:
      - vital_signs_csv:   CSV with timestamp, HR, BP, Temp, RR, SpO2
      - lab_results_csv:   CSV with timestamp, test_name, value, unit, ref_range
      - clinical_text_txt: TXT with clinical notes
      - medical_images:    JPG/PNG medical images (up to 5)
    """
    temp_dir = tempfile.mkdtemp(prefix="mapfm_api_")

    try:
        # Save uploaded files to temp directory
        vital_path = await _save_upload(vital_signs_csv, temp_dir)
        lab_path = await _save_upload(lab_results_csv, temp_dir)
        text_path = await _save_upload(clinical_text_txt, temp_dir)
        image_paths = await _save_uploads(medical_images, temp_dir, max_count=5)

        # Preprocess multimodal data
        patient_data = _preprocessor.process(
            vital_signs_csv=vital_path,
            lab_results_csv=lab_path,
            clinical_text_txt=text_path,
            image_paths=image_paths,
        )

        # Build multimodal_input dict for the ecosystem
        multimodal_input: Dict[str, Any] = {}
        if patient_data.has_vital_signs():
            multimodal_input["time_series"] = patient_data.vital_signs_time_series().tolist()
            multimodal_input["vital_signs_count"] = len(patient_data.vital_signs)
        if patient_data.has_lab_results():
            multimodal_input["lab_results"] = [
                {"test": r.test_name, "value": r.value, "unit": r.unit,
                 "abnormal": r.abnormal}
                for r in patient_data.lab_results
            ]
        if patient_data.has_clinical_texts():
            multimodal_input["clinical_text"] = " ".join(
                t.content for t in patient_data.clinical_texts
            )[:2000]
        if patient_data.has_medical_images():
            multimodal_input["image_count"] = len(patient_data.medical_images)
            multimodal_input["image_modalities"] = [
                img.modality for img in patient_data.medical_images
            ]

        # Set clinical signals from vital signs
        if patient_data.has_vital_signs():
            vs_arr = patient_data.vital_signs_array()
            if vs_arr.size > 0:
                multimodal_input["posture_instability"] = float(
                    vs_arr[-1, 0] / 100.0 if not np.isnan(vs_arr[-1, 0]) else 0.5
                )

        # Run through ecosystem
        result = _get_ecosystem().run_collaborative_task(
            patient_query=patient_query,
            multimodal_input=multimodal_input if multimodal_input else None,
            true_label=true_label,
            force_high_risk=force_high_risk,
        )

        # Attach patient data metadata
        result["multimodal_meta"] = {
            "vital_signs_count": len(patient_data.vital_signs),
            "lab_results_count": len(patient_data.lab_results),
            "clinical_text_count": len(patient_data.clinical_texts),
            "medical_image_count": len(patient_data.medical_images),
            "preprocessing_errors": patient_data.metadata,
        }

        return result

    finally:
        patient_data.cleanup_temp_dir()
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/diagnose/multimodal/simple")
async def diagnose_multimodal_simple(
    patient_query: str = Form(..., min_length=1),
    vital_signs_csv: Optional[UploadFile] = File(None),
    lab_results_csv: Optional[UploadFile] = File(None),
    clinical_text_txt: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Simplified multimodal diagnosis endpoint (no images).

    For quick CSV/TXT clinical data uploads without complex multipart setup.
    """
    return await diagnose_multimodal(
        patient_query=patient_query,
        vital_signs_csv=vital_signs_csv,
        lab_results_csv=lab_results_csv,
        clinical_text_txt=clinical_text_txt,
    )


@app.get("/system/health")
def system_health() -> Dict[str, Any]:
    """Return detailed system health including degradation status."""
    from mapfm_utils.system_monitor import get_system_monitor

    monitor = get_system_monitor()
    health = monitor.check()
    return {
        "ollama_healthy": health.ollama_healthy,
        "medtsllm_loaded": health.medtsllm_loaded,
        "gpu_available": health.gpu_available,
        "gpu_memory_ratio": health.gpu_memory_ratio,
        "memory_ratio": health.memory_ratio,
        "degradation_level": health.degradation_level.name,
        "degradation_reason": health.degradation_reason,
        "user_message": monitor.get_user_message(),
        "history": monitor.get_history(limit=10),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _save_upload(upload: Optional[UploadFile], dest_dir: str) -> Optional[str]:
    """Save a single uploaded file to dest_dir. Returns path or None."""
    if upload is None or upload.filename is None:
        return None
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(upload.filename).name}"
    dest = str(Path(dest_dir) / safe_name)
    with open(dest, "wb") as fh:
        content = await upload.read()
        fh.write(content)
    return dest


async def _save_uploads(
    uploads: Optional[List[UploadFile]], dest_dir: str, max_count: int = 5
) -> List[str]:
    """Save multiple uploaded files. Returns list of paths."""
    if not uploads:
        return []
    paths: List[str] = []
    for upload in uploads[:max_count]:
        path = await _save_upload(upload, dest_dir)
        if path:
            paths.append(path)
    return paths


# Lazy numpy import for vital_signs processing in multimodal endpoint
import numpy as np
