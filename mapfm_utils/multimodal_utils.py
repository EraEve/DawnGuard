"""
Multimodal data preprocessing pipeline for MAPFM HMAE system.

Supports CSV (vital signs, lab results), TXT (clinical notes),
and JPG/PNG (medical imaging) input. Produces MultimodalPatientData
objects for consumption by TFA and other modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# Multimodal patient data container
# ---------------------------------------------------------------------------

@dataclass
class VitalSignRecord:
    """Single vital-sign observation."""
    timestamp: datetime
    heart_rate: Optional[float] = None       # bpm
    blood_pressure_sys: Optional[float] = None  # mmHg
    blood_pressure_dia: Optional[float] = None  # mmHg
    temperature: Optional[float] = None      # Celsius
    respiratory_rate: Optional[float] = None # breaths/min
    spo2: Optional[float] = None             # %


@dataclass
class LabResultRecord:
    """Single laboratory test result."""
    timestamp: datetime
    test_name: str
    value: float
    unit: str = ""
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    abnormal: bool = False


@dataclass
class ClinicalTextRecord:
    """Single clinical text note."""
    timestamp: datetime
    text_type: str   # e.g. "admission_note", "progress_note", "discharge_summary"
    content: str


@dataclass
class MedicalImageRecord:
    """Single medical image reference."""
    timestamp: datetime
    modality: str     # e.g. "X-ray", "CT", "MRI", "ultrasound"
    body_part: str
    file_path: str    # path to the image file
    precomputed_features: Optional[np.ndarray] = None


@dataclass
class MultimodalPatientData:
    """Unified multimodal patient data container.

    This is the canonical in-memory representation passed between modules.
    All preprocessing pipelines produce instances of this class.
    """
    patient_id: str = ""
    vital_signs: List[VitalSignRecord] = field(default_factory=list)
    lab_results: List[LabResultRecord] = field(default_factory=list)
    clinical_texts: List[ClinicalTextRecord] = field(default_factory=list)
    medical_images: List[MedicalImageRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _temp_dir: Optional[str] = None

    def has_vital_signs(self) -> bool:
        return len(self.vital_signs) > 0

    def has_lab_results(self) -> bool:
        return len(self.lab_results) > 0

    def has_clinical_texts(self) -> bool:
        return len(self.clinical_texts) > 0

    def has_medical_images(self) -> bool:
        return len(self.medical_images) > 0

    def is_empty(self) -> bool:
        return not (self.has_vital_signs() or self.has_lab_results()
                    or self.has_clinical_texts() or self.has_medical_images())

    def vital_signs_array(self) -> np.ndarray:
        """Return vital signs as (N, 6) float array [hr, bp_sys, bp_dia, temp, rr, spo2]."""
        rows = []
        for r in sorted(self.vital_signs, key=lambda x: x.timestamp):
            rows.append([
                r.heart_rate if r.heart_rate is not None else np.nan,
                r.blood_pressure_sys if r.blood_pressure_sys is not None else np.nan,
                r.blood_pressure_dia if r.blood_pressure_dia is not None else np.nan,
                r.temperature if r.temperature is not None else np.nan,
                r.respiratory_rate if r.respiratory_rate is not None else np.nan,
                r.spo2 if r.spo2 is not None else np.nan,
            ])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6), dtype=np.float32)

    def vital_signs_time_series(self) -> np.ndarray:
        """Return a flattened time-series representation for TFA consumption.

        Interpolates missing values and returns a 1-D array of concatenated
        normalized vital sign channels, suitable as TFA history input.
        """
        arr = self.vital_signs_array()
        if arr.size == 0:
            return np.array([], dtype=np.float32)
        # Forward-fill then backward-fill NaNs
        arr_filled = arr.copy()
        for col in range(arr_filled.shape[1]):
            col_data = arr_filled[:, col]
            mask = ~np.isnan(col_data)
            if not mask.any():
                continue
            indices = np.arange(len(col_data))
            col_data[~mask] = np.interp(indices[~mask], indices[mask], col_data[mask])
        # Normalize each channel to [0, 1] using fixed clinical ranges
        ranges = np.array([
            [30, 220],   # HR: 30-220 bpm
            [60, 250],   # SYS: 60-250 mmHg
            [30, 150],   # DIA: 30-150 mmHg
            [34, 43],    # Temp: 34-43 C
            [4, 60],     # RR: 4-60 breaths/min
            [70, 100],   # SpO2: 70-100%
        ], dtype=np.float32)
        mins = ranges[:, 0]
        maxs = ranges[:, 1]
        arr_norm = np.clip((arr_filled - mins) / (maxs - mins + 1e-8), 0.0, 1.0)
        return arr_norm.flatten().astype(np.float32)

    def cleanup_temp_dir(self) -> None:
        """Remove the temporary directory used for multimodal data storage."""
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

class MultimodalPreprocessor:
    """Unified preprocessing pipeline for multimodal medical data.

    Steps:
      1. Data cleaning (remove duplicates, outliers, excessive missing values)
      2. Data normalization (unify time format, units, data structures)
      3. Data desensitization (strip patient-identifying information)
      4. Data formatting (convert to MultimodalPatientData)

    Usage:
        preprocessor = MultimodalPreprocessor()
        patient_data = preprocessor.process(
            vital_signs_csv="path/to/vitals.csv",
            lab_results_csv="path/to/labs.csv",
            clinical_text_txt="path/to/notes.txt",
            image_paths=["path/to/scan1.jpg"],
        )
    """

    # Clinical reference ranges for outlier detection
    VITAL_RANGES: Dict[str, Tuple[float, float]] = {
        "heart_rate": (20, 250),
        "blood_pressure_sys": (40, 280),
        "blood_pressure_dia": (20, 180),
        "temperature": (33.0, 44.0),
        "respiratory_rate": (2, 80),
        "spo2": (50, 100),
    }

    LAB_RANGES: Dict[str, Tuple[float, float]] = {
        "wbc": (0.1, 500),
        "rbc": (0.5, 10),
        "hemoglobin": (2, 25),
        "platelets": (5, 2000),
        "creatinine": (0.1, 30),
        "bun": (1, 200),
        "glucose": (20, 800),
        "sodium": (100, 200),
        "potassium": (1.5, 10),
        "calcium": (3, 20),
        "crp": (0, 500),
        "troponin": (0, 100),
        "inr": (0.5, 15),
        "alt": (1, 3000),
        "ast": (1, 3000),
        "bilirubin": (0.1, 40),
    }

    # PII patterns for desensitization
    _PII_PATTERNS: List[Tuple[str, str]] = [
        (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', '[NAME]'),   # Full names
        (r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[SSN]'),           # SSN-like
        (r'\b\d{15,19}\b', '[ID_NUMBER]'),                           # Long ID numbers
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{10,11}\b', '[PHONE]'),                               # Phone numbers
        (r'\b(?:MRN|Medical\s*Record)(?:\s*#?\s*\d+)?\b', '[MRN]'),  # MRNs
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]'),        # IP addresses
    ]

    def __init__(self, temp_dir: Optional[str] = None) -> None:
        self._temp_base = temp_dir or tempfile.mkdtemp(prefix="mapfm_multimodal_")

    # ----- Vital signs CSV processing -----

    def process_vital_signs_csv(self, csv_path: str) -> List[VitalSignRecord]:
        """Parse and clean a vital-signs CSV file.

        Expected columns (case-insensitive, partial match):
          timestamp / time / datetime, heart_rate / hr / pulse,
          blood_pressure / bp_sys+bp_dia or systolic/diastolic,
          temperature / temp, respiratory_rate / rr, spo2 / oxygen
        """
        if pd is None:
            raise ImportError("pandas is required for CSV processing")
        df = pd.read_csv(csv_path)
        df = self._clean_dataframe(df)
        records: List[VitalSignRecord] = []
        for _, row in df.iterrows():
            ts = self._parse_timestamp(row)
            records.append(VitalSignRecord(
                timestamp=ts,
                heart_rate=self._safe_float(self._find_column(row, [
                    "heart_rate", "hr", "pulse", "heartrate"
                ])),
                blood_pressure_sys=self._safe_float(self._find_column(row, [
                    "blood_pressure_sys", "bp_sys", "systolic", "sbp", "bp_systolic"
                ])),
                blood_pressure_dia=self._safe_float(self._find_column(row, [
                    "blood_pressure_dia", "bp_dia", "diastolic", "dbp", "bp_diastolic"
                ])),
                temperature=self._safe_float(self._find_column(row, [
                    "temperature", "temp", "body_temp"
                ])),
                respiratory_rate=self._safe_float(self._find_column(row, [
                    "respiratory_rate", "rr", "resp_rate", "respiration"
                ])),
                spo2=self._safe_float(self._find_column(row, [
                    "spo2", "oxygen", "o2_sat", "oxygen_saturation", "pulse_ox"
                ])),
            ))
        return self._remove_outliers_vital(records)

    # ----- Lab results CSV processing -----

    def process_lab_results_csv(self, csv_path: str) -> List[LabResultRecord]:
        """Parse and clean a lab-results CSV file.

        Expected columns: timestamp, test_name / test / item, value / result,
        unit, reference_range / ref_low+ref_high
        """
        if pd is None:
            raise ImportError("pandas is required for CSV processing")
        df = pd.read_csv(csv_path)
        df = self._clean_dataframe(df)
        records: List[LabResultRecord] = []
        for _, row in df.iterrows():
            ts = self._parse_timestamp(row)
            test_name = str(self._find_column(row, ["test_name", "test", "item", "name", "analyte"], default="Unknown"))
            value = self._safe_float(self._find_column(row, ["value", "result", "result_value"]))
            unit = str(self._find_column(row, ["unit", "units"], default=""))
            ref_low = self._safe_float(self._find_column(row, ["reference_low", "ref_low", "low", "ref_range_low"]))
            ref_high = self._safe_float(self._find_column(row, ["reference_high", "ref_high", "high", "ref_range_high"]))
            abnormal = False
            if ref_low is not None and ref_high is not None and value is not None:
                abnormal = value < ref_low or value > ref_high
            records.append(LabResultRecord(
                timestamp=ts, test_name=test_name, value=value or 0.0,
                unit=unit, reference_range_low=ref_low,
                reference_range_high=ref_high, abnormal=abnormal,
            ))
        return self._remove_outliers_lab(records)

    # ----- Clinical text TXT processing -----

    def process_clinical_text(self, txt_path: str) -> List[ClinicalTextRecord]:
        """Parse a clinical text file (TXT format).

        Expected format — each section starts with a header line like:
            [YYYY-MM-DD HH:MM] text_type
            Content lines...
        Or plain text is treated as a single progress_note.
        """
        with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        # Try structured parse
        records = self._parse_structured_clinical_text(raw)
        if not records:
            # Fallback: treat whole file as one note
            records = [ClinicalTextRecord(
                timestamp=datetime.fromtimestamp(os.path.getmtime(txt_path)),
                text_type="progress_note",
                content=raw.strip(),
            )]
        # Desensitize
        for rec in records:
            rec.content = self._desensitize(rec.content)
        return records

    # ----- Medical image processing -----

    def process_medical_image(self, image_path: str) -> Optional[MedicalImageRecord]:
        """Process a single medical image (JPG/PNG).

        Validates the file, extracts basic metadata, and copies to
        a temporary location for safe processing.
        """
        if not os.path.isfile(image_path):
            return None
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".dcm"):
            return None
        # Copy to temp dir for safe access
        dest = os.path.join(self._temp_base, f"{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}")
        shutil.copy2(image_path, dest)
        # Try to extract metadata with PIL
        modality = "unknown"
        body_part = "unknown"
        if Image is not None:
            try:
                img = Image.open(dest)
                modality = self._guess_modality(image_path)
                body_part = self._guess_body_part(image_path)
                img.close()
            except Exception:
                pass
        ts = datetime.fromtimestamp(os.path.getmtime(image_path))
        return MedicalImageRecord(
            timestamp=ts, modality=modality, body_part=body_part,
            file_path=dest,
        )

    # ----- Main processing entry point -----

    def process(
        self,
        vital_signs_csv: Optional[str] = None,
        lab_results_csv: Optional[str] = None,
        clinical_text_txt: Optional[str] = None,
        image_paths: Optional[Sequence[str]] = None,
        patient_id: str = "",
    ) -> MultimodalPatientData:
        """Run the full preprocessing pipeline across all input modalities.

        Returns a MultimodalPatientData instance. Caller is responsible for
        calling .cleanup_temp_dir() when the data is no longer needed.
        """
        patient_data = MultimodalPatientData(
            patient_id=patient_id or self._generate_patient_id(),
            _temp_dir=self._temp_base,
        )

        if vital_signs_csv and os.path.isfile(vital_signs_csv):
            try:
                patient_data.vital_signs = self.process_vital_signs_csv(vital_signs_csv)
            except Exception as exc:
                patient_data.metadata["vital_signs_error"] = str(exc)

        if lab_results_csv and os.path.isfile(lab_results_csv):
            try:
                patient_data.lab_results = self.process_lab_results_csv(lab_results_csv)
            except Exception as exc:
                patient_data.metadata["lab_results_error"] = str(exc)

        if clinical_text_txt and os.path.isfile(clinical_text_txt):
            try:
                patient_data.clinical_texts = self.process_clinical_text(clinical_text_txt)
            except Exception as exc:
                patient_data.metadata["clinical_text_error"] = str(exc)

        if image_paths:
            for img_path in image_paths:
                try:
                    rec = self.process_medical_image(img_path)
                    if rec:
                        patient_data.medical_images.append(rec)
                except Exception as exc:
                    patient_data.metadata.setdefault("image_errors", []).append(
                        {img_path: str(exc)}
                    )

        patient_data.metadata["processed_at"] = datetime.now().isoformat()
        patient_data.metadata["source_files"] = {
            k: v for k, v in {
                "vital_signs_csv": vital_signs_csv,
                "lab_results_csv": lab_results_csv,
                "clinical_text_txt": clinical_text_txt,
                "image_paths": list(image_paths) if image_paths else None,
            }.items() if v is not None
        }
        return patient_data

    # ----- Internal helpers -----

    def _clean_dataframe(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Remove duplicate rows and rows with >50% missing values."""
        df = df.drop_duplicates()
        df = df.dropna(thresh=max(1, len(df.columns) // 2))
        return df.reset_index(drop=True)

    @staticmethod
    def _find_column(row, candidates: List[str], default: Any = None) -> Any:
        """Find the first matching column in a DataFrame row (case-insensitive)."""
        if hasattr(row, "index"):
            row_lower = {str(k).lower().replace(" ", "_"): v for k, v in row.items()}
            for cand in candidates:
                if cand in row_lower:
                    return row_lower[cand]
        return default

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_timestamp(row) -> datetime:
        """Parse timestamp from a DataFrame row with flexible format handling."""
        for col_name in ["timestamp", "time", "datetime", "date", "date_time", "recorded_at"]:
            val = MultimodalPreprocessor._find_column(row, [col_name])
            if val is not None:
                try:
                    if isinstance(val, datetime):
                        return val
                    if isinstance(val, (int, float)):
                        # Unix timestamp (seconds or ms)
                        if val > 1e12:
                            val = val / 1000.0
                        return datetime.fromtimestamp(float(val))
                    # Try common string formats
                    for fmt in [
                        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                        "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                        "%d/%m/%Y %H:%M:%S", "%Y-%m-%d",
                    ]:
                        try:
                            return datetime.strptime(str(val).strip(), fmt)
                        except ValueError:
                            continue
                    return pd.Timestamp(str(val)).to_pydatetime()
                except Exception:
                    pass
        return datetime.now()

    def _remove_outliers_vital(self, records: List[VitalSignRecord]) -> List[VitalSignRecord]:
        """Flag (don't remove) vital sign values outside clinical ranges."""
        for rec in records:
            for attr, (lo, hi) in self.VITAL_RANGES.items():
                val = getattr(rec, attr, None)
                if val is not None and (val < lo or val > hi):
                    setattr(rec, attr, None)  # Set to None to mark as outlier
        # Remove records where ALL values are None
        def _has_any(rec: VitalSignRecord) -> bool:
            return any(getattr(rec, a) is not None
                       for a in ["heart_rate", "blood_pressure_sys", "temperature",
                                  "respiratory_rate", "spo2"])
        return [r for r in records if _has_any(r)]

    def _remove_outliers_lab(self, records: List[LabResultRecord]) -> List[LabResultRecord]:
        """Remove lab results with values outside absolute physiological ranges."""
        kept = []
        for rec in records:
            key = rec.test_name.lower().replace(" ", "_")
            lo, hi = self.LAB_RANGES.get(key, (None, None))
            if lo is not None and hi is not None:
                if rec.value < lo or rec.value > hi:
                    continue
            kept.append(rec)
        return kept

    @staticmethod
    def _parse_structured_clinical_text(raw: str) -> List[ClinicalTextRecord]:
        """Parse structured [timestamp] type\\ncontent blocks."""
        pattern = re.compile(
            r'\[([^\]]+)\]\s*(?:type[=:]\s*)?(\w+)\s*\n(.*?)(?=\n\[|$)',
            re.DOTALL | re.IGNORECASE,
        )
        records = []
        for match in pattern.finditer(raw):
            ts_str = match.group(1).strip()
            text_type = match.group(2).strip().lower()
            content = match.group(3).strip()
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
            except ValueError:
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d")
                except ValueError:
                    ts = datetime.now()
            if content:
                records.append(ClinicalTextRecord(timestamp=ts, text_type=text_type, content=content))
        return records

    @classmethod
    def _desensitize(cls, text: str) -> str:
        """Remove PII from clinical text."""
        for pattern, replacement in cls._PII_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _generate_patient_id() -> str:
        return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]

    @staticmethod
    def _guess_modality(path: str) -> str:
        name = os.path.basename(path).lower()
        for keyword, modality in [
            ("xray", "X-ray"), ("x_ray", "X-ray"), ("xr_", "X-ray"),
            ("ct_", "CT"), ("ct-", "CT"), ("_ct", "CT"),
            ("mri", "MRI"), ("mr_", "MRI"),
            ("us_", "Ultrasound"), ("ultrasound", "Ultrasound"),
            ("pet", "PET"), ("spect", "SPECT"),
        ]:
            if keyword in name:
                return modality
        return "unknown"

    @staticmethod
    def _guess_body_part(path: str) -> str:
        name = os.path.basename(path).lower()
        for keyword, part in [
            ("chest", "Chest"), ("thorax", "Chest"), ("lung", "Chest"),
            ("head", "Head"), ("brain", "Head"), ("skull", "Head"),
            ("abdomen", "Abdomen"), ("abdominal", "Abdomen"),
            ("spine", "Spine"), ("spinal", "Spine"),
            ("pelvis", "Pelvis"), ("pelvic", "Pelvis"),
            ("knee", "Knee"), ("shoulder", "Shoulder"), ("hip", "Hip"),
            ("wrist", "Wrist"), ("ankle", "Ankle"), ("elbow", "Elbow"),
        ]:
            if keyword in name:
                return part
        return "unknown"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_multimodal_patient_data(
    vital_signs_csv: Optional[str] = None,
    lab_results_csv: Optional[str] = None,
    clinical_text_txt: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    patient_id: str = "",
) -> MultimodalPatientData:
    """One-shot creation of MultimodalPatientData from file paths.

    Creates a temporary directory for image storage. Caller should call
    .cleanup_temp_dir() when done.
    """
    preprocessor = MultimodalPreprocessor()
    return preprocessor.process(
        vital_signs_csv=vital_signs_csv,
        lab_results_csv=lab_results_csv,
        clinical_text_txt=clinical_text_txt,
        image_paths=image_paths,
        patient_id=patient_id,
    )
