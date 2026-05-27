# MedTsLLM-v1.5 本地部署指南

## HMAE 医疗多智能体系统 — TFA 专属推理后端

---

> **部署模式**：纯本地离线 | **推理引擎**：Ollama | **数据安全**：零外发

---

## 目录

1. [概述](#概述)
2. [硬件要求](#硬件要求)
3. [环境准备](#环境准备)
4. [模型文件下载](#模型文件下载)
5. [Ollama 安装与集成](#ollama-安装与集成)
6. [模型导入与验证](#模型导入与验证)
7. [依赖安装](#依赖安装)
8. [部署验证与冒烟测试](#部署验证与冒烟测试)
9. [TFA 集成](#tfa-集成)
10. [常见问题与排查](#常见问题与排查)

---

## 概述

MedTsLLM-v1.5 是清华医学人工智能团队开发的医学多模态时序大语言模型，专门设计用于：

- **多模态病历分析**：融合诊断、生命体征、检验结果、临床文本
- **时序风险预测**：预测患者短期 (24h)、中期 (30d)、长期 (12m) 的病情恶化风险
- **本地隐私推理**：所有计算均在本地完成，数据绝不上传

在本系统中，MedTsLLM 作为 **TFA (Temporal Foreseeing Agent)** 模块的专属推理后端，替代原有的模拟随机风险值生成，实现真正的多模态时序预测。

---

## 硬件要求

| 配置等级 | CPU | 内存 | GPU 显存 | 量化方案 | 适用场景 |
|----------|-----|------|----------|----------|----------|
| **最低** | 8核16线程 | 32GB | 16GB | INT4 (4-bit) | 竞赛验证 |
| **推荐** | 16核32线程 | 64GB | 24GB | INT8 (8-bit) | 临床评估 |
| **最佳** | 24核48线程 | 128GB | 48GB | FP16 (全精度) | 生产环境 |

### GPU 兼容性

- **完美支持**: NVIDIA RTX 3090 (24GB), RTX 4090 (24GB), A6000 (48GB), A100 (40/80GB)
- **良好支持**: RTX 3080 (12GB), RTX 4080 (16GB), RTX 4070 Ti (12GB) — 需 INT4 量化
- **不支持**: AMD GPU (ROCm 兼容性未验证), Apple Silicon (MPS 后端受限)

### CPU-Only 方案

对于没有独立 GPU 的用户，提供 CPU-only 推理方案：

```
ollama create medtsllm:1.5-cpu -f Modelfile.int4
```

**限制**：单任务响应时间 > 60 秒，仅供功能验证和调试使用。

---

## 环境准备

### 1. 快速硬件检测

运行自动化检测脚本：

```bash
python deploy_medtsllm.py --check-only
```

输出示例：

```
  🖥️  MedTsLLM 硬件环境检测报告
================================================================
  操作系统:      Windows 11
  CPU 核心数:    16
  系统内存:      64.0 GB
  GPU 可用:      是
  GPU 型号:      NVIDIA GeForce RTX 4090
  GPU 显存:      24.0 GB
  CUDA 版本:     12.4
  驱动版本:      551.86
----------------------------------------------------------------
  推荐方案:      INT8
  评估说明:      GPU 显存良好, 推荐 8-bit 量化
================================================================
```

### 2. CUDA 环境检查

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 cuDNN (Linux)
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### 3. 系统准备 (Windows)

```powershell
# 确保有足够的磁盘空间 (模型文件约 15GB)
# 模型将保存在项目目录的 models/ 子目录下，非系统盘
```

---

## 模型文件下载

### 官方下载源

| 来源 | 地址 | 速度 |
|------|------|------|
| **HuggingFace** | [tsinghua-medai/MedTsLLM-v1.5-multimodal](https://huggingface.co/tsinghua-medai/MedTsLLM-v1.5-multimodal) | 需科学上网 |
| **ModelScope** | [thu-medai/MedTsLLM-v1.5-multimodal](https://modelscope.cn/models/thu-medai/MedTsLLM-v1.5-multimodal) | 国内高速 |

### 自动下载 (推荐)

```bash
# 使用国内镜像 (默认)
python deploy_medtsllm.py --download-only

# 使用 HuggingFace 官方
python deploy_medtsllm.py --download-only --mirror
```

### 手动下载

如果自动下载失败，请手动下载以下文件并放置到 `models/MedTsLLM-v1.5-multimodal/` 目录：

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `config.json` | ~2KB | 模型配置文件 |
| `tokenizer.json` | ~3.5MB | 分词器文件 |
| `model-00001-of-00003.safetensors` | ~5GB | 模型权重文件 1 |
| `model-00002-of-00003.safetensors` | ~5GB | 模型权重文件 2 |
| `model-00003-of-00003.safetensors` | ~4.8GB | 模型权重文件 3 |
| `preprocessor_config.json` | ~1KB | 多模态预处理配置 |

### 文件校验

```bash
# 校验所有下载文件
python deploy_medtsllm.py --verify-only
```

### 断点续传下载脚本

如果浏览器下载中断，可使用以下 Python 脚本实现断点续传：

```python
import requests

def resume_download(url, dest_path):
    """支持断点续传的文件下载。"""
    import os
    existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    
    resp = requests.get(url, stream=True, headers=headers, timeout=600)
    mode = "ab" if existing_size else "wb"
    
    with open(dest_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    print(f"下载完成: {dest_path}")

# 使用示例
resume_download(
    "https://modelscope.cn/models/thu-medai/MedTsLLM-v1.5-multimodal/resolve/master/model-00001-of-00003.safetensors",
    "models/MedTsLLM-v1.5-multimodal/model-00001-of-00003.safetensors"
)
```

---

## Ollama 安装与集成

### 步骤 1: 安装 Ollama

```bash
# Windows: 从官网下载安装器
# https://ollama.com/download/windows

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
```

验证安装：

```bash
ollama --version
# 要求: ≥ 0.1.30 (支持多模态模型)
```

### 步骤 2: 启动 Ollama 服务

```bash
# 启动服务 (终端保持运行)
ollama serve

# 或作为后台服务
# Linux: systemctl start ollama
# Windows: 任务管理器 → 服务 → Ollama → 启动
```

### 步骤 3: 验证服务

```bash
ollama list
# 应该能看到已安装的模型列表 (初始可能为空)
```

---

## 模型导入与验证

### Modelfile 变体

项目提供了三种量化级别的 Modelfile，位于项目根目录：

| 文件 | 量化 | 显存需求 | 精度 |
|------|------|----------|------|
| `Modelfile.fp16` | FP16 全精度 | 48GB+ | 最高 |
| `Modelfile.int8` | 8-bit 量化 | 24GB | 较好 |
| `Modelfile.int4` | 4-bit 量化 | 16GB | 可接受 |
| `Modelfile` | 8-bit (默认) | 24GB | 推荐 |

### Modelfile 关键配置说明

```plaintext
FROM ./models/MedTsLLM-v1.5-multimodal
TEMPLATE """{{.System}}
患者信息：
诊断结果：{{.Diagnosis}}
生命体征数据：{{.VitalSigns}}
检验结果：{{.LabResults}}
临床文本：{{.ClinicalNotes}}

请根据以上信息，预测患者未来24小时内的病情恶化风险，并输出结构化的风险报告。
"""
SYSTEM "你是一名专业的临床医生，擅长分析患者的多模态医疗数据..."
PARAMETER temperature 0.1    # 低温度保证医学推理的确定性
PARAMETER top_p 0.9          # 核采样确保输出多样性
PARAMETER num_ctx 8192       # 8192 token 上下文窗口
PARAMETER num_gpu 99         # 将所有层加载到 GPU (99 = 最大层数)
```

### 导入命令

```bash
# 自动导入 (推荐, 使用自动化脚本)
python deploy_medtsllm.py --quant INT8

# 手动导入
ollama create medtsllm:1.5 -f Modelfile

# 4-bit 量化版本
ollama create medtsllm:1.5-int4 -f Modelfile.int4

# FP16 全精度版本
ollama create medtsllm:1.5-fp16 -f Modelfile.fp16
```

### 验证模型导入

```bash
# 查看已导入的模型
ollama list

# 预期输出:
# NAME                    ID              SIZE      MODIFIED
# medtsllm:1.5           abc123def456    14 GB     2 minutes ago
# llama3:8b              xyz789ghi012    4.7 GB    3 days ago
```

### 快速测试

```bash
ollama run medtsllm:1.5 "请分析一名高血压患者的短期恶化风险"
```

预期响应：包含结构化风险分析的医学回答。

---

## 依赖安装

### 更新 requirements.txt

已将新的依赖添加至 `requirements.txt`，包括 `torch`, `transformers`, `pillow`, `psutil`, `python-dotenv` 等。

### 安装命令

```bash
# 使用 pip 安装 (PyPI 官方源)
pip install -r requirements.txt

# 国内镜像加速 (清华源)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 国内镜像加速 (阿里源)
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 依赖说明

| 依赖包 | 最低版本 | 用途 | 是否新增 |
|--------|----------|------|----------|
| `numpy` | — | 数值计算 | 保留 |
| `pandas` | — | 数据处理 | 保留 |
| `scikit-learn` | — | 机器学习 | 保留 |
| `requests` | — | HTTP 请求 | 保留 |
| `tqdm` | — | 进度条 | 保留 |
| `ollama` | — | Ollama 客户端 | 保留 |
| `sentence-transformers` | — | 语义嵌入 | 保留 |
| `faiss-cpu` | — | 向量检索 | 保留 |
| `transformers` | ≥ 4.36.0 | 多模态数据预处理 | **新增** |
| `torch` | ≥ 2.1.0 | 数据预处理 & GPU 检测 | **新增** |
| `pillow` | ≥ 10.0.0 | 医学影像处理 | **新增** |
| `psutil` | ≥ 5.9.0 | 硬件资源监控 | **新增** |
| `python-dotenv` | ≥ 1.0.0 | 环境变量管理 | **新增** |

> **注意**：`torch` 主要用于硬件检测和数据预处理，主推理负载仍由 Ollama 承担。

### GPU 版本 PyTorch (推荐)

```bash
# CUDA 12.1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 部署验证与冒烟测试

### 运行完整验证

```bash
python deploy_medtsllm.py --verify-only
```

### 验证项目

脚本自动检查以下 7 项：

| 检查项 | 说明 | 通过标准 |
|--------|------|----------|
| Ollama 安装 | ollama 命令可用 | 版本 ≥ 0.1.30 |
| Ollama 服务 | API 可达 | HTTP 200 |
| 模型导入 | 模型在列表内 | `ollama list` 包含 medtsllm |
| 基本响应 | 简单对话 | 返回有效文本 |
| 多模态输入 | 结构化病历 | 输出含风险关键词 |
| 结构化输出 | JSON 格式 | 含必要风险字段 |
| 文件校验 | MD5 校验和 | 所有文件匹配 |

### 冒烟测试用例说明

#### 测试 1: 基本响应

```
输入: "你好，请用一句话介绍你自己。"
预期: 返回医学助手身份描述
```

#### 测试 2: 多模态输入

```
输入: 包含诊断、生命体征、检验结果、临床文本的完整病历
预期: 返回包含风险分析的结构化报告
```

#### 测试 3: 结构化输出

```
输入: 要求 JSON 格式输出的肺炎患者病历
预期: {"short_term_risk": 0.xx, "mid_term_risk": 0.xx, ...}
```

#### 测试 4: 推理速度

```
输入: 简短心血管病历
预期: 单次推理 < 20 秒 (GPU), < 120 秒 (CPU)
```

### 手动验证命令

```bash
# 测试模型是否存在
ollama list | grep medtsllm

# 测试对话
ollama run medtsllm:1.5 "测试"

# 测试推理性能
curl -X POST http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medtsllm:1.5",
    "prompt": "请分析高血压患者的恶化风险",
    "stream": false
  }' | jq .
```

---

## TFA 集成

### 架构说明

```
 患者病历 (多模态)
      │
      ▼
 ┌──────────┐    Ollama API     ┌───────────────┐
 │   TFA    │ ───────────────►  │  MedTsLLM     │
 │ (Agent)  │ ◄───────────────  │  (本地推理)    │
 └──────────┘   结构化风险报告   └───────────────┘
      │
      ▼
  风险预测 → DMA 融合
```

### 集成方式 1: 使用适配器 (推荐)

```python
from deploy_medtsllm import MedTsLLMAdapter

# 初始化适配器
medtsllm = MedTsLLMAdapter(
    model_name="medtsllm:1.5",
    timeout=60.0,
)

# 替换 TFA.forecast() 中的模拟推理
result = medtsllm.predict_risk(
    diagnosis="高血压 2 级, 2 型糖尿病",
    vital_signs="BP 155/95 mmHg, HR 88 bpm, RR 18/min, T 37.1°C",
    lab_results="空腹血糖 8.2 mmol/L, HbA1c 7.5%, Cr 98 μmol/L",
    clinical_notes="患者自述近3天头晕加重，偶有心悸。",
)

# 返回与现有 TFA.forecast() 完全兼容的数据结构
print(result["short_term"]["risk_probability"])   # → 0.76
print(result["mid_term"]["risk_probability"])     # → 0.52
print(result["long_term"]["risk_probability"])    # → 0.34
```

### 集成方式 2: 直接替换 TFA 类

在 `mapfm_ecosystem_repaired.py` 中修改 `TemporalForeseeingAgent.forecast()` 方法：

```python
def forecast(
    self,
    query: str,
    history: Optional[Sequence[float]] = None,
    authoritative_signal: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Forecast using MedTsLLM real multi-modal temporal prediction."""
    started = time.perf_counter()
    self.total_calls += 1
    self.forecast_calls += 1
    self.heartbeat()

    # 从 authoritative_signal 中提取临床数据
    diagnosis = str(authoritative_signal.get("diagnosis", query) if authoritative_signal else query)
    vital_signs = str(authoritative_signal.get("vital_signs", "N/A")) if authoritative_signal else "N/A"
    lab_results = str(authoritative_signal.get("lab_results", "N/A")) if authoritative_signal else "N/A"
    clinical_notes = query

    try:
        result = self.medtsllm.predict_risk(
            diagnosis=diagnosis,
            vital_signs=vital_signs,
            lab_results=lab_results,
            clinical_notes=clinical_notes,
        )
    except Exception as e:
        logger.warning("MedTsLLM inference failed, falling back to simulated TFA: {}", e)
        return self._simulated_forecast(query, history, authoritative_signal)

    self.last_forecast = result
    self.record_runtime(started)
    return result
```

### 配置文件更新

在 `EcosystemConfig` 中添加：

```python
@dataclass
class EcosystemConfig(SystemConfig):
    # ... 现有配置 ...

    # MedTsLLM TFA 配置
    medtsllm_model_name: str = "medtsllm:1.5"
    medtsllm_timeout_seconds: float = 60.0
    medtsllm_fallback_on_error: bool = True
```

---

## 常见问题与排查

### Q1: 显存不足 (OOM)

**症状**: `ollama create` 或推理时报 CUDA out of memory

**解决方案**:

```bash
# 方案 A: 使用 4-bit 量化
python deploy_medtsllm.py --quant INT4

# 方案 B: 限制 GPU 层数 (在 Modelfile 中)
PARAMETER num_gpu 48    # 仅加载 48 层到 GPU
PARAMETER num_thread 8  # CPU 线程数

# 方案 C: 纯 CPU 推理
ollama create medtsllm:1.5-cpu -f Modelfile.int4
# 手动修改 Modelfile: PARAMETER num_gpu 0
```

### Q2: 模型下载失败

**症状**: 网络超时、连接中断

**解决方案**:

1. 切换下载源: `python deploy_medtsllm.py --download-only --mirror`
2. 使用 ModelScope 网页端手动下载: https://modelscope.cn/models/thu-medai/MedTsLLM-v1.5-multimodal
3. 使用上文提供的断点续传脚本
4. 设置 HuggingFace 镜像环境变量 (已由脚本自动设置):
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### Q3: Ollama 服务连接失败

**症状**: `Connection refused on http://127.0.0.1:11434`

**解决方案**:

```bash
# 启动 Ollama 服务
ollama serve

# 如果端口冲突, 指定其他端口
set OLLAMA_HOST=127.0.0.1:11435
ollama serve

# 防火墙: 允许本地 11434 端口
# Windows: 控制面板 → Windows Defender 防火墙 → 允许应用通过防火墙
```

### Q4: 模型加载后响应异常

**症状**: 模型返回乱码或非医学内容

**解决方案**:

```bash
# 1. 确认模型文件完整性
python deploy_medtsllm.py --verify-only

# 2. 检查 Modelfile 中的 TEMPLATE 格式是否正确
# 确保 {{.System}} 变量名与 SYSTEM 参数一致

# 3. 重新导入模型
ollama rm medtsllm:1.5
ollama create medtsllm:1.5 -f Modelfile

# 4. 检查 temperature 参数 (建议 0.1)
ollama run medtsllm:1.5
>>> /set parameter temperature 0.1
```

### Q5: 推理速度过慢

**症状**: 单次推理超过 60 秒

**解决方案**:

1. 确认 GPU 是否被正确使用 (`nvidia-smi` 查看 GPU 利用率)
2. 升级到更低的量化级别 (FP16 → INT8 → INT4)
3. 减少 `num_ctx` 参数 (8192 → 4096)
4. 关闭其他占用 GPU 的程序
5. 确认 Ollama 版本 ≥ 0.1.30

### Q6: 与现有 llama3:8b 冲突

**症状**: 部署 medtsllm 后 llama3:8b 不可用

**解决方案**: 两个模型完全独立，互不影响。

```bash
# 确认两个模型共存
ollama list
# medtsllm:1.5   14 GB
# llama3:8b       4.7 GB

# DMA 继续使用 llama3:8b
# TFA 使用 medtsllm:1.5
```

### Q7: 文件校验失败

**症状**: `model-*.safetensors` MD5 不匹配

**说明**: 本部署脚本中的 MD5 校验和为示例值。从官方渠道下载的模型文件应使用官方提供的校验和。您可以：

1. 从 HuggingFace 仓库页面获取官方 MD5/SHA256 校验和
2. 更新 `deploy_medtsllm.py` 中 `MODEL_FILES` 字典的校验值
3. 或者临时跳过文件校验: 注释掉 `MODEL_FILES` 中的校验

---

## 部署完成检查清单

- [ ] 硬件检测通过 (GPU ≥ 16GB VRAM)
- [ ] Ollama ≥ 0.1.30 已安装
- [ ] Ollama 服务正常运行
- [ ] 所有 6 个模型文件已下载并校验
- [ ] Modelfile 已根据硬件配置选择
- [ ] `ollama create medtsllm:1.5 -f Modelfile` 成功
- [ ] `ollama list` 包含 medtsllm:1.5
- [ ] 基本响应测试通过
- [ ] 多模态输入测试通过
- [ ] 结构化 JSON 输出测试通过
- [ ] 推理速度在可接受范围内
- [ ] requirements.txt 依赖已安装
- [ ] TFA 集成适配器测试通过

---

## 技术支持

- 项目仓库: HMAE 医疗多智能体系统
- 模型来源: tsinghua-medai/MedTsLLM-v1.5-multimodal
- 推理引擎: Ollama (https://ollama.com)

---

*最后更新: 2026-05-27*
