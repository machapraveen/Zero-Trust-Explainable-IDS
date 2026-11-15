# Explainable Deep Learning Framework for Zero-Trust Intrusion Detection Using Multi-Level Network Traffic Analytics

**Institution:** Aditya College of Engineering and Technology  
**Department:** Computer Science and Engineering (Artificial Intelligence & Machine Learning — CSE-AIML)

---

## Project Team & Supervision

### Project Team Members
| S.No | Name | Roll Number |
| :---: | :--- | :--- |
| 1 | **P. Dwaraka Ushashwini** | `23MH1A42J4` |
| 2 | **G. Ramya** | `23MH1A42E4` |
| 3 | **K. Kavya** | `23MH1A42F3` |
| 4 | **M. T. Ravi Kishore** | `23MH1A42G6` |

### Project Supervision / Guide
- **Ms. S. Saranya**, *M.Tech*  
  Assistant Professor, Department of CSE - AIML  
  Aditya College of Engineering and Technology

---

## Overview

Modern enterprise and edge networks face sophisticated, multi-stage cyber threats where traditional perimeter-based intrusion detection systems suffer from high false-positive rates and opaque ("black-box") deep learning predictions.

**ZT-XAIIDS** implements an end-to-end **Explainable Deep Learning Framework for Zero-Trust Intrusion Detection Using Multi-Level Network Traffic Analytics** under the core Zero-Trust mandate (*"Never Trust, Always Verify"*):

1. **Stage 1 — Multi-Level Network Traffic Input:** Extracts 24 behavioral features spanning **L1 Packet-Level** (TCP flags, packet size distribution, payload entropy), **L2 Flow-Level** (duration, throughput, inter-arrival timing), and **L3 Session / Zero-Trust / Edge-IoT** (failed authentication rate, east-west lateral movement, device posture, and Edge/IoT drift).
2. **Stage 2 — Leak-Free Preprocessing & Feature Extraction:** Performs strict train-split median imputation and `StandardScaler` normalization without data leakage.
3. **Stage 3 — 10-Model Deep Learning & ML Classification:** Combines **6 PyTorch Deep Learning architectures** (`1D-CNN`, `Bi-LSTM Temporal Attention`, `Transformer Self-Attention`, `Graph Neural Network`, `Deep Residual MLP`, `Zero-Day Reconstruction Autoencoder`) with **4 Ensemble ML detectors** (`XGBoost`, `Random Forest`, `Extra-Trees`, `Isolation Forest`).
4. **Stage 4 — Explainability Module (SHAP Interpretability Layer):** Uses `shap.TreeExplainer` to generate per-feature SHAP attributions, L1/L2/L3 multi-level contribution percentages, and human-readable SOC analyst explanations.
5. **Stage 5 — Zero-Trust Decision Engine:** Computes a continuous **Entity Trust Score (`0–100`)**, maps detections to **MITRE ATT&CK**, and triggers deterministic micro-segmentation and enforcement policies (`ALLOW_CONTINUOUS_VERIFY`, `STEP_UP_MFA_AND_INSPECT`, `RATE_LIMIT_AND_DPI`, `MICRO_SEGMENT_QUARANTINE`, `ZERO_TRUST_BLOCK_AND_ISOLATE`).

---

## Architecture Pipeline

```text
Multi-Level Network Traffic (L1 Packet + L2 Flow + L3 Session/IoT)
        │
        ▼
Preprocessing & 24-Feature Extraction (Median Imputer + StandardScaler)
        │
        ▼
10-Model Deep Learning & ML Ensemble (6 PyTorch DL + 4 ML Models)
        │
        ▼
Explainability Layer (TreeSHAP Feature & Multi-Level Attribution)
        │
        ▼
Zero-Trust Decision Engine (Entity Trust Score + Automated Policy Enforcement)
```

---

## Quick Start

### Option 1: Docker Deployment (Recommended)
```powershell
docker compose up --build -d
docker exec ztxai-app python -m utils.seed_data --clear --flows 1500 --alerts 150
```
- **SOC Dashboard & Zero-Trust Lab:** http://localhost (or http://localhost:5000)
- **Zero-Trust Live XAI Lab:** http://localhost/zero-trust
- **Default Login:** `admin` / `admin123` (or `demo` / `demo123`)

### Option 2: Local Python Virtual Environment
```powershell
.\.venv\Scripts\python.exe -m utils.seed_data --clear --flows 1500 --alerts 150
.\.venv\Scripts\python.exe run.py
```

### Verification Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_zero_trust_e2e.py -v
```
