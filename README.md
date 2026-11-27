# Deepfake Detection & Three-Zone Content Moderation Triage

**Author:** Sai Nithin Battu  
**Institution:** Teesside University | MSc Artificial Intelligence with Advanced Practice  
**Supervisor:** Prof. Claudio Angione  
**Ethics Clearance Ref:** `2026 Oct 36071 BATTU` (SCEDT Research Ethics sub-Committee, 19 October 2026)

## Project Overview

This repository contains the experimental notebook (`Deep_fake_Detector_code.ipynb`) and interactive Streamlit inference prototype (`app.py`) for an MSc research paper investigating cross-dataset generalisation in deepfake detection and human-in-the-loop content moderation workflows under UK regulatory mandates (Online Safety Act 2023 and Crime and Policing Act 2026).

Across a four-stage progression—from single-domain `EfficientNet-B4` and multi-source `EfficientNet-B4 + SAM` with forensic augmentations to frozen `CLIP ViT-L/14` linear probes—our final **Multi-Layer Frozen CLIP ViT-L/14 + Horizontal-Flip TTA** model (fusing layers 12, 16, 20, layer-20 spatial patch mean, and layer 24) clears the 70% cross-dataset generalisation threshold across all evaluated benchmarks.

## Benchmark Results (4-Stage Progression)

| Benchmark | Model 1: EffNet-B4 (140k) | Model 2: EffNet-B4 + SAM | Model 3: CLIP (Final Layer) | Model 4: Multi-Layer CLIP + TTA |
| :--- | :---: | :---: | :---: | :---: |
| **140k Official Test** *(In-Domain)* | 95.67% (0.9942) | 92.05% (0.9781) | 98.03% (0.9979) | **99.37% (0.9995)** |
| **FaceForensics++ C23** *(Video-Safe)* | 51.00% (0.5127) | 56.65% (0.6079) | 75.33% (0.8384) | **82.27% (0.9171)** |
| **CIPLab Faces** *(Unseen External)* | 51.74% (0.4670) | 52.33% (0.4932) | 82.70% (0.8895) | **85.15% (0.9183)** |
| **Celeb-DF v2 Frame-Level** *(Unseen)* | 53.40% (0.5767) | 65.20% (0.7112) | 66.53% (0.7833) | **73.10% (0.8197)** |
| **Celeb-DF v2 Video-Level** *(n = 2,045)* | — | — | — | **70.71% (0.8259)** |

## Three-Zone Content Moderation Triage (Model 4)

- **Zone 1 — Auto-Flag / Quarantine:** `P(Real) < 0.30`
- **Zone 2 — Human Moderator Triage Queue:** `0.30 <= P(Real) <= 0.70`
- **Zone 3 — Auto-Approve / Pass:** `P(Real) > 0.70`

| Benchmark | Auto-Resolved | Automated Accuracy | Human Review Queue |
| :--- | :---: | :---: | :---: |
| **140k Official Test** | 98.1% | 99.69% | 1.9% |
| **FaceForensics++ C23 (Video-Safe)** | 71.4% | 90.47% | 28.6% |
| **CIPLab Faces (Unseen External)** | 92.3% | 87.10% | 7.7% |
| **Celeb-DF v2 (Unseen External)** | 63.2% | 81.60% | 36.8% |

## Running the Streamlit Prototype

```bash
pip install -r requirements.txt
# Place clip_vit_l14_multilayer_head.pth in weights/ (or specify its path in the app sidebar)
streamlit run app.py
