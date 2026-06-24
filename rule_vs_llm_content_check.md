# Ablation Study Report: LLM vs. Non-LLM Pipeline
This report presents a pairwise comparison between the **LLM-Based Pipeline** and the **Non-LLM Rule-Based Pipeline** (integrated with FastText & Stopwords Radar heuristics) on OpenAlex scholarly documents.

---

## 1. Pipeline Dataset Statistics
- **Total Records (LLM Pipeline)**: 3,695
- **Total Records (Rule-Based Pipeline)**: 3,695
- **Successfully Aligned Pairwise Records**: 3,695

---

## 2. Pass Rate & Strictness Comparison
| Cleaning Pipeline | Approved Papers | Rejected Papers | Pass Rate (%) |
| :--- | :---: | :---: | :---: |
| **🤖 LLM Pipeline** | 1255 | 2440 | 33.96% |
| **⚡ Non-LLM Rule-Based** | 1128 | 2567 | 30.53% |

### Strictness Analysis
> **Conclusion:** The [Non-LLM Rule-Based Pipeline] is MORE strict (lower acceptance rate) than the LLM counterpart.

---

## 3. Aligned Confusion Matrix
Using the **LLM Pipeline** as the reference standard (Ground Truth), the classification matrix of the **Rule-Based Pipeline** is detailed below:

| | Rule-Based APPROVED (Predicted Positive) | Rule-Based REJECTED (Predicted Negative) |
| :--- | :---: | :---: |
| **LLM APPROVED (Actual Positive)** | **944** <br>*(True Consensus / APPROVED in Both)* | **311** <br>*(Overkill / False Negative)* <br>👉 See `rule_reject_llm_approved.csv` |
| **LLM REJECTED (Actual Negative)** | **184** <br>*(Leakage / False Positive)* <br>👉 See `rule_approved_llm_reject.csv` | **2256** <br>*(False Consensus / REJECTED in Both)* |

### Summary Statistic
- **Pairwise Consensus Rate**: **86.60%**

---

## 4. Extreme Diagnostic Case Inspection (Samples)

### 🚨 Case A: Pipeline Leakage (FP Count: 184)
*These are documents rejected by LLM's deep semantic understanding but passed by the physical heuristic rules. Full list in `rule_approved_llm_reject.csv`.*
- **ID**: `https://openalex.org/W3089888409` | **Title**: KLASIFIKASI CITRA DIGITAL BUMBU DAN REMPAH DENGAN ALGORITMA CONVOLUTIONAL NEURAL NETWORK (CNN)
- **ID**: `https://openalex.org/W2619507781` | **Title**: ANALISA PERBANDINGAN METODE FILTER GAUSSIAN, MEAN DAN MEDIAN TERHADAP REDUKSI NOISE
- **ID**: `https://openalex.org/W3197728515` | **Title**: Literature Study of Convolutional Neural Network Algorithm for Batik Classification
- **ID**: `https://openalex.org/W4285341955` | **Title**: Eksperimen Penerapan Sistem Traffic Counting dengan Algoritma YOLO (You Only Look Once) V.4.
- **ID**: `https://openalex.org/W4236491321` | **Title**: Learnart : Drawing Environment using Convolutional Neural Networks

### 🎯 Case B: Pipeline Overkill (FN Count: 311)
*These are documents marked as APPROVED by LLM, but prematurely filtered out by the Rule-Based Pipeline. Full list in `rule_reject_llm_approved.csv`.*
- **ID**: `https://openalex.org/W2893344297` | **Title**: Klasifikasi Daun Dengan Perbaikan Fitur Citra Menggunakan Metode K-Nearest Neighbor
- **ID**: `https://openalex.org/W3217734993` | **Title**: Klasifikasi Penyakit Daun Padi Menggunakan Convolutional Neural Network
- **ID**: `https://openalex.org/W3126558687` | **Title**: Klasifikasi Penyakit Tanaman Padi Menggunakan Model Deep Learning Efficientnet B3 dengan Transfer Learning
- **ID**: `https://openalex.org/W3038165747` | **Title**: Klasifikasi Kematangan Stroberi Berbasis Segmentasi Warna dengan Metode HSV
- **ID**: `https://openalex.org/W3157439671` | **Title**: Low resource deep learning to detect waste intensity in the river flow

---
*Report dynamically generated via compare_reports.py.*
