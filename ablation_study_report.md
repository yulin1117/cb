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
| **⚡ Non-LLM Rule-Based** | 1103 | 2592 | 29.85% |

### Strictness Analysis
> **Conclusion:** The [Non-LLM Rule-Based Pipeline] is MORE strict (lower acceptance rate) than the LLM counterpart.

---

## 3. Aligned Confusion Matrix
Using the **LLM Pipeline** as the reference standard (Ground Truth), the classification matrix of the **Rule-Based Pipeline** is detailed below:

| | Rule-Based APPROVED (Predicted Positive) | Rule-Based REJECTED (Predicted Negative) |
| :--- | :---: | :---: |
| **LLM APPROVED (Actual Positive)** | **971** <br>*(True Consensus / APPROVED in Both)* | **284** <br>*(Overkill / False Negative)* <br>👉 See `rule_reject_llm_approved.csv` |
| **LLM REJECTED (Actual Negative)** | **132** <br>*(Leakage / False Positive)* <br>👉 See `rule_approved_llm_reject.csv` | **2308** <br>*(False Consensus / REJECTED in Both)* |

### Summary Statistic
- **Pairwise Consensus Rate**: **88.74%**

---

## 4. Extreme Diagnostic Case Inspection (Samples)

### 🚨 Case A: Pipeline Leakage (FP Count: 132)
*These are documents rejected by LLM's deep semantic understanding but passed by the physical heuristic rules. Full list in `rule_approved_llm_reject.csv`.*
- **ID**: `https://openalex.org/W3197728515` | **Title**: Literature Study of Convolutional Neural Network Algorithm for Batik Classification
- **ID**: `https://openalex.org/W4236491321` | **Title**: Learnart : Drawing Environment using Convolutional Neural Networks
- **ID**: `https://openalex.org/W4294930474` | **Title**: GLCM Based Locally Feature Extraction On Natural Image
- **ID**: `https://openalex.org/W3016324002` | **Title**: Identification of Madura Tobacco Leaf Disease Using Gray- Level Co-Occurrence Matrix, Color Moments and Naïve Bayes
- **ID**: `https://openalex.org/W3114781889` | **Title**: Fruit Maturity Classification Using Convolutional Neural Networks Method

### 🎯 Case B: Pipeline Overkill (FN Count: 284)
*These are documents marked as APPROVED by LLM, but prematurely filtered out by the Rule-Based Pipeline. Full list in `rule_reject_llm_approved.csv`.*
- **ID**: `https://openalex.org/W2893344297` | **Title**: Klasifikasi Daun Dengan Perbaikan Fitur Citra Menggunakan Metode K-Nearest Neighbor
- **ID**: `https://openalex.org/W3217734993` | **Title**: Klasifikasi Penyakit Daun Padi Menggunakan Convolutional Neural Network
- **ID**: `https://openalex.org/W3126558687` | **Title**: Klasifikasi Penyakit Tanaman Padi Menggunakan Model Deep Learning Efficientnet B3 dengan Transfer Learning
- **ID**: `https://openalex.org/W3038165747` | **Title**: Klasifikasi Kematangan Stroberi Berbasis Segmentasi Warna dengan Metode HSV
- **ID**: `https://openalex.org/W2886263264` | **Title**: VARIASI JUMLAH ELEKTRODA DAN BESAR TEGANGAN DALAM MENURUNKAN KANDUNGAN COD DAN TSS LIMBAH CAIR TEKSTIL DENGAN METODE ELEKTROKOAGULASI

---
*Report dynamically generated via compare_reports.py.*
