# Structural Semantic Alignment Diagnostics Report
This validation assessment traces the behavioral alignment between **LLM Semantic Filtering** and **Bi-Encoder Vector Similarities** applied explicitly to documents flagged as mismatched (`is_matching=False`).

---

## 1. Threshold Benchmark Metrics
- **Target Calibration Dataset Size**: 213 papers
- **Evaluated Vector Pipeline Threshold**: `0.5`
- **Identical Pipeline Decisions (Consensus)**: 184 papers
- **Decision Alignment Rate (Consensus Rate)**: **86.38%**

---

## 2. Decision Distribution Matrix
| Evaluation Profile | Measurement Count | System Operational Meaning |
| :--- | :---: | :--- |
| **True Negative Alignment** | 184 | Both LLM and Vector Pipeline agree the paper is a mismatch (Score < 0.5). |
| **System Leakage Vulnerability** | 29 | LLM rejected the paper, but the Vector Pipeline passed it (Score >= 0.5). |

---

## 3. Heuristic Threshold Calibration Insight
### Current Observation
- Out of 213 documents that the LLM explicitly flagged as garbage or out-of-domain noise, the vector evaluation pipeline confirmed **86.38%** of them using a threshold value of `0.5`.

> ⚠️ **Recommendation:** The current threshold of `0.5` might be too lenient, allowing 29 noisy records to slip through. The average score of these leaked items is **0.6074**. Consider bumping the threshold parameter up to **0.66** to tighten down structural security boundaries.

---
*Report dynamically generated via check_semantic_threshold.py.*
