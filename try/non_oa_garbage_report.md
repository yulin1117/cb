# Diagnostic Scan: Non-OA Paywall & Semantic Garbage Audit
This analytical scanner parses scientific works marked as **strictly non-open-access (`open_access.is_oa = false`)** on OpenAlex to audit structural paywall noise and title-abstract mismatches.

---

## 📊 Core Diagnostic Metrics
- **Target Topic**: `T13674`
- **Total Non-OA Works Audited**: 1000 papers
- **Semantic Similarity Threshold**: `0.5`
- **Total Garbage Content Identified**: **315 papers**
- **🎯 Overall Garbage Content Ratio**: **31.50%**

---

## 🔍 Structural Failure Vectors
| Failure Category | Triggered Count | Percentage | Operational Threat |
| :--- | :---: | :---: | :--- |
| **Heuristic Garbage** (Paywalls, Socials, Widgets) | 14 | 1.40% | Corrupts clustering and poisons downstream LLM/TopicGPT labels. |
| **Semantic Mismatch** (Similarity < 0.5) | 301 | 30.10% | Distorts topic boundary consistency. |

---

## 💡 Heuristic Diagnostic Conclusions
### Why Non-OA Papers Suffer High Garbage Rates
Under the `is_oa = false` restriction, publisher sites frequently block OpenAlex web crawlers. Consequently, rather than a genuine abstract, the `abstract_inverted_index` fields are loaded with:
1. **Redirection Residuals** (e.g., *"This content is only available as a PDF"*, *"You do not currently have access"*).
2. **Citation Widget Captures** (e.g., *"Download citation file: Ris (Zotero) Reference Manager EasyBib"*).
3. **Journal Metadata Overflow** (Page counters, online ISSNs, editor headers).

---
*Report dynamically generated via non_oa_garbage_scanner.py.*
