# 📊 Rule-based vs LLM Quality Filter Evaluation

- **Total Evaluated Papers**: 1294
- **Accuracy (一致率)**: **84.23%**
- **Precision (精準度)**: 91.28%
- **Recall (召回率)**: 45.38%

## 🧮 Confusion Matrix (混淆矩陣)

| | LLM 判定為雜訊 (Noise) | LLM 判定為乾淨 (Clean) |
|---|:---:|:---:|
| **Rule 判定為雜訊** | **TP**: 157 (成功攔截) | **FP**: 15 (Rule 誤殺) |
| **Rule 判定為乾淨** | **FN**: 189 (Rule 漏網) | **TN**: 933 (雙方皆認可) |

### 🔍 錯誤分析 (Error Diagnostics)

**🔴 False Negatives (Rule 沒抓到，但 LLM 說有雜訊的漏網之魚 Top 5):**
- Work URL: `W3004842646` | LLM 原因: Contains a double period after 'academic year..' and a malformed parenthesis/percent sequence 'fruit (22.1%). %)' in the results sentence.
- Work URL: `W4252609127` | LLM 原因: Spelling errors ('canne' for cane, 'patter' for pattern) and a malformed statistic 'vegetables 1,33±;' with missing standard deviation and stray semicolon.
- Work URL: `W4393327428` | LLM 原因: The meal list contains duplicated entries: 'breakfast, breakfast, lunch, lunch, dinner and supper'.
- Work URL: `W4210653435` | LLM 原因: Stray leading character 'O' before 'Objective:' in the abstract.
- Work URL: `W2441013479` | LLM 原因: Title is severely corrupted, repeating truncated fragments such as DETERMINA and TION OF THE USERS' PROFILE AND OF THE CHEMICAL AND NUTRITIONAL multiple times, indicating duplication/garbling.

**🟡 False Positives (Rule 判斷是雜訊，但 LLM 說沒問題的誤殺 Top 5):**
- Work URL: `W2022728701`
- Work URL: `W7144644309`
- Work URL: `W2355359607`
- Work URL: `W640606391`
- Work URL: `W2565543990`
