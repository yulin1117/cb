from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import enchant

# ==========================================
# ⚙️ Configuration & Paths
# ==========================================
INPUT_CSV_PATH = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/openalex/papers.csv"
COMPARE_FILE = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/openalex/gold_check.csv"

OUT_MD_FILE = "detect_html.md"
OUT_CSV_FILE = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/openalex/html_detect.csv"

# 初始化英文字典 (US English)
dictionary = enchant.Dict("en_US")

# ==========================================
# 🧠 Rule-based Filter Functions
# ==========================================
def check_html(text: str) -> bool:
    """
    判斷字串是否含有 HTML 標籤或未處置的 HTML 實體
    """
    if not isinstance(text, str) or not text:
        return False
        
    # 判斷是否含有真正的 HTML 標籤 (如 <p>, <i>, <sub> 等)
    has_tag = bool(BeautifulSoup(text, "html.parser").find())
    
    # 判斷是否含有 HTML 實體編碼 (例如 &amp;, &lt;, &gt;)
    has_entity = "&" in text and ";" in text
    
    return has_tag or has_entity

def check_broken_word(text: str, max_allowed_broken: int = 2) -> bool:
    """
    利用字典判斷字串中是否含有破損連字 (如 com-press -> compress)
    """
    if not isinstance(text, str) or not text:
        return False
    
    normalized_text = re.sub(r'\u00AD\s*', '-', text)
    # 抓出所有由連字號連接的單字 (如 com-press)
    matches = re.findall(r'\b[a-zA-Z]{2,}-\s*[a-zA-Z]{2,}\b', normalized_text)
    
    broken_count = 0
    for match in matches:
        # 如果原始字串（例如 "ques- tionnaire"）不在字典裡
        if not dictionary.check(match):
            # 移除連字號與所有空格 (ques- tionnaire -> questionnaire)
            joined_word = re.sub(r'-\s*', '', match)
            
            # 若去除連字號與空格後，變成了合法的英文單字，代表這是排版破損字！
            if dictionary.check(joined_word):
                broken_count += 1
                
    # 若一篇論文中出現超過指定數量的破損連字，判定為格式噪音
    return broken_count >= max_allowed_broken
def check_missing_char_unicode(text: str) -> bool:
    """
    偵測文字中是否含有 Unicode Private Use Area (PUA) 缺字/亂碼字元
    範圍：\uE000-\uF8FF (包含 , ,  等豆腐塊/亂碼) 以及 '\ufffd' ()
    """
    if not isinstance(text, str) or not text:
        return False
        
    # 匹配 PUA 區段字元與 Replacement Character ()
    pua_pattern = r'[\uE000-\uF8FF\uFFFD]'
    
    matches = re.findall(pua_pattern, text)
    return len(matches) > 0  # 一旦出現即代表字體對映損壞

# ==========================================
# 🚀 Main Execution & Evaluation
# ==========================================
def main():
    print("=" * 65)
    print("🔍 Evaluating Rule-based Filter vs LLM Gold Check")
    print("=" * 65)

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input file missing: {INPUT_CSV_PATH}")
    if not os.path.exists(COMPARE_FILE):
        raise FileNotFoundError(f"Compare file missing: {COMPARE_FILE}")
    
    # 1. 讀取資料
    df_input = pd.read_csv(INPUT_CSV_PATH)
    df_compare = pd.read_csv(COMPARE_FILE)
    
    # 統一 ID 欄位以便 Merge (對齊 work_url)
    df_input["work_url"] = df_input.get("id", df_input.get("work_url")).astype(str).str.strip()
    df_compare["work_url"] = df_compare["work_url"].astype(str).str.strip()
    
    # 將 LLM 的判定轉為 boolean (True 代表 LLM 說這篇有 quality_issue)
    df_compare["llm_has_noise"] = df_compare["quality_issue"].astype(str).str.upper() == "TRUE"
    
    # 合併兩份資料
    merged_df = pd.merge(
        df_input, 
        df_compare[["work_url", "llm_has_noise", "quality_detail"]], 
        on="work_url", 
        how="inner"
    )
    total_eval = len(merged_df)
    
    print(f"📊 Loaded {len(df_input)} raw papers.")
    print(f"📊 Merged {total_eval} papers with LLM Gold Check for evaluation.\n")

    if total_eval == 0:
        print("⚠️ No matching papers found between input and compare files!")
        return

    # 2. 逐篇檢測並記錄
    results = []
    tp = tn = fp = fn = 0
    fp_examples = []
    fn_examples = []

    for idx, row in merged_df.iterrows():
        work_id = row["work_url"]
        field = str(row.get("field", "Unknown")).strip()
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        
        text = f"{title}. {abstract}"
        
        # 執行規則檢測
        has_html_noise = check_html(text)
        has_broken_noise = check_broken_word(text, max_allowed_broken=2)
        has_missing_char_noise = check_missing_char_unicode(text)
        # Rule 判定是否有雜訊
        rule_has_noise = has_html_noise or has_broken_noise or has_missing_char_noise
        rule_pass = not rule_has_noise
        
        llm_has_noise = row["llm_has_noise"]
        
        # 統計混淆矩陣
        if rule_has_noise and llm_has_noise:
            tp += 1
            match_status = "True Positive (Both caught noise)"
        elif not rule_has_noise and not llm_has_noise:
            tn += 1
            match_status = "True Negative (Both clean)"
        elif rule_has_noise and not llm_has_noise:
            fp += 1
            match_status = "False Positive (Rule overly strict)"
            if len(fp_examples) < 5: fp_examples.append(work_id)
        else:
            fn += 1
            match_status = "False Negative (Rule missed noise)"
            if len(fn_examples) < 5: fn_examples.append((work_id, row["quality_detail"]))

        # 儲存每篇論文的結果
        results.append({
            "field": field,
            "work_id": work_id,
            "has_html": has_html_noise,
            "has_broken_word": has_broken_noise,
            "has_missing_char": has_missing_char_noise,
            "pass": rule_pass,
            "rule_has_noise": rule_has_noise,
            "llm_has_noise": llm_has_noise,
            "match_status": match_status,
            "llm_quality_detail": row.get("quality_detail", "")
        })

    # 3. 匯出完整結果至 CSV 供手動抽查
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV_FILE), exist_ok=True)
    df_results.to_csv(OUT_CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"💾 Per-paper audit results saved to CSV: {OUT_CSV_FILE}")

    # 4. 計算評估指標並輸出 Markdown 報告
    accuracy = (tp + tn) / total_eval * 100
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0

    md_content = f"""# 📊 Rule-based vs LLM Quality Filter Evaluation

- **Total Evaluated Papers**: {total_eval}
- **Accuracy (一致率)**: **{accuracy:.2f}%**
- **Precision (精準度)**: {precision:.2f}%
- **Recall (召回率)**: {recall:.2f}%

## 🧮 Confusion Matrix (混淆矩陣)

| | LLM 判定為雜訊 (Noise) | LLM 判定為乾淨 (Clean) |
|---|:---:|:---:|
| **Rule 判定為雜訊** | **TP**: {tp} (成功攔截) | **FP**: {fp} (Rule 誤殺) |
| **Rule 判定為乾淨** | **FN**: {fn} (Rule 漏網) | **TN**: {tn} (雙方皆認可) |

### 🔍 錯誤分析 (Error Diagnostics)

**🔴 False Negatives (Rule 沒抓到，但 LLM 說有雜訊的漏網之魚 Top 5):**
"""
    for ex in fn_examples:
        md_content += f"- Work URL: `{ex[0]}` | LLM 原因: {ex[1]}\n"

    md_content += "\n**🟡 False Positives (Rule 判斷是雜訊，但 LLM 說沒問題的誤殺 Top 5):**\n"
    for ex in fp_examples:
        md_content += f"- Work URL: `{ex}`\n"

    os.makedirs(os.path.dirname(os.path.abspath(OUT_MD_FILE)) or ".", exist_ok=True)
    with open(OUT_MD_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✅ Evaluation summary saved to MD: {OUT_MD_FILE}")
    print(f"📈 Overall Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()