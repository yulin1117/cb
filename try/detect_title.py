import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import fasttext

# ==========================================
# ⚙️ Configuration & Paths
# ==========================================
INPUT_CSV_PATH = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/openalex/papers.csv"
OUTPUT_CSV_PATH = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/openalex/lang_detect_exp/lang_detect.csv"
OUTPUT_MD_PATH = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/openalex/subfiles/lang_detect_summary.md"
# FastText 官方預訓練語言模型路徑 (若路徑不同請調整)
FASTTEXT_MODEL_PATH = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/CitationBias/lid.176.bin"

# 確保 NLTK punkt 斷句模型已下載
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# 載入 FastText 語言辨識模型
if not os.path.exists(FASTTEXT_MODEL_PATH):
    # 備用相容：嘗試從當前目錄或預設快取尋找
    alt_path = "lid.176.bin"
    if os.path.exists(alt_path):
        FASTTEXT_MODEL_PATH = alt_path
    else:
        raise FileNotFoundError(f"Cannot find FastText model file 'lid.176.bin' at: {FASTTEXT_MODEL_PATH}")

print(f"📦 Loading FastText model from: {FASTTEXT_MODEL_PATH}...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# ==========================================
# 🧠 Language Detection Core Logic
# ==========================================
def title_lang_detect(text: str, model,threshold: float = 0.7) -> bool:
    """
    Sentence-by-sentence validation using FastText.
    Returns True if non-English sentence ratio < threshold, else False.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if len(text) < 5:
        return False
    (labels, probs) = model.predict(text.strip().replace("\n", " "), k=1)
    
    lang = labels[0].replace("__label__", "")
    confidence = probs[0]  # 這裡拿到的才是真正的 float 機率值 (e.g. 0.85)

    # 1. 判定預測語言是否為英文
    if lang != "en":
        return False
        
    # 2. 判定英文判定信心度是否達到門檻 (例如 >= 0.5)
    if confidence < threshold:
        return False

    return True


def lang_detect(text: str, model, threshold: float = 0.2) -> bool:
    """
    Sentence-by-sentence validation using FastText.
    Returns True if non-English sentence ratio < threshold, else False.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    sentences = sent_tokenize(text)
    valid_sentences_count = 0
    non_eng_sentences = 0  

    for s in sentences:
        s = s.strip().replace("\n", " ")
        if len(s) < 5:
            continue
            
        valid_sentences_count += 1
        labels, _ = model.predict(s, k=1)
        lang = labels[0].replace("__label__", "")

        if lang != "en":
            non_eng_sentences += 1

    # 防禦機制：若完全沒有長度 >= 5 的有效句子，判斷為無效文字
    if valid_sentences_count == 0:
        return False

    # 計算非英文句子比例，大於等於門檻則判斷不通過 (False)
    non_eng_ratio = non_eng_sentences / valid_sentences_count
    return non_eng_ratio < threshold

# ==========================================
# 🚀 Execution Flow
# ==========================================
def main():
    print("=" * 65)
    print("🔍 Testing Language Detection Thresholds Across Fields")
    print(f"📥 Input : {INPUT_CSV_PATH}")
    print(f"📤 CSV   : {OUTPUT_CSV_PATH}")
    print(f"📝 MD    : {OUTPUT_MD_PATH}")
    print("=" * 65)

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input file missing: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)
    total_papers = len(df)
    print(f"📊 Loaded {total_papers} papers for testing.")

    results = []

    for idx, row in df.iterrows():
        field = str(row.get("field", "Unknown")).strip()
        work_id = str(row.get("id", row.get("work_url", f"paper_{idx}"))).strip()
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()

        # 1. Title 檢測 (英文且 Confidence >= 0.5)
        title_eng = title_lang_detect(title, ft_model, threshold=0.5)

        # 2. Abstract 檢測 (非英文句子比例 < 0.2)
        abstract_eng = lang_detect(abstract, ft_model, threshold=0.2)

        results.append({
            "field": field,
            "work_id": work_id,
            "title_eng": title_eng,
            "abstract_eng": abstract_eng,
            "both_pass": title_eng and abstract_eng
        })

    # Save CSV Output
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df[["field", "work_id", "title_eng", "abstract_eng"]].to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"💾 Detailed results saved to: {OUTPUT_CSV_PATH}")

    # ==========================================
    # 📊 Statistical Analysis & MD Report
    # ==========================================
    md_lines = []
    md_lines.append("# 📊 OpenAlex Language Detection Audit Report\n")
    md_lines.append(f"- **Input File**: `{INPUT_CSV_PATH}`")
    md_lines.append(f"- **Total Audited Papers**: `{total_papers}`\n")

    # Overall Metrics
    overall_title_pass = res_df["title_eng"].sum()
    overall_abstract_pass = res_df["abstract_eng"].sum()
    overall_both_pass = res_df["both_pass"].sum()

    md_lines.append("## 📈 Overall Summary Metrics\n")
    md_lines.append(f"- **Overall Title Pass Rate (Conf ≥ 0.5)**: {overall_title_pass}/{total_papers} (**{overall_title_pass/total_papers*100:.2f}%**)")
    md_lines.append(f"- **Overall Abstract Pass Rate (Ratio < 0.2)**: {overall_abstract_pass}/{total_papers} (**{overall_abstract_pass/total_papers*100:.2f}%**)")
    md_lines.append(f"- **🟢 Overall Both Pass Rate (Final Retained)**: {overall_both_pass}/{total_papers} (**{overall_both_pass/total_papers*100:.2f}%**)")
    md_lines.append(f"- **🔴 Overall Filtered Out Rate**: {total_papers - overall_both_pass}/{total_papers} (**{(total_papers - overall_both_pass)/total_papers*100:.2f}%**)\n")

    # Grouped Metrics by Field
    md_lines.append("## 🏷️ Breakdown Pass Rate by Field\n")
    md_lines.append("| Field | Total Papers | Title Pass Rate | Abstract Pass Rate | Both Pass Rate | Retained Count |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    grouped = res_df.groupby("field")
    for field_name, group in grouped:
        f_total = len(group)
        f_title_pass = group["title_eng"].sum()
        f_abstract_pass = group["abstract_eng"].sum()
        f_both_pass = group["both_pass"].sum()

        t_rate = f"{f_title_pass/f_total*100:.2f}%"
        a_rate = f"{f_abstract_pass/f_total*100:.2f}%"
        b_rate = f"{f_both_pass/f_total*100:.2f}%"

        md_lines.append(f"| {field_name} | {f_total} | {t_rate} | {a_rate} | **{b_rate}** | {f_both_pass}/{f_total} |")

    # Save to Markdown
    os.makedirs(os.path.dirname(OUTPUT_MD_PATH), exist_ok=True)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"📝 Summary report successfully generated at: {OUTPUT_MD_PATH}")
    print("\n" + "=" * 65)
    print(f"🟢 BOTH PASS (Overall Retained) : {overall_both_pass}/{total_papers} ({overall_both_pass/total_papers*100:.2f}%)")
    print("=" * 65)

if __name__ == "__main__":
    main()