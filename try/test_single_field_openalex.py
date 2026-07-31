import os
import requests
import nltk
from nltk.tokenize import sent_tokenize
import fasttext
from tqdm import tqdm

# ==========================================
# ⚙️ Configurations
# ==========================================
TARGET_FIELD = "Business, Management and Accounting"
FETCH_COUNT = 5000  # 欲爬取的論文數量

# FastText 官方預訓練模型路徑
FASTTEXT_MODEL_PATH = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/CitationBias/lid.176.bin"
# NLTK Check
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# FastText Model Load
if not os.path.exists(FASTTEXT_MODEL_PATH):
    FASTTEXT_MODEL_PATH = "lid.176.bin"

print(f"📦 Loading FastText model from: {FASTTEXT_MODEL_PATH}...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# ==========================================
# 🧠 Language Detection Core Functions
# ==========================================
def title_lang_detect(text: str, model, min_confidence: float = 0.5) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    clean_text = text.strip().replace("\n", " ")
    if len(clean_text) < 5:
        return False

    (labels, probs) = model.predict(clean_text, k=1)
    lang = labels[0].replace("__label__", "")
    confidence = probs[0]

    return (lang == "en") and (confidence >= min_confidence)


def lang_detect(text: str, model, threshold: float = 0.2) -> bool:
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

    if valid_sentences_count == 0:
        return False

    return (non_eng_sentences / valid_sentences_count) < threshold

# ==========================================
# 🌐 OpenAlex Fetcher
# ==========================================
def fetch_openalex_papers(field_name: str, total_needed: int = 5000):
    """
    Directly fetches papers from OpenAlex API based on field topic / display_name
    """
    print(f"🌐 Fetching top {total_needed} papers for field: '{field_name}' from OpenAlex API...")
    papers = []
    page = 1
    per_page = 200 # OpenAlex max limit per page
    
    headers = {
        "User-Agent": "CitationBiasBenchmark (mailto:tobias.schreieder@tu-dresden.de)"
    }
    ALLOWED_TYPES = {
        "article",
        "book-chapter",
        "preprint",
        "dissertation",
        "book",
        "review",
        "report",
    }
    type_filter = "|".join(sorted(ALLOWED_TYPES))
    select_fields = "id,title,type,language,abstract_inverted_index"
    
    while len(papers) < total_needed:
        # 搜尋包含特定 Concept / Field 的論文，且必須有 title
        # https://openalex.org/works?page=1&filter=primary_topic.field.id:14
        url = f"https://api.openalex.org/works?filter=primary_topic.field.id:14,has_abstract:true,language:en,type:{type_filter}&per_page={per_page}&page={page}"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"⚠️ API request failed with status {response.status_code}")
            break
            
        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            work_id = item.get("id", "")
            title = item.get("title", "")
            
            # 反轉 Abstract (OpenAlex API 的 abstract 是 Inverted Index 格式)
            abstract_dict = item.get("abstract_inverted_index", {})
            abstract = ""
            if abstract_dict:
                word_positions = []
                for word, pos_list in abstract_dict.items():
                    for pos in pos_list:
                        word_positions.append((pos, word))
                word_positions.sort(key=lambda x: x[0])
                abstract = " ".join([word for _, word in word_positions])

            papers.append({
                "work_id": work_id,
                "title": title if title else "",
                "abstract": abstract if abstract else ""
            })

            if len(papers) >= total_needed:
                break
        page += 1

    print(f"✅ Successfully fetched {len(papers)} papers from OpenAlex.\n")
    return papers

# ==========================================
# 🚀 Execution & Error Diagnostics
# ==========================================
def main():
    papers = fetch_openalex_papers(TARGET_FIELD, FETCH_COUNT)
    
    if not papers:
        print("❌ No papers retrieved. Exiting.")
        return

    title_pass = 0
    abstract_pass = 0
    both_pass = 0
    
    failed_samples = []

    print("🔍 Testing Language Detection on fetched dataset...")
    for p in tqdm(papers):
        t_eng = title_lang_detect(p["title"], ft_model, min_confidence=0.5)
        a_eng = lang_detect(p["abstract"], ft_model, threshold=0.2)

        if t_eng: title_pass += 1
        if a_eng: abstract_pass += 1
        if t_eng and a_eng: 
            both_pass += 1
        else:
            # 記錄失敗案例做後續分析
            if len(failed_samples) < 10:
                # 取得 FastText 對 Title 的預測細節
                (labels, probs) = ft_model.predict(p["title"].strip().replace("\n", " "), k=1) if p["title"] else (["__label__unknown"], [0.0])
                failed_samples.append({
                    "title": p["title"],
                    "title_pred_lang": labels[0].replace("__label__", ""),
                    "title_conf": probs[0],
                    "title_pass": t_eng,
                    "abstract_pass": a_eng
                })

    total = len(papers)

    # 📊 印出結果
    print("\n" + "=" * 65)
    print(f"📊 DIAGNOSTIC RESULTS FOR FIELD: '{TARGET_FIELD}'")
    print("=" * 65)
    print(f"Total Fetch & Audited  : {total}")
    print(f"Title Pass (Conf ≥ 0.5): {title_pass}/{total} ({title_pass/total*100:.2f}%)")
    print(f"Abstract Pass (Ratio < 0.2): {abstract_pass}/{total} ({abstract_pass/total*100:.2f}%)")
    print(f"🟢 BOTH PASS (Retained): {both_pass}/{total} (**{both_pass/total*100:.2f}%**)")
    print(f"🔴 Filtered Out Rate    : {total - both_pass}/{total} ({(total - both_pass)/total*100:.2f}%)")
    print("=" * 65)

if __name__ == "__main__":
    main()