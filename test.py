import re
import time
import requests
from collections import Counter
# 從專案現有的模組中匯入摘要還原功能
from openalex.openalex import reconstruct_abstract

def fetch_unfiltered_raw_works(topic_id: str = "T13674", num_papers: int = 50, oa_status: str = "diamond", lang_filter: str = "en") -> list[dict]:
    """
    直接從 OpenAlex API 撈取特定主題未經篩選的原始論文（可篩選指定的 Open Access 狀態與語言）
    當要求數量大於 100 時，會自動進行多頁（Pagination）安全抓取，避免 HTTP 400 錯誤。
    """
    print(f"正在從 OpenAlex API 撈取主題 [{topic_id}] 的原始文獻（目標數量: {num_papers}，篩選 OA 狀態: {oa_status}，語言限制: {lang_filter}）...")
    base_url = "https://api.openalex.org/works"
    
    results = []
    page = 1
    page_size = 100  # OpenAlex 官方規定的單頁安全上限值
    
    headers = {
        "User-Agent": "CitationBiasBenchmark/DiagnosticTest (mailto:tobias.schreieder@tu-dresden.de)"
    }
    
    # 透過迴圈進行分頁請求，直到累積數量達到要求為止
    while len(results) < num_papers:
        # 計算此頁需要撈取的剩餘數量
        current_per_page = min(page_size, num_papers - len(results))
        
        # ─── 核心修改：在 filter 中加入 language 過濾條件 ───
        filter_str = f"primary_topic.id:{topic_id},open_access.oa_status:{oa_status}"
        if lang_filter:
            filter_str += f",language:{lang_filter}"
            
        url = f"{base_url}?filter={filter_str}&per-page={current_per_page}&page={page}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                page_results = data.get("results", [])
                if not page_results:
                    print(f"   [提示] 已無更多文獻資料可供下載。")
                    break  # API 已經沒有更多論文了
                
                results.extend(page_results)
                print(f"   [進度] 成功獲取第 {page} 頁，目前累計: {len(results)}/{num_papers} 篇")
                page += 1
                
                # 稍微延遲 0.1 秒防止對伺服器發送過於密集的請求
                time.sleep(0.1)
            else:
                print(f"❌ API 請求失敗 (第 {page} 頁)，狀態碼: {resp.status_code}，錯誤訊息: {resp.text}")
                break
        except Exception as e:
            print(f"❌ 請求第 {page} 頁時發生異常: {e}")
            break
            
    # 切片確保最終回傳的資料長度與要求完全一致
    final_works = results[:num_papers]
    print(f"🎉 抓取任務完成！最終成功取得 {len(final_works)} 篇原始文獻。\n")
    return final_works

def run_diagnostic_suite(works: list[dict]):
    """
    核心診斷測試套件：針對 Problems.docx 提到的四類問題進行深度掃描
    """
    if not works:
        print("無可用的文獻資料進行分析。")
        return

    print("\n" + "="*80)
    print("                    🔍 OPENALEX 原始資料品質診斷報告 🔍")
    print("="*80)

    # 初始化偵測器變數
    abstract_map = {}  # 用於偵測「相同摘要 (Same Abstract)」
    web_noise_count = 0
    language_issue_count = 0
    missing_metadata_count = 0
    wrong_type_count = 0

    # 1. 網頁雜訊 (Web Noise) 黑名單關鍵字（依據 Problems.docx）
    noise_keywords = [
    # 付費牆與權限阻擋字眼 (對應 source 19, 20)
    r"this content is only available as a pdf",
    r"you do not currently have access to this content",
    r"article pdf first page preview",
    
    # 資料庫功能鍵與搜尋導覽 (對應 source 14, 16, 19, 30)
    r"search for other works by this author",
    r"oxford academic", r"pubmed", r"google scholar",
    r"download citation file", r"toolbar search",
    
    # 社群分享與版權/文獻導出功能列 (對應 source 17, 19, 27, 28)
    r"share icon share", r"share on", r"facebook", r"twitter", r"linkedin", r"wechat",
    r"reprints and permissions", r"cite icon cite",
    
    # 網頁流量指標說明長文 (對應 source 22, 24, 26)
    r"article views are the counter-compliant sum",
    r"altmetric attention score",
    r"clicking on the donut icon will load a page"
    ]
    noise_regex = re.compile("|".join(noise_keywords), re.IGNORECASE)

    # 2. 允許的五大文獻類型 (Filter types)
    ALLOWED_TYPES = {"article", "dissertation", "book-chapter", "preprint", "book"}

    # 3. 常用英文停用詞（用於輕量級語系檢測判定）
    english_stopwords = {"the", "and", "of", "is", "to", "in", "that", "it", "with", "as", "for", "was"}

    print(f"系統開始掃描 {len(works)} 篇論文...\n")

    for idx, work in enumerate(works, 1):
        work_id = work.get("id", "").split("/")[-1]
        title = work.get("title") or ""
        raw_language = work.get("language", "unknown")
        work_type = work.get("type", "unknown")
        
        # 獲取實際的 Open Access 狀態以便在診斷中顯示確認
        oa_info = work.get("open_access") or {}
        actual_oa_status = oa_info.get("oa_status", "unknown") if isinstance(oa_info, dict) else "unknown"
        
        # 嘗試還原摘要
        inv_index = work.get("abstract_inverted_index")
        abstract_text = reconstruct_abstract(inv_index) if inv_index else ""

        # ----------------------------------------------------
        # 診斷 A: 相同摘要偵測 (Same Abstract for multiple papers)
        # ----------------------------------------------------
        if abstract_text and abstract_text.strip():
            # 將摘要進行基礎標準化後做 Key 比對
            normalized_abs = " ".join(abstract_text.lower().split())
            abstract_map.setdefault(normalized_abs, []).append((work_id, title))

        # ----------------------------------------------------
        # 診斷 B: 網頁雜訊偵測 (Web Noise)
        # ----------------------------------------------------
        noise_found_title = noise_regex.findall(title)
        noise_found_abs = noise_regex.findall(abstract_text) if abstract_text else []
        
        if noise_found_title or noise_found_abs:
            web_noise_count += 1
            all_noises = list(set(noise_found_title + noise_found_abs))
            print(f"❌ [問題類型: Web Noise] 論文: {work_id} (OA狀態: {actual_oa_status})")
            print(f"   └─ 標題: {title[:80]}...")
            print(f"   └─ 偵測到網頁雜訊關鍵字: {all_noises}")
            if noise_found_abs:
                print(f"   └─ 摘要片段: ...{abstract_text[:150]}...")
            print("-" * 50)

        # ----------------------------------------------------
        # 診斷 C: 語言標記異常偵測 (Language Issue)
        # ----------------------------------------------------
        # 如果 OpenAlex 標記為英文 'en'，我們用停用詞檢測和非 ASCII 比例來評估它是否真的是英文
        if raw_language == "en" and (title or abstract_text):
            combined_text = f"{title} {abstract_text}".lower()
            words = re.findall(r"\b[a-z']+\b", combined_text)
            
            # 統計這段文字包含多少常見英文基本詞彙
            overlap = [w for w in words if w in english_stopwords]
            non_ascii_chars = len(re.sub(r"[\x00-\x7F]+", "", combined_text))
            total_chars = len(combined_text) if combined_text else 1
            non_ascii_ratio = non_ascii_chars / total_chars

            # 如果單字量夠多但完全沒有出現 basic English stopwords，或是非 ASCII 字元佔比過高（如中文/俄文等），判定為異常
            is_suspicious_non_english = False
            if len(words) > 10 and len(overlap) == 0:
                is_suspicious_non_english = True
            elif non_ascii_ratio > 0.15:  # 超過 15% 為非英文字元
                is_suspicious_non_english = True

            if is_suspicious_non_english:
                language_issue_count += 1
                print(f"❌ [問題類型: Language Issue] 論文: {work_id} (OA狀態: {actual_oa_status})")
                print(f"   └─ OpenAlex 標註語系: {raw_language}")
                print(f"   └─ 疑似非英文特徵: 英文停用詞重疊數={len(overlap)}，非ASCII字元佔比={non_ascii_ratio:.2%}")
                print(f"   └─ 標題: {title}")
                print("-" * 50)

        # ----------------------------------------------------
        # 診斷 D: 缺失元數據偵測 (Missing Metadata)
        # ----------------------------------------------------
        authors = work.get("authorships") or []
        pub_year = work.get("publication_year")
        
        is_missing = False
        missing_fields = []
        if not title:
            missing_fields.append("title")
            is_missing = True
        if not pub_year:
            missing_fields.append("publication_year")
            is_missing = True
        if not authors:
            missing_fields.append("authorships")
            is_missing = True
        if not inv_index:
            missing_fields.append("abstract")
            is_missing = True

        if is_missing:
            missing_metadata_count += 1
            print(f"⚠️ [問題類型: Missing Metadata] 論文: {work_id} (OA狀態: {actual_oa_status})")
            print(f"   └─ 遺失之關鍵元數據: {missing_fields}")
            print("-" * 50)

        # ----------------------------------------------------
        # 診斷 E: 類型過濾器檢查 (Filter Types)
        # ----------------------------------------------------
        if work_type not in ALLOWED_TYPES:
            wrong_type_count += 1
            print(f"⚠️ [問題類型: Unexpected Type] 論文: {work_id} (OA狀態: {actual_oa_status})")
            print(f"   └─ 該文獻類型為 '{work_type}'，不在規定的 5 大類型 {ALLOWED_TYPES} 內")
            print("-" * 50)

    # ----------------------------------------------------
    # 分析「相同摘要 (Same Abstract)」的統計結果
    # ----------------------------------------------------
    duplicate_abstract_groups = {k: v for k, v in abstract_map.items() if len(v) > 1}
    
    if duplicate_abstract_groups:
        print("\n" + "!"*40)
        print("🚨 警告：偵測到嚴重的「同一個摘要套用在不同論文」現象！")
        print("!"*40)
        for i, (abs_text, linked_papers) in enumerate(duplicate_abstract_groups.items(), 1):
            print(f"\n[重複摘要組別 #{i}] 共有 {len(linked_papers)} 篇論文使用了完全相同的摘要：")
            for pid, ptitle in linked_papers:
                print(f"   👉 ID: {pid} | 標題: {ptitle}")
            print(f"   👉 摘要內容預覽: \"{abs_text[:250]}...\"")
            print("="*40)

    # ─── 最終診斷統計摘要 ───
    print("\n" + "="*80)
    print("                    📊 最終診斷統計摘要報告 📊")
    print("="*80)
    print(f" 總掃描論文數          : {len(works)} 篇")
    print(f" 1. 相同摘要異常組數    : {len(duplicate_abstract_groups)} 組")
    print(f" 2. 混入網頁/付費牆雜訊  : {web_noise_count} 篇")
    print(f" 3. 語言標記異常 (非英文): {language_issue_count} 篇")
    print(f" 4. 關鍵元數據遺失     : {missing_metadata_count} 篇")
    print(f" 5. 非預期類型文獻     : {wrong_type_count} 篇")
    print("="*80)
    print(" 💡 解決方案提示：")
    print("   - 相同摘要 & 網頁雜訊：請確保下游管線套用 `get_works_for_topic()` 中的 `is_complete_work()` 過濾器。")
    print("   - 語言標記異常：建議在 pipeline 中引入 `langdetect` 套件來對還原後的 abstract 進行二次語系校正。")
    print("="*80 + "\n")

if __name__ == "__main__":
    # 你可以更換成你想觀察的 Topic ID (例如 T13674 或是歷史學相關的主題 T11117)
    # 預設會過濾：開放獲取狀態為 "diamond"、語言為 "en"（英文）
    raw_unfiltered_papers = fetch_unfiltered_raw_works(topic_id="T13674", num_papers=1000, oa_status="diamond", lang_filter="en")
    
    run_diagnostic_suite(raw_unfiltered_papers)