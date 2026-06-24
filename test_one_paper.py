import json
import os
import requests
# 修正匯入路徑，直接匯入 reconstruct_abstract
from openalex.openalex import reconstruct_abstract


def fetch_single_work_by_id(work_url_or_id: str) -> dict | None:
    """
    直接向 OpenAlex API 請求單篇特定論文的完整資料
    """
    # 提取純 Work ID（例如從 "https://openalex.org/works/W1492614353" 提取出 "W1492614353"）
    work_id = str(work_url_or_id).rstrip("/").split("/")[-1]
    api_url = f"https://api.openalex.org/works/{work_id}"

    headers = {
        "User-Agent": "CitationBiasBenchmark/SinglePreview (mailto:tobias.schreieder@tu-dresden.de)"
    }

    print(f"正在直接向 OpenAlex API 請求單篇論文資料 (ID: {work_id})...")
    
    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
        if resp.status_code == 200:
            work_data = resp.json()
            if isinstance(work_data, dict):
                return work_data
            print(f"⚠️ 警告：回傳的資料格式異常。")
            return None
        else:
            print(f"❌ 請求失敗，狀態碼: {resp.status_code}，訊息: {resp.text}")
            return None
    except Exception as e:
        print(f"❌ 發生異常: {e}")
        return None


def preview_paper_format():
    # 設定你想觀察的特定單篇論文 URL
    test_work_url = "https://openalex.org/works/W1492614353"
    work_id = test_work_url.rstrip("/").split("/")[-1]
    
    # 呼叫直接獲取單篇論文的函式
    work = fetch_single_work_by_id(work_url_or_id=test_work_url)
    
    if not work:
        print("無法載入論文，請確認網路連線或論文 ID 是否正確。")
        return

    print("\n" + "="*50)
    print("【單篇特定論文原始資料格式預覽 (含摘要還原)】")
    print("="*50)
    
    # ─── 核心修改：在此處呼叫還原摘要函式 ───
    inv_index = work.get("abstract_inverted_index")
    if inv_index:
        # 呼叫函式還原純文字摘要
        normal_abstract = reconstruct_abstract(inv_index)
        # 將還原後的純文字塞回 work 字典的一個新欄位中，方便一起被 json.dumps 印出
        work["reconstructed_abstract_pure_text"] = normal_abstract
    else:
        work["reconstructed_abstract_pure_text"] = "【此論文無摘要資料 (abstract_inverted_index 缺失)】"
    # ───────────────────────────────────────

    # 使用 json.dumps 讓排版更漂亮、易讀
    # 你可以在輸出的 JSON 結尾處找到 "reconstructed_abstract_pure_text" 欄位
    print(json.dumps(work, indent=2, ensure_ascii=False))

    # ─── 新增：確保 data 資料夾存在，並將結果儲存至 data/ 目錄下 ───
    output_dir = "data"
    try:
        # 自動建立 data 資料夾 (若不存在，等同於 mkdir -p 效果)
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"{work_id}.json")
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            # ensure_ascii=False 確保非英文（如中文字元等）能以正常可讀純文字儲存，不變成 \uXXXX 編碼形式
            json.dump(work, f, indent=2, ensure_ascii=False)
        print("\n" + "="*50)
        print(f"💾 成功將論文資料（含還原摘要）儲存至：{output_filepath}")
        print("="*50)
    except Exception as e:
        print(f"❌ 建立資料夾或儲存 JSON 檔案時發生錯誤: {e}")


if __name__ == "__main__":
    preview_paper_format()