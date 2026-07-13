import os
import time
import random
import requests
import pandas as pd

def batch_compare_oa_metrics(input_csv_path="openalex_all_topics_mapping.csv", output_csv_path="topic_oa_comparison_report.csv"):
    """
    讀取主題 CSV，透過 OpenAlex API 的 group_by 機制同時對比：
    1. 全域開放獲取 (af_oa) 殘餘數量與是否達標 (pass_af_oa)
    2. 鑽石開放獲取 (af_oa_diamond) 殘餘數量與是否達標 (pass_af_oa_diamond)
    
    終點目標：評估是否有足夠的緩衝文獻量（閾值設定為 > 10000 篇）。
    """
    if not os.path.exists(input_csv_path):
        print(f"❌ 找不到輸入檔案: {input_csv_path}，請檢查路徑。")
        return

    # 1. 讀取你持有的映射表
    print(f"📖 正在讀取主題檔案: {input_csv_path} ...")
    df_input = pd.read_csv(input_csv_path)
    
    # 欄位防呆對齊
    id_col = "Topic ID" if "Topic ID" in df_input.columns else "id"
    name_col = "Topic Name" if "Topic Name" in df_input.columns else "display_name"
    field_col = "Field Name" if "Field Name" in df_input.columns else "field_name"
    
    results = []
    headers = {
        "User-Agent": "CitationBiasBenchmark (mailto:tobias.schreieder@tu-dresden.de)"
    }
    type_filter = "book|dissertation|book-chapter|preprint|article|preprint|review|report"
    print(f"🚀 開始向 OpenAlex API 批量聚合查詢 {len(df_input)} 個主題的雙軌 OA 數量變化...")

    # 2. 核心查詢循環
    for idx, row in df_input.iterrows():
        raw_id = str(row[id_col]).strip()
        topic_name = str(row[name_col]).strip()
        field_name = str(row[field_col]).strip() if field_col in df_input.columns else "Unknown"
        topic_id = raw_id.rstrip("/").split("/")[-1]
        
        # ─── 條件 A：全域開放獲取 (is_oa:true) ───
        url_oa = (
            f"https://api.openalex.org/works"
            f"?filter=primary_topic.id:{topic_id},"
            f"open_access.is_oa:true,"
            f"language:en,"
            f"type:{type_filter}"
            f"&group_by=primary_topic.id"
        )
        
        # ─── 條件 B：鑽石開放獲取 (oa_status:diamond) ───
        url_diamond = (
            f"https://api.openalex.org/works"
            f"?filter=primary_topic.id:{topic_id},"
            f"open_access.oa_status:diamond,"
            f"language:en,"
            f"type:{type_filter}"
            f"&group_by=primary_topic.id"
        )
        
        af_oa = 0
        af_oa_diamond = 0
        
        # 執行 A 請求 (帶有指數退避保險機制)
        try:
            for _ in range(3):
                r = requests.get(url_oa, headers=headers, timeout=15)
                if r.status_code == 200:
                    break
                time.sleep(1)
            if r.status_code == 200:
                groups = r.json().get("group_by", [])
                for g in groups:
                    if topic_id in g.get("key", ""):
                        af_oa = g.get("count", 0)
                        break
        except Exception as e:
            print(f"⚠️ {topic_id} 全域 OA 讀取失敗: {e}")

        # 執行 B 請求
        try:
            for _ in range(3):
                r = requests.get(url_diamond, headers=headers, timeout=15)
                if r.status_code == 200:
                    break
                time.sleep(1)
            if r.status_code == 200:
                groups = r.json().get("group_by", [])
                for g in groups:
                    if topic_id in g.get("key", ""):
                        af_oa_diamond = g.get("count", 0)
                        break
        except Exception as e:
            print(f"⚠️ {topic_id} Diamond 讀取失敗: {e}")

        # 3. 計算閾值檢測標籤 (True/False 代表 works > 10000)
        pass_af_oa = af_oa > 10000
        pass_af_oa_diamond = af_oa_diamond > 10000
        
        # 即時反饋
        print(f" [{idx+1}/{len(df_input)}] 主題: {topic_id} | 全域 OA: {af_oa:<6} ({'✅' if pass_af_oa else '❌'}) | Diamond: {af_oa_diamond:<5} ({'✅' if pass_af_oa_diamond else '❌'})")
        
        results.append({
            "Topic ID": topic_id,
            "Topic Name": topic_name,
            "Field Name": field_name,  # 留存分類依據
            "af_oa": af_oa,
            "af_oa_diamond": af_oa_diamond,
            "pass_af_oa": pass_af_oa,
            "pass_af_oa_diamond": pass_af_oa_diamond
        })
        time.sleep(0.1) # 禮貌延時

    # 4. 建立與儲存全新欄位的結果 CSV
    df_output = pd.DataFrame(results)
    
    # 重新整理輸出欄位（拋棄 Field Name 保持你指定的簡潔欄位輸出）
    csv_fields = ["Topic ID", "Topic Name", "af_oa", "af_oa_diamond", "pass_af_oa", "pass_af_oa_diamond"]
    df_output[csv_fields].to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n🎉 雙軌 OA 審計 CSV 報告導出成功！檔案位於: {output_csv_path}")

    # ──────────────────────────────────────────────────────────────────────
    # 核心需求 3：輸出每個 Field 通過數量檢測的 Topic 數量報告
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("      📊 【全流程學術領域 (Field) 分布與通過率審查統計報告】")
    print("="*80)
    
    # 按 Field Name 分組，分別計算 pass_af_oa 為 True 與 pass_af_oa_diamond 為 True 的總個數
    # 由於 True 在 Python 中等價於 1，直接用 sum() 就能精準算出通過的主題個數
    summary_df = df_output.groupby("Field Name").agg(
        總測試主題數=("Topic ID", "count"),
        全域OA通過數=("pass_af_oa", "sum"),
        鑽石OA通過數=("pass_af_oa_diamond", "sum")
    ).reset_index()
    
    # 漂亮格式化打印
    print(summary_df.to_string(index=False))
    print("="*80)
    print("💡 提示：若發現某領域的【鑽石OA通過數】過低，強烈建議在後續爬蟲腳本中")
    print("   將篩選防火牆放寬至全域的 `open_access.is_oa:true`，以防採樣失敗。")
    print("="*80 + "\n")

if __name__ == "__main__":
    batch_compare_oa_metrics()