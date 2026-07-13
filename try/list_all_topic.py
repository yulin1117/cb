import os
import json
import pandas as pd  # 💡 記得先在環境中 pip install pandas
from openalex.openalex import get_openalex_topics  # 確保這個函式在同一目錄下的 get_openalex_topics.py 中定義
def visualize_and_save_topics(cache_file="data/openalex/topics.json"):
    """
    讀取或下載全量 OpenAlex Topics，提取關鍵欄位，並用 Pandas 漂亮地展示與儲存。
    """
    # 1. 確保拿到全量資料 (呼叫你原本寫好的 get_openalex_topics 函式)
    # 如果 data/openalex/topics.json 已經存在，它會秒載入，不用重新連網
    all_raw_topics = get_openalex_topics(cache_file=cache_file)
    
    print(f"\n📊 開始解析 {len(all_raw_topics)} 個 OpenAlex 原始主題資料...")
    
    # 2. 提取我們關心的核心欄位，把巢狀 JSON 扁平化
    flattened_data = []
    for topic in all_raw_topics:
        topic_id = topic.get("id", "").split("/")[-1] # 提取 "T13674"
        topic_name = topic.get("display_name", "Unknown")
        
        # 關鍵：從巢狀的 field 字典中挖出領域名稱
        field_info = topic.get("field") or {}
        field_id = field_info.get("id", "").split("/")[-1] if field_info.get("id") else "Unknown"
        field_name = field_info.get("display_name", "Unknown") # 例如 "Computer Science"
        
        # 順便把更上層的 Domain 也抓下來，讓分類更清楚
        domain_info = topic.get("domain") or {}
        domain_name = domain_info.get("display_name", "Unknown")
        
        flattened_data.append({
            "Topic ID": topic_id,
            "Topic Name": topic_name,
            "Field ID": field_id,
            "Field Name": field_name,
            "Domain Name": domain_name
        })
        
    # 3. 轉化為 Pandas DataFrame 
    df = pd.DataFrame(flattened_data)
    
    # 4. 在控制台漂亮地印出統計摘要與前 15 行
    print("\n" + "="*80)
    print("                    核心學術領域（Fields）分佈統計")
    print("="*80)
    print(df["Field Name"].value_counts()) # 顯示每個領域各有多少個子主題
    print("="*80)
    
    print("\n💡 隨機抽樣 15 條主題對應關係預覽：")
    # 設定 Pandas 顯示選項，防止字串太長被截斷
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)
    print(df.sample(15, random_state=42).to_string(index=False))
    print("="*80)
    
    # 5. 導出成 CSV 檔案方便你用 Excel 隨時查看
    output_csv = "openalex_all_topics_mapping.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig") # utf-8-sig 能防 Excel 打開時亂碼
    print(f"🎉 完整的對應關係表格已成功儲存至: {os.path.abspath(output_csv)}")

# 測試呼叫
if __name__ == "__main__":
    visualize_and_save_topics()