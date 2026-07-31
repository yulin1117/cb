
"""
IN_FOLDER="/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/topics"
讀取此檔案夾下每一個folder中的/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/topics/TXXXXX/audit_results.json
格式：
[
  {
    "id": "https://openalex.org/W1994312337",
    "title": "...",
    "abstract": "...",
    "text": "...",
    "match": true,
    "eng": true,
    "clean": true,
    "pass": true,
    "llm_explanation": "..."
    },...
]
取出所有的pass為true的數量，計算pass率，目前有26個topics，就會有26個pass率，最後計算平均pass率
最後以圖表呈現，並儲存圖表
"""



import os
import json
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. 設定路徑與初始化變數
# -----------------------------
IN_FOLDER = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/out/topics"
topic_pass_rates = {}  # 紀錄各個 Topic 的 pass 率

print(f"🔍 開始掃描資料夾：{IN_FOLDER}")

# -----------------------------
# 2. 走訪資料夾並讀取 audit_results.json
# -----------------------------
# 排序資料夾名稱，確保圖表呈現時有順序
for folder_name in sorted(os.listdir(IN_FOLDER)):
    folder_path = os.path.join(IN_FOLDER, folder_name)
    
    # 確保是資料夾（例如 T10366）
    if not os.path.isdir(folder_path):
        continue
        
    audit_file = os.path.join(folder_path, "audit_results.json")
    
    # 檢查該 Topic 是否有 audit_results.json
    if os.path.exists(audit_file):
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                papers = json.load(f)
            
            if not isinstance(papers, list) or len(papers) == 0:
                print(f"⚠️ {folder_name}: audit_results.json 為空或格式不正確，跳過。")
                continue
                
            # 計算該 Topic 中 pass 為 True 的數量
            total_papers = len(papers)
            pass_count = sum(1 for p in papers if p.get("pass") is True)
            
            # 計算 Pass 率 (百分比)
            pass_rate = (pass_count / total_papers) * 100
            topic_pass_rates[folder_name] = pass_rate
            
            print(f"✅ {folder_name}: total {total_papers} | pass {pass_count} | pass rate {pass_rate:.2f}%")
            
        except Exception as e:
            print(f"❌ 讀取 {folder_name} 的 audit_results.json 失敗: {e}")
    else:
        # 有些資料夾可能還沒跑到這一步
        print(f"ℹ️ {folder_name}: 尚未生成 audit_results.json")

# -----------------------------
# 3. 計算全體平均 Pass 率
# -----------------------------
if not topic_pass_rates:
    print("❌ 沒有找到任何有效的 audit_results.json 檔案，程式結束。")
    exit()

all_rates = list(topic_pass_rates.values())
average_pass_rate = np.mean(all_rates)
total_topics_found = len(topic_pass_rates)

print("\n" + "="*40)
print(f"📊 Summary：")
print(f"   num of topics: {total_topics_found}")
print(f"   Avg Pass Rate: {average_pass_rate:.2f}%")
print("="*40 + "\n")

# -----------------------------
# 4. 繪製圖表並儲存
# -----------------------------
topics = list(topic_pass_rates.keys())
rates = list(topic_pass_rates.values())

# 設定畫布大小與解析度
plt.figure(figsize=(14, 6), dpi=150)

# 繪製長條圖
bars = plt.bar(topics, rates, color="#3498db", edgecolor="#2980b9", alpha=0.85, width=0.6)

# 加上平均線 (虛線)
plt.axhline(y=average_pass_rate, color="#e74c3c", linestyle="--", linewidth=1.5, 
            label=f"Average Pass Rate ({average_pass_rate:.2f}%)")

# 在每個長條圖上方顯示精準數字
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", 
             ha='center', va='bottom', fontsize=8, fontweight='bold', color="#2c3e50")

# 設定圖表標籤與標題
plt.title("Semantic Verification Pass Rate by Topic (LLM Gate)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Topic ID", fontsize=11, labelpad=10)
plt.ylabel("Pass Rate (%)", fontsize=11, labelpad=10)
plt.ylim(0, 110)  # 留點頂部空間給文字標籤
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(loc="upper right", frameon=True, shadow=False)

plt.tight_layout()

# 儲存圖表到最外層目錄
save_plot_path = "/home/yuzh952h/workspaces/horse/yuzh952h-cb/CitationBias/try/topics_pass_rate_analysis.png"
plt.savefig(save_plot_path)
plt.close()

print(f"💾 圖表已成功儲存至: {save_plot_path}")
