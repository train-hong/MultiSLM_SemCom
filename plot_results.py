import matplotlib.pyplot as plt

# 實驗設定 (不畫出 64 img，因為 64 img 時 Batch 版本 OOM 了)
img_counts = [1, 2, 4, 8, 16, 32]
x_labels = [f"{n} img" for n in img_counts]

# ==========================================
# 1. Sequential 版本的數據 (從先前的 Log 擷取並轉換)
# ==========================================
# 當時測出的單張「平均」推論時間 (ms)
seq_client_avg = [1600.5, 984.7, 981.0, 982.8, 985.8, 992.7]
seq_server_avg = [1842.1, 1228.3, 1268.7, 1435.0, 1392.6, 1446.7]
seq_total_avg = [3442.7, 2213.0, 2249.7, 2417.8, 2378.4, 2439.4]

# 計算 Sequential 處理 N 張圖片的「總耗時」 (平均耗時 * 數量)
seq_client_total = [avg * n for avg, n in zip(seq_client_avg, img_counts)]
seq_server_total = [avg * n for avg, n in zip(seq_server_avg, img_counts)]
seq_total_time = [avg * n for avg, n in zip(seq_total_avg, img_counts)]

# ==========================================
# 2. Batch 版本的數據 (從最新的 OOM 實驗表格擷取)
# ==========================================
batch_client = [569.15, 447.48, 701.16, 1137.16, 2037.54, 4756.80]
batch_server = [1166.95, 1384.62, 1753.94, 1765.84, 2608.20, 3639.82]
batch_total = [1736.10, 1832.11, 2455.10, 2903.00, 4645.74, 8396.62]

# 設定圖表全域字體大小與樣式
plt.rcParams.update({'font.size': 12})

# ==========================================
# 繪製圖表一：Sequential 版本的細部拆解折線圖
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(x_labels, seq_total_time, marker='o', linestyle='-', color='red', linewidth=2.5, markersize=8, label='Total Inference')
plt.plot(x_labels, seq_server_total, marker='^', linestyle='--', color='blue', linewidth=2, markersize=8, label='Server Inference (7B)')
plt.plot(x_labels, seq_client_total, marker='v', linestyle='-.', color='green', linewidth=2, markersize=8, label='Client Inference (2B)')

plt.title('Sequential Mode: Inference Time vs. Image Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of Images', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('sequential_latency.png', dpi=300)
print("✅ 已成功儲存 Sequential 版本折線圖：sequential_latency.png")

plt.clf() # 清除畫布

# ==========================================
# 繪製圖表二：Batch 版本的細部拆解折線圖
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(x_labels, batch_total, marker='s', linestyle='-', color='purple', linewidth=2.5, markersize=8, label='Total Inference')
plt.plot(x_labels, batch_server, marker='^', linestyle='--', color='blue', linewidth=2, markersize=8, label='Server Inference (7B)')
plt.plot(x_labels, batch_client, marker='v', linestyle='-.', color='green', linewidth=2, markersize=8, label='Client Inference (2B)')

plt.title('Batch Mode: Inference Scalability & Bottleneck', fontsize=14, fontweight='bold')
plt.xlabel('Number of Concurrent Images (Batch Size)', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('batch_latency.png', dpi=300)
print("✅ 已成功儲存 Batch 版本折線圖：batch_latency.png")