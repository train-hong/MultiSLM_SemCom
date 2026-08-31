import argparse
import pandas as pd
from pipeline import EndToEndEvaluator
from utils.latency_tracker import LatencyTracker
import torch

def run_all_experiments(data_dir):
    # 你表格中定義的測試數量
    # test_cases = [1, 2, 4, 8, 16, 32]
    test_cases = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # 準備用來存結果的字典
    results = {
        "Client Inference (ms)": [],
        "Server Inference (ms)": [],
        "Inference (ms)": [],
        "VIEScore": [],
        "Transmission (ms)": []
    }
    
    print("🚀 開始執行 ImageNet 自動化實驗...")
    print("-" * 50)

    # evaluator = EndToEndEvaluator(data_dir=data_dir, max_test_samples=1)
    evaluator = EndToEndEvaluator(data_dir=data_dir, batch_size=1, num_runs=3)
    
    for samples in test_cases:
        print(f"\n▶️ 正在測試 {samples} img...")

        evaluator.update_batch_size(samples)
        evaluator.tracker = LatencyTracker(evaluator.device)

        try:
            evaluator.run()
            
            # 從 tracker 拿取總時間並計算平均 (秒 -> 毫秒)
            # Inference = Client_Inference + Server_Inference
            client_time = evaluator.tracker.records["Client_Inference"]
            server_time = evaluator.tracker.records["Server_Inference"]
            transmission_time = evaluator.tracker.records["Transmission"]
            
            # 確保有跑出數據，避免除以 0
            count = evaluator.tracker.counts["Client_Inference"]
            if count > 0:
                avg_client_ms = (client_time / count) * 1000
                avg_server_ms = (server_time / count) * 1000
                avg_inference_ms = ((client_time + server_time) / count) * 1000
                avg_transmission_ms = (transmission_time / count) * 1000
            else:
                avg_inference_ms = 0
                avg_transmission_ms = 0
                
            # 記錄到表格中 (VIEScore 尚未實作，先填入 N/A)
            results["Client Inference (ms)"].append(f"{avg_client_ms:.2f}")
            results["Server Inference (ms)"].append(f"{avg_server_ms:.2f}")
            results["Inference (ms)"].append(f"{avg_inference_ms:.2f}")
            results["VIEScore"].append("N/A")  
            results["Transmission (ms)"].append(f"{avg_transmission_ms:.2f}")

        except torch.cuda.OutOfMemoryError:
            print(f"❌ [OOM Error] 在測試 {samples} img 時，GPU 記憶體不足！")
            # 清除 GPU 暫存，避免影響下一個可能的測試 (雖然這裡通常建議直接中斷)
            torch.cuda.empty_cache()
            
            # 發生 OOM 時填入標記
            results["Client Inference (ms)"].append("OOM")
            results["Server Inference (ms)"].append("OOM")
            results["Inference (ms)"].append("OOM")
            results["VIEScore"].append("N/A")  
            results["Transmission (ms)"].append("OOM")
            
            print("⚠️ 系統已達極限，停止後續更大負載的測試。")
            break
        
    print("\n✅ 所有實驗執行完畢！\n")
    
    # 使用 pandas 產生精美的 Markdown 表格
    actual_test_cases = test_cases[:len(results["Inference (ms)"])]
    df = pd.DataFrame(results, index=[f"{n} img" for n in actual_test_cases]).T
    
    print("📊 實驗結果表格：")
    print("=" * 60)
    print(df.to_markdown())
    print("=" * 60)
    
    # 將結果存成 CSV 方便後續作圖
    df.to_csv("experiment_results.csv")
    print("\n💾 數據已儲存至 experiment_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SemCom Experiments")
    parser.add_argument("--data_dir", type=str, required=True, help="資料集路徑")
    args = parser.parse_args()
    
    run_all_experiments(args.data_dir)