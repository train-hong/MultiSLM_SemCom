import argparse
import torch

# 引入我們剛剛寫好的各個模組
from data.imagenet_loader import ImageNetLoader
from models.client_slms import ClientMultiSLM
from models.server_llm import ServerLLM
from transmission.token_channel import TokenChannel
from utils.latency_tracker import LatencyTracker

class EndToEndEvaluator:
    def __init__(self, data_dir: str, max_test_samples: int = 5):
        """
        初始化端雲協同實驗 Pipeline。
        :param data_dir: 測試圖片所在的資料夾路徑
        :param max_test_samples: 初期為了快速驗證，限制最多跑幾張圖
        """
        # 決定硬體 (優先使用 CUDA)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 啟動 SemCom Pipeline，全域執行環境: {self.device}\n" + "-"*40)
        
        # 初始化時間追蹤器
        self.tracker = LatencyTracker(self.device)
        self.max_test_samples = max_test_samples
        
        # 初始化四大模組
        self.dataloader = ImageNetLoader(data_dir=data_dir, split="val").get_dataloader(batch_size=1)
        self.client = ClientMultiSLM(device=self.device)
        self.channel = TokenChannel()
        self.server = ServerLLM(device=self.device)
        print("-" * 40 + "\n[Pipeline] 所有模組初始化完成，準備開始推論測試...")

    def run(self):
        """執行端到端 (End-to-End) 的實驗流程"""
        import itertools # 👉 [新增] 引入無限輪迴套件
        
        # 👉 [修改] 把 dataloader 變成可以無限循環的 iterator
        dataloader_iterator = itertools.cycle(self.dataloader)
        
        for step in range(self.max_test_samples):
            # if step >= self.max_test_samples:
            #     print(f"已達到設定的測試數量上限 ({self.max_test_samples} 張)，停止推論。")
            #     break

            batch = next(dataloader_iterator)
                
            print(f"\n▶️ 正在處理第 {step + 1} 張圖片...")
            
            # 因為 batch_size=1，我們直接取出第一張 PIL Image
            image = batch["images"][0]

            self.tracker.start("Total_Pipeline")
            
            # ---------------------------------------------------------
            # 1. Client 端運算 (SLM_s 處理與 Token 合併)
            # ---------------------------------------------------------
            self.tracker.start("Client_Inference")
            merged_tokens = self.client.process_and_merge(image)
            self.tracker.stop("Client_Inference")
            
            # ---------------------------------------------------------
            # 2. 模擬 Token 傳輸
            # ---------------------------------------------------------
            self.tracker.start("Transmission")
            received_tokens = self.channel.transmit(merged_tokens)
            self.tracker.stop("Transmission")
            
            # ---------------------------------------------------------
            # 3. Server 端運算 (LLM 推導 Final Decision)
            # ---------------------------------------------------------
            self.tracker.start("Server_Inference")
            final_decision = self.server.generate_final_decision(received_tokens)
            self.tracker.stop("Server_Inference")
            
            self.tracker.stop("Total_Pipeline")

            print(f"✅ Server Final Decision: {final_decision.strip()}")
            
        # 所有測試跑完後，印出平均 Latency 報告
        self.tracker.report_average()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-SLM Semantic Communication Pipeline")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="指向你的 ImageNet 資料夾，或自己建的 dummy_data 資料夾路徑"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=3, 
        help="要跑幾張圖片進行 Latency 測試 (預設 3 張)"
    )
    args = parser.parse_args()
    
    # 建立 Evaluator 並開始執行
    evaluator = EndToEndEvaluator(data_dir=args.data_dir, max_test_samples=args.samples)
    evaluator.run()