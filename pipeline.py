import argparse
import torch
import itertools

from data.imagenet_loader import ImageNetLoader
from models.client_slms import ClientMultiSLM
from models.server_llm import ServerLLM
from transmission.token_channel import TokenChannel
from utils.latency_tracker import LatencyTracker

class EndToEndEvaluator:
    def __init__(self, data_dir: str, batch_size: int = 1, num_runs: int = 3):
        """
        初始化端雲協同實驗 Pipeline (Batch Mode)。
        :param data_dir: 測試圖片所在的資料夾路徑
        :param batch_size: 初始併發數量
        :param num_runs: 每個 Batch Size 組合要重複跑幾次取平均
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 啟動 SemCom Pipeline (Batch Mode)，全域執行環境: {self.device}\n" + "-"*40)
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_runs = num_runs 
        
        self.tracker = LatencyTracker(self.device)
        
        # 初始化四大模組
        # 依據初始 batch_size 建立 DataLoader
        self.dataloader = ImageNetLoader(data_dir=data_dir, split="val").get_dataloader(batch_size=self.batch_size)
        self.client = ClientMultiSLM(device=self.device)
        self.channel = TokenChannel()
        self.server = ServerLLM(device=self.device)
        print("-" * 40 + "\n[Pipeline] 所有模組初始化完成，準備開始推論測試...")

    def update_batch_size(self, new_batch_size: int):
        """提供給 run_exp.py 使用：動態更新併發數量，並重建 DataLoader"""
        self.batch_size = new_batch_size
        # 重建 DataLoader 以符合新的 Batch Size
        self.dataloader = ImageNetLoader(data_dir=self.data_dir, split="val").get_dataloader(batch_size=self.batch_size)

    def run(self):
        """執行批次 (Batched) 的端到端實驗流程"""
        dataloader_iterator = itertools.cycle(self.dataloader)
        
        # 跑 num_runs 次來取平均，確保時間數據穩定
        for step in range(self.num_runs):
            batch = next(dataloader_iterator)
            
            # 💡 [關鍵改變] 這裡的 images 已經是一個 List[PIL.Image]，長度為 self.batch_size
            images = batch["images"]
            
            # 強制將圖片數量複製擴充到指定的 batch_size
            while len(images) < self.batch_size:
                images.extend(batch["images"])
            images = images[:self.batch_size] # 截斷多餘的，確保精準等於 batch_size

            print(f"\n▶️ 正在處理 Batch {step + 1}/{self.num_runs} (併發 {self.batch_size} 張圖片)...")

            self.tracker.start("Total_Pipeline")
            
            # ---------------------------------------------------------
            # 1. Client 端運算 (同時處理 N 張圖片)
            # ---------------------------------------------------------
            self.tracker.start("Client_Inference")
            # 呼叫新的 batch 處理函數
            merged_tokens = self.client.process_and_merge_batch(images) 
            self.tracker.stop("Client_Inference")
            
            # ---------------------------------------------------------
            # 2. 模擬 Token 傳輸
            # ---------------------------------------------------------
            self.tracker.start("Transmission")
            received_tokens = self.channel.transmit(merged_tokens)
            self.tracker.stop("Transmission")
            
            # ---------------------------------------------------------
            # 3. Server 端運算 (同時推導 N 份決策)
            # ---------------------------------------------------------
            self.tracker.start("Server_Inference")
            # 呼叫新的 batch 處理函數
            final_decisions = self.server.generate_final_decision_batch(received_tokens)
            self.tracker.stop("Server_Inference")
            
            self.tracker.stop("Total_Pipeline")

        self.tracker.report_average()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-SLM Semantic Communication Pipeline (Batch Mode)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="併發圖片數量")
    parser.add_argument("--runs", type=int, default=3, help="重複執行次數")
    args = parser.parse_args()
    
    evaluator = EndToEndEvaluator(data_dir=args.data_dir, batch_size=args.batch_size, num_runs=args.runs)
    evaluator.run()