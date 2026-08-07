import time
import torch

class LatencyTracker:
    def __init__(self, device):
        self.device = device
        self.records = {
            "Client_Inference": 0.0,
            "Transmission": 0.0,
            "Server_Inference": 0.0,
            "Total_Pipeline": 0.0
        }
        self.start_ticks = {}
        self.counts = {key: 0 for key in self.records.keys()}

    def _synchronize(self):
        """確保硬體加速器的運算隊列已清空，保證計時精確。"""
        if self.device == "cuda":
            torch.cuda.synchronize()

    def start(self, event_name: str):
        """開始計時"""
        if event_name not in self.records:
            raise ValueError(f"未知的測量事件: {event_name}")
            
        self._synchronize()
        self.start_ticks[event_name] = time.perf_counter()

    def stop(self, event_name: str):
        """停止計時並累加時間"""
        self._synchronize()
        end_time = time.perf_counter()
        
        if event_name not in self.start_ticks:
            raise RuntimeError(f"事件 {event_name} 尚未呼叫 start() 就嘗試 stop()")
            
        elapsed = end_time - self.start_ticks[event_name]
        self.records[event_name] += elapsed
        self.counts[event_name] += 1
        
        return elapsed

    def report_average(self):
        """印出平均 Latency 報告"""
        print("\n" + "="*40)
        print("📊 Latency Evaluation Report (Average)")
        print("="*40)
        for event, total_time in self.records.items():
            count = self.counts[event]
            if count > 0:
                avg_time = total_time / count
                print(f"⏱️ {event: <18}: {avg_time:.4f} seconds (over {count} runs)")
            else:
                print(f"⏱️ {event: <18}: N/A (0 runs)")
        print("="*40 + "\n")