import time
import torch

class LatencyTracker:
    def __init__(self, device="cuda"):
        self.device = device
        self.records = {}
        self.start_ticks = {}
        self.counts = {}

    def _synchronize(self):
        """確保硬體加速器的運算隊列已清空，保證計時精確。"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, event_name: str):
        """開始計時"""
        # 如果是新的測量事件，動態初始化
        if event_name not in self.records:
            self.records[event_name] = 0.0
            self.counts[event_name] = 0
            
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
        
        # 刪除該次 tick 確保狀態乾淨 (選用)
        del self.start_ticks[event_name]
        
        return elapsed

    def get_total_time(self, event_name: str):
        """取得特定事件的總累加時間 (供實驗輸出 CSV 使用)"""
        return self.records.get(event_name, 0.0)

    def reset(self):
        """重置所有紀錄 (給不同 trial 迴圈使用)"""
        self.records = {}
        self.start_ticks = {}
        self.counts = {}

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