import time
import torch

class LatencyTracker:
    def __init__(self, device):
        self.device = device
        self.times = {}
        self.start_ticks = {}

    def _synchronize(self):
        # 確保硬體加速器的運算隊列已清空，這樣測出的時間才準確
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def start(self, event_name):
        self._synchronize()
        self.start_ticks[event_name] = time.perf_counter()

    def stop(self, event_name):
        self._synchronize()
        end_time = time.perf_counter()
        elapsed = end_time - self.start_ticks[event_name]
        self.times[event_name] = self.times.get(event_name, 0) + elapsed
        return elapsed

    def report(self):
        print("--- Latency Report (seconds) ---")
        for event, duration in self.times.items():
            print(f"{event}: {duration:.4f}s")
        print("--------------------------------")