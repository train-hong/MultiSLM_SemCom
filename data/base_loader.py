from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseDatasetLoader(ABC):
    """
    資料讀取器的抽象基底類別 (Abstract Base Class)。
    確保未來的 ImageNetLoader 與 CocoLoader 都有統一的呼叫介面。
    """
    
    @abstractmethod
    def get_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        """
        回傳 PyTorch DataLoader 物件。
        """
        pass