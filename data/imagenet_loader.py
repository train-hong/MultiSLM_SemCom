import os
from torch.utils.data import DataLoader
from torchvision import datasets
from .base_loader import BaseDatasetLoader

class ImageNetLoader(BaseDatasetLoader):
    def __init__(self, data_dir: str, split: str = "val"):
        """
        初始化 ImageNet 資料讀取器。
        :param data_dir: ImageNet 資料集的根目錄 (例如 '/dataset/imagenet')
        :param split: 'train' 或 'val'
        """
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)
        
        # 檢查路徑是否存在 (方便你在 Server 上 Debug)
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"[Error] 找不到 ImageNet 目錄: {self.split_dir}。請確認 Server 上的路徑是否正確。")
            
        print(f"[Data] 載入 ImageNet ({split} split) from {self.split_dir}")
        
        # 注意：我們刻意不放入 torchvision.transforms.ToTensor()
        # 因為 Client 端的 SLM 需要原始的 PIL.Image 才能做 Grid Crop
        self.dataset = datasets.ImageFolder(root=self.split_dir)

    def _pil_collate_fn(self, batch):
        """
        自訂的 batch 打包邏輯。
        PyTorch 預設的 default_collate 無法處理 PIL Image，
        所以我們手動把它們拆解成 images list 和 labels list。
        """
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # 回傳 dict 格式，方便後續擴充 (例如未來 COCO 可加入 'captions')
        return {
            "images": images, 
            "labels": labels
        }

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        """
        生成 DataLoader。
        對於第一階段測量 Latency，batch_size 通常設為 1 即可。
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._pil_collate_fn,
            num_workers=4,        # 可依據 Server CPU 核心數調整，加速資料讀取
            pin_memory=False      # 因為是 PIL Image，不需要 pin_memory
        )