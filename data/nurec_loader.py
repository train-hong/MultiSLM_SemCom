import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

class NuRecDataset(Dataset):
    def __init__(self, split="train", transform=None, max_samples=160):
        """
        Args:
            split (str): 資料集分割，通常為 'train' 或 'validation'
            transform (callable, optional): 影像前處理
            max_samples (int): 為了實驗一的設定，預設只取連續 160 張 frames
        """
        super().__init__()
        self.split = split
        self.transform = transform
        
        print(f"Initializing NuRec Dataset (split: {split})...")
        
        # 使用 streaming=True 避免下載整個龐大的自駕車資料集
        self.hf_dataset = load_dataset("nvidia/PhysicalAI-Autonomous-Vehicles-NuRec", split=split, streaming=True)
        
        # 將連續的 frames 預先拉到記憶體中，確保後續 batching 時的順序性與計算準確性
        self.frames = []
        print(f"Extracting {max_samples} continuous frames for batch observation...")
        
        for i, data in enumerate(self.hf_dataset):
            if i >= max_samples:
                break
                
            # NuRec 內部的影像欄位名稱可能會變動 (例如 'image', 'cam_front', 或 'frame')
            # 這裡做一個動態偵測，抓取第一個 Image 型態的欄位
            img = None
            for key, value in data.items():
                if isinstance(value, Image.Image):
                    img = value
                    break
            
            if img is None:
                # 如果沒有直接找到 PIL Image，退回嘗試找常見欄位
                img_key = 'image' if 'image' in data else list(data.keys())[0]
                img = data[img_key]
                
            self.frames.append(img)
            
        print(f"Successfully loaded {len(self.frames)} continuous frames.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img

def get_nurec_dataloader(batch_size=32, split="train", max_samples=160, img_size=(224, 224)):
    """
    建立實驗一使用的 DataLoader。
    注意：shuffle 必須強制為 False，以維持連續影像的特性。
    """
    # 基礎的影像前處理 (針對一般 SLM/LLM Vision Encoder 需求)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = NuRecDataset(split=split, transform=transform, max_samples=max_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,      # 關鍵：為了測量連續 frame，絕對不能打亂
        num_workers=4,      # 在 gpu06 上可以開多一點 workers 加速 data loading
        pin_memory=True     # 加速 CPU to GPU 的資料傳輸
    )
    
    return dataloader