import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining # 替換為你實際使用的 VLM class

class ClientMultiSLM:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None):
        """
        初始化 Client 端的 SLM。
        預設會自動偵測硬體環境，優先使用 MPS (Apple Silicon) 或 CUDA。
        """
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Client] 初始化 SLM 模型，使用裝置: {self.device}")
        
        # 載入 Processor 與 Model
        # 注意：這裡使用較通用的 Qwen-VL 作為範例，請依據你實際拿到的 Qwen3 0.6B 權重類型替換 Class
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 使用 bfloat16 或 float16 可以大幅節省記憶體並加速推論
        self.model = AutoModelForPreTraining.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.model.eval()

    def _split_image_into_patches(self, image: Image.Image, grid_size=(2, 2)):
        """
        將原始圖片切割成多個 Patches。
        :param image: PIL Image 對象
        :param grid_size: (高切割數, 寬切割數)，預設 2x2 會切出 4 張圖
        :return: list of PIL Images
        """
        w, h = image.size
        grid_h, grid_w = grid_size
        patch_w = w // grid_w
        patch_h = h // grid_h
        
        patches = []
        for i in range(grid_h):
            for j in range(grid_w):
                # 定義 crop 的 Bounding Box: (left, upper, right, lower)
                box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
                patch = image.crop(box)
                patches.append(patch)
                
        return patches

    def process_and_merge(self, image: Image.Image, prompt="Extract key visual features from this image region:"):
        """
        模擬多個 SLM 處理各個 Patch，並將生成的 Token 進行合併。
        """
        # 1. 圖片切割 (模擬將 Data 分發給多個 SLM_s)
        patches = self._split_image_into_patches(image, grid_size=(2, 2))
        
        slm_output_tokens = []
        
        # 2. 模擬多個 SLM_s 進行推論
        # 實務上為了節省記憶體，使用同一個 model 依序處理；若 VRAM 夠大也可改寫為 batch inference
        for idx, patch in enumerate(patches):
            # 準備輸入格式 (依據使用的 Qwen 模型 API 可能需要微調)
            inputs = self.processor(
                text=prompt, 
                images=patch, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # 讓 SLM 產生輸出 tokens
                # max_new_tokens 可依據實驗對 Transmission size 的限制進行調整
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=32,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                # 這裡通常只需保留新生成的 tokens，濾掉 prompt 的部分
                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0, input_len:] 
                slm_output_tokens.append(generated_tokens)
                
        # 3. Token 合併 (Concatenate)
        # 將所有 SLM_s 輸出的 1D token tensor 串接成一個長 sequence，準備進行 Token Transmission
        merged_tokens = torch.cat(slm_output_tokens, dim=0)
        
        # 增加 batch 維度 (1, seq_len)，以符合 Server 端 LLM 的 input 預期
        merged_tokens = merged_tokens.unsqueeze(0) 
        
        return merged_tokens