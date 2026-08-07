import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

class ClientMultiSLM:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None):
        """
        初始化 Client 端的 SLM。
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Client] 初始化 SLM 模型，使用裝置: {self.device}")
        
        # 1. 載入 Processor (負責處理圖片轉 Tensor 與 Prompt Tokenization)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # 2. 載入 VLM
        # 使用 AutoModelForVision2Seq 會自動對應到 Qwen2VLForConditionalGeneration
        # torch_dtype=torch.bfloat16 可以大幅降低 Linux Server GPU 的 VRAM 佔用
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=self.device # 讓 transformers 自動把模型放到指定裝置
        )
        
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

    def process_and_merge(self, image: Image.Image, prompt="Describe the key features of this image region briefly."):
        """
        模擬多個 SLM 處理各個 Patch，並將生成的 Token 進行合併。
        """
        patches = self._split_image_into_patches(image, grid_size=(2, 2))
        slm_output_tokens = []
        
        for idx, patch in enumerate(patches):
            # Qwen2-VL 的官方建議輸入格式 (Chat Template)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # 將 messages 轉為模型能吃得純文字 prompt
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 將圖片與文字交給 processor 轉成 Tensor
            inputs = self.processor(
                text=[text], 
                images=[patch], 
                padding=True, 
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
                
                # 裁切掉 input_ids，只保留生成的 output tokens
                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0, input_len:] 
                slm_output_tokens.append(generated_tokens)
                
        # 3. Token 合併 (Concatenate)
        # 將所有 SLM_s 輸出的 1D token tensor 串接成一個長 sequence，準備進行 Token Transmission
        merged_tokens = torch.cat(slm_output_tokens, dim=0)
        
        # 增加 batch 維度 (1, seq_len)，以符合 Server 端 LLM 的 input 預期
        merged_tokens = merged_tokens.unsqueeze(0) 
        
        return merged_tokens