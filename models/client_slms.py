import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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
        # torch_dtype=torch.bfloat16 可以大幅降低 Linux Server GPU 的 VRAM 佔用
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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

    def process_and_merge(self, image: Image.Image, prompt="Describe the key objects and background in this image region using one short, complete sentence."):
        patches = self._split_image_into_patches(image, grid_size=(2, 2))
        slm_output_tokens = []
        patch_labels = ["左上 (Patch 1)", "右上 (Patch 2)", "左下 (Patch 3)", "右下 (Patch 4)"]
        
        # 取得一個換行符號的 Token ID，用來隔開每個 Patch 的特徵
        # 這對 Server LLM 理解「這是 4 塊不同的區域」非常有幫助
        newline_token_id = self.processor.tokenizer.encode("\n", add_special_tokens=False)
        newline_tensor = torch.tensor([newline_token_id], device=self.device)
        
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
                    max_new_tokens=40,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                # 裁切掉 input_ids，只保留生成的 output tokens
                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0, input_len:] 
                
                # 把這個 Patch 的 Token 加進 list，並且在尾巴補上一個「換行 Token」
                slm_output_tokens.append(generated_tokens)
                slm_output_tokens.append(newline_tensor[0]) 
                
                patch_feature = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"    - {patch_labels[idx]} 特徵: '{patch_feature.strip()}'")
                
        # 現在串接起來的 Tokens，中間就會自帶 \n 換行了！
        merged_tokens = torch.cat(slm_output_tokens, dim=0)
        
        # 增加 batch 維度 (1, seq_len)，以符合 Server 端 LLM 的 input 預期
        merged_tokens = merged_tokens.unsqueeze(0) 
        
        return merged_tokens