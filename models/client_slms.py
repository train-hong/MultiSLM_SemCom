import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence

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
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=self.device
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
                box = (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
                patch = image.crop(box)
                patches.append(patch)
                
        return patches

    def process_and_merge_batch(self, images: list, prompt="Describe the key objects and background in this image region using one short, complete sentence."):
        """
        批次處理 (Batched Processing)：同時處理 N 張圖片 (4N 個 Patches)。
        """
        batch_size = len(images)
        all_patches = []
        
        # 1. 批次切割：將所有圖片都切成 4 個 Patch，打平成一個大 List
        for img in images:
            all_patches.extend(self._split_image_into_patches(img, grid_size=(2, 2)))
            
        total_patches = len(all_patches) # 這會是 4 * batch_size
        
        # 取得換行 Token
        newline_token_id = self.processor.tokenizer.encode("\n", add_special_tokens=False)[0]
        newline_tensor = torch.tensor([newline_token_id], device=self.device)
        
        # 2. 批次建立 Prompt
        texts = []
        for _ in range(total_patches):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            
        # 3. 批次轉換為 Tensor (一口氣將所有 Patches 送給 Processor)
        inputs = self.processor(
            text=texts, 
            images=all_patches, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # 4. 真正發揮 GPU 算力的「平行推論」
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=40,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
            input_len = inputs["input_ids"].shape[1]
            # 取得所有 Patches 新生成的 Tokens，形狀為 (total_patches, max_gen_len)
            generated_tokens = outputs[:, input_len:] 
            
        # 5. Token 重組與合併 (將 4N 個結果還原成 N 個 Client 的輸出)
        merged_batch_list = []
        pad_id = self.processor.tokenizer.pad_token_id
        
        for i in range(batch_size):
            img_tokens_parts = []
            for j in range(4): # 每 4 個 Patch 屬於同一張圖片
                patch_idx = i * 4 + j
                toks = generated_tokens[patch_idx]
                
                # 剔除掉為了對齊長度而產生的 Pad Token
                valid_toks = toks[toks != pad_id]
                
                img_tokens_parts.append(valid_toks)
                img_tokens_parts.append(newline_tensor)
                
            # 將這 4 個 Patch 串接成一條長 Token 序列 (代表這一個 Client 的輸出)
            concatenated = torch.cat(img_tokens_parts, dim=0)
            merged_batch_list.append(concatenated)
            
        # 6. 將長度不一的 Token 序列補齊 (Padding)，才能組裝成 (batch_size, seq_len) 的矩陣交給 Server
        merged_batch_tensor = pad_sequence(
            merged_batch_list, 
            batch_first=True, 
            padding_value=pad_id
        )
        
        return merged_batch_tensor