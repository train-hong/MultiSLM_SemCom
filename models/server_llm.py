import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ServerLLM:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device=None):
        """
        初始化 Server 端的 LLM (Cloud Computing Node)。
        採用較大參數量的模型來處理 Client 傳來的語意特徵，並做出最終決策。
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Server] 初始化大型 LLM 模型 ({model_name})，使用裝置: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # ⚠️ [關鍵設定]：在進行 Batch Generation 時，Tokenizer 必須設定為左側補齊
        # 這樣模型才能正確對齊所有 Prompt 的結尾，進行平行推論
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        self.model.eval()

    def generate_final_decision_batch(self, received_tokens: torch.Tensor):
        """
        批次接收來自 Client 端傳輸的 Tokens，融合並推導出最終決策。
        :param received_tokens: 形狀為 (batch_size, seq_len) 的 Tensor
        :return: list of strings, 最終的推論結果列表
        """
        batch_size = received_tokens.shape[0]
        texts_to_prompt = []
        
        # 1. 批次語意解碼 (Decoding transmitted symbols)
        for i in range(batch_size):
            # 取出該 Client 的 Token 序列
            toks = received_tokens[i]
            
            # 過濾掉 Client 端因為對齊而產生的 Pad Token
            valid_toks = toks[toks != self.tokenizer.pad_token_id]
            edge_semantic_features = self.tokenizer.decode(valid_toks, skip_special_tokens=True)
            
            # 構建單一 Prompt
            system_prompt = "You are a powerful cloud server AI. Your task is to analyze semantic visual features extracted by edge devices and make a final comprehensive decision."
            user_prompt = f"Based on the following visual features extracted from different patches of an image, provide a concise, one-paragraph summary of the overall scene (maximum 3 sentences):\n\n{edge_semantic_features}"        
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts_to_prompt.append(text)
        
        # 2. 將整批 Prompt 轉換為模型輸入 Tensor
        inputs = self.tokenizer(
            texts_to_prompt, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # 3. 真正發揮 Server 算力的「平行推論」 (Heavy Computation)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # 4. 批次提取與解碼最終輸出
        input_len = inputs["input_ids"].shape[1]
        final_decisions = []
        
        for i in range(batch_size):
            generated_tokens = outputs[i, input_len:]
            decision_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            final_decisions.append(decision_text.strip())
            
        return final_decisions