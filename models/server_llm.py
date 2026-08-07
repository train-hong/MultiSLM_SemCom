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
        
        # 1. 載入 Tokenizer (用來解碼接收到的 Token 以及編碼 Server 的 Prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 2. 載入強大的 Server 端語言模型 (純 LLM)
        # 7B 等級的模型在 Linux Server 上使用 bfloat16 非常重要，可避免 OOM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        self.model.eval()

    def generate_final_decision(self, received_tokens: torch.Tensor):
        """
        接收來自 Client 端傳輸的 Tokens，融合並推導出最終決策。
        :param received_tokens: 形狀為 (1, seq_len) 的 Tensor
        :return: string, 最終的推論結果
        """
        # 1. 語意解碼 (Decoding transmitted symbols)
        # 將 Client 傳來的 Token ID 轉換回可讀的文字特徵
        # 由於 Client 也是 Qwen 家族，共用相似的詞表，這裡解碼能還原 SLM 提取的語意
        edge_semantic_features = self.tokenizer.decode(received_tokens[0], skip_special_tokens=True)
        
        # 2. 構建 Server 端的 Prompt (融合邊緣端資訊)
        # 告訴 LLM 這些資訊是來自多個影像區塊的特徵，請它做 Final Decision
        system_prompt = "You are a powerful cloud server AI. Your task is to analyze semantic visual features extracted by edge devices and make a final comprehensive decision."
        user_prompt = f"Based on the following visual features extracted from different patches of an image, provide a brief overall description of the scene:\n\n{edge_semantic_features}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 套用 Qwen2.5 的 Chat Template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 將 Server 的 Prompt 轉為模型輸入 Tensor
        inputs = self.tokenizer(
            text, 
            return_tensors="pt"
        ).to(self.device)
        
        # 3. Server 端推論 (Heavy Computation)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Server 可以輸出較長的詳細決策
                temperature=0.7,     # 稍微增加一點生成多樣性
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # 4. 提取最終輸出
        input_len = inputs["input_ids"].shape[1]
        final_decision_tokens = outputs[0, input_len:]
        final_decision_text = self.tokenizer.decode(final_decision_tokens, skip_special_tokens=True)
        
        return final_decision_text