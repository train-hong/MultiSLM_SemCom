import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

class VIEScoreTextEvaluator:
    """
    為文字預測任務（SemCom VQA）客製化的 VIEScore 評估器。
    精神源自 VIEScore：結合 MLLM 裁判、Explainable 推理、子分數與 min 嚴格懲罰機制。
    """
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda"):
        print(f"Initializing VIEScore Judge Model: {model_id}...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.judge_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        ).eval()

    def evaluate_single(self, image: Image.Image, prediction: str, ground_truth: str = None, prompt: str = ""):
        """
        針對單張影像與預測文字進行 Viescore 評估。
        """
        judge_prompt = f"""
        You are a professional autonomous driving evaluation judge. 
        Evaluate the effectiveness and accuracy of the AI-generated text response based on the provided driving image, prompt, and optional ground truth.

        [Input Context]
        - Prompt/Instruction: {prompt}
        - Model Prediction: {prediction}
        - Ground Truth / Reference: {ground_truth if ground_truth else "N/A"}

        [Rules]
        On a scale of 0 to 10:
        1. Visual Faithfulness Score: Does the prediction accurately reflect objects, hazards, and details visible in the image? (0 = completely hallucinated or wrong, 10 = perfectly accurate).
        2. Task & Safety Relevance Score: Does the prediction correctly address the driving task and safety analysis? (0 = irrelevant or unsafe logic, 10 = professional and correct).

        [Output Format]
        You must output in valid JSON format:
        {{
          "score": [visual_score, task_score],
          "reasoning": "Concise explanation of why these scores were given."
        }}
        """

        # 封裝成 Qwen3-VL 的多模態 Chat 格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": judge_prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.judge_model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # 解析裁判輸出的 JSON 與分數
        scores, reasoning = self._parse_response(response_text)
        
        # 核心精神：使用 min 運算強調「不容許關鍵項目崩潰」，並正規化到 [0, 1]
        if scores and len(scores) == 2:
            overall_score = min(scores) / 10.0
        else:
            overall_score = 0.0

        return {
            "overall_score": overall_score,
            "sub_scores": scores,
            "reasoning": reasoning,
            "raw_response": response_text
        }

    def _parse_response(self, response_str: str):
        """
        參考 VIEScore 官方做法：透過 JSON 解析，失敗時改用 Regex 撈出分數，再失敗則給予零分懲罰。
        """
        try:
            match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                scores = data.get("score", [0, 0])
                reasoning = data.get("reasoning", "")
                return [float(s) for s in scores], reasoning
        except Exception:
            pass
        
        try:
            score_match = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*]', response_str)
            if score_match:
                return [float(score_match.group(1)), float(score_match.group(2))], response_str
        except Exception:
            pass
            
        return [0.0, 0.0], f"Parsing failed. Raw output: {response_str}"

def calculate_vie_score(predictions, images, ground_truths=None, prompts=None, judge_evaluator=None):
    """
    批次計算 ViE Score 的介面函數，供實驗一收集平均 Accuracy 使用。
    """
    if judge_evaluator is None:
        raise ValueError("Please provide an initialized VIEScoreTextEvaluator instance.")

    total_scores = []
    for i in range(len(predictions)):
        img = images[i]
        pred = predictions[i]
        gt = ground_truths[i] if ground_truths else None
        prompt = prompts[i] if prompts else "Analyze this driving scene."
        
        result = judge_evaluator.evaluate_single(img, pred, gt, prompt)
        total_scores.append(result["overall_score"])
        
    return sum(total_scores) / len(total_scores) if total_scores else 0.0