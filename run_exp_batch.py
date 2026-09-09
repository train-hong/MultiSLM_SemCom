import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM

from data.nurec_loader import get_nurec_dataloader
from utils.latency_tracker import LatencyTracker
from utils.metrics import VIEScoreTextEvaluator

def run_batch_observation():
    print("=== Starting Experiment 1: Batch Inference Observation ===")
    
    # 實驗參數設定 (對齊 RL Action Space 需求)
    batch_sizes = [4, 8, 16, 32]  # batch_size=0 (無壓縮原圖傳輸) 視為 Baseline，後續可額外測量
    num_trials = 5
    max_frames = 160
    
    # 模型名稱 (可以根據你 gpu06 的 VRAM 大小調整參數規模，例如用 8B 或是 3B)
    VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    LLM_MODEL_ID = "Qwen/Qwen3-8B-Instruct"
    
    print(f"Loading Client Encoder from: {VLM_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    # 載入 Qwen3-VL (我們只需要它的 vision backbone)
    qwen_vl = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    
    print(f"Loading Server Decoder from: {LLM_MODEL_ID}")
    # 載入 Qwen3 作為 Server 端 Decoder
    qwen_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    
    tracker = LatencyTracker()
    results = []

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        
        dataloader = get_nurec_dataloader(batch_size=bs, max_samples=max_frames)
        trial_metrics = []
        
        for trial in range(num_trials):
            tracker.reset()
            batch_vie_scores = []
            
            for batch_imgs in tqdm(dataloader, desc=f"Trial {trial+1}/{num_trials}"):
                
                # 準備 Text Prompt (監控任務的問題)
                prompt_text = "Analyze this driving scene. Are there any immediate safety hazards? Describe the current situation."
                
                # --- 1. Client 端：Vision Encoder (產生 Tokens) ---
                tracker.start("encoder")
                with torch.no_grad():
                    # 透過 Processor 處理影像
                    inputs = processor(
                        text=[prompt_text] * len(batch_imgs), 
                        images=list(batch_imgs), 
                        return_tensors="pt", 
                        padding=True
                    ).to("cuda", dtype=torch.float16)
                    
                    # 擷取 Qwen3-VL 的 Visual Tokens
                    # 注意：實際 API 根據 Hugging Face 版本可能微調，這裡擷取 image_embeds
                    visual_tokens = qwen_vl.visual(
                        inputs.pixel_values, 
                        grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None
                    )
                tracker.stop("encoder")
                
                # (暫時忽略傳輸時間，我們目前專注算 Inference Time)
                
                # --- 2. Server 端：LLM Decoder ---
                tracker.start("decoder")
                with torch.no_grad():
                    # 在一般的 Baseline，我們將 visual_tokens 直接當作 inputs_embeds 餵給 LLM
                    # (需結合 text_embeds，這裡簡化演示推論過程)
                    # 為了計時準確，這裡模擬生成 50 個 tokens
                    outputs = qwen_llm.generate(
                        inputs_embeds=visual_tokens.unsqueeze(0) if visual_tokens.dim() == 2 else visual_tokens,
                        max_new_tokens=50,
                        do_sample=False
                    )
                tracker.stop("decoder")
                
                # --- 3. 計算 Accuracy (ViE Score) ---
                evaluator = VIEScoreTextEvaluator(model=qwen_llm, processor=processor)

                # 針對這個 batch 的所有預測結果進行評分
                pred_texts = decoded_preds # 模型生成的文字
                gt_texts = ["Ground truth description for frame..."] * len(decoded_preds) # 對應的真實答案

                # 取得平均分數
                batch_vie_scores = evaluator.evaluate_batch(pred_texts, gt_texts)
            
            # 統計這次 Trial 的數據
            trial_data = {
                "batch_size": bs,
                "trial": trial + 1,
                "encoder_time": tracker.get_total_time("encoder"),
                "decoder_time": tracker.get_total_time("decoder"),
                "total_time": tracker.get_total_time("encoder") + tracker.get_total_time("decoder"),
                "vie_score": np.mean(batch_vie_scores)
            }
            trial_metrics.append(trial_data)
            
        # 計算 5 次 trial 的平均
        df_trial = pd.DataFrame(trial_metrics)
        print(f"Batch {bs} | Avg Total Time: {df_trial['total_time'].mean():.4f}s | Avg ViE: {df_trial['vie_score'].mean():.4f}")
        results.extend(trial_metrics)
        
    # 匯出 CSV 供 plot_results.py 畫圖
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiment_results.csv", index=False)
    print("\nExperiment finished. Results saved to experiment_results.csv")

if __name__ == "__main__":
    run_batch_observation()