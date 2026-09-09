import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM

from data.nurec_loader import get_nurec_dataloader
from utils.latency_tracker import LatencyTracker

def run_batch_observation():
    print("=== Starting Experiment 1: Batch Inference Observation ===")
    
    batch_sizes = [4, 8, 16, 32]
    num_trials = 5
    max_frames = 160
    
    VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    LLM_MODEL_ID = "Qwen/Qwen3-8B-Instruct"
    
    print(f"Loading Client Encoder from: {VLM_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    qwen_vl = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    
    print(f"Loading Server Decoder from: {LLM_MODEL_ID}")
    qwen_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    
    tracker = LatencyTracker()
    results = []
    all_predictions = []  # 專門用來存給離線 VIEScore 用的預測記錄

    global_frame_counter = 0

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        
        dataloader = get_nurec_dataloader(batch_size=bs, max_samples=max_frames)
        trial_metrics = []
        
        for trial in range(num_trials):
            tracker.reset()
            
            # 追蹤這個 trial 內的 frame index 起點
            current_frame_idx = 0
            
            for batch_imgs in tqdm(dataloader, desc=f"Trial {trial+1}/{num_trials}"):
                prompt_text = "Analyze this driving scene. Are there any immediate safety hazards? Describe the current situation."
                
                # --- 1. Client 端：Vision Encoder ---
                tracker.start("encoder")
                with torch.no_grad():
                    inputs = processor(
                        text=[prompt_text] * len(batch_imgs), 
                        images=list(batch_imgs), 
                        return_tensors="pt", 
                        padding=True
                    ).to("cuda", dtype=torch.float16)
                    
                    visual_tokens = qwen_vl.visual(
                        inputs.pixel_values, 
                        grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None
                    )
                tracker.stop("encoder")
                
                # --- 2. Server 端：LLM Decoder ---
                tracker.start("decoder")
                with torch.no_grad():
                    outputs = qwen_llm.generate(
                        inputs_embeds=visual_tokens.unsqueeze(0) if visual_tokens.dim() == 2 else visual_tokens,
                        max_new_tokens=50,
                        do_sample=False
                    )
                tracker.stop("decoder")
                
                # 解碼文字預測結果
                decoded_preds = processor.batch_decode(outputs, skip_special_tokens=True)
                
                # 將這一個 batch 的每張圖預測結果記錄下來
                for i, pred in enumerate(decoded_preds):
                    record = {
                        "batch_size": bs,
                        "trial": trial + 1,
                        "frame_idx": current_frame_idx + i,
                        "prompt": prompt_text,
                        "prediction": pred,
                        "ground_truth": "Safe driving conditions."  # 可替換為你的真實 GT
                    }
                    all_predictions.append(record)
                
                current_frame_idx += len(batch_imgs)
            
            # 統計這次 Trial 的數據
            trial_data = {
                "batch_size": bs,
                "trial": trial + 1,
                "encoder_time": tracker.get_total_time("encoder"),
                "decoder_time": tracker.get_total_time("decoder"),
                "total_time": tracker.get_total_time("encoder") + tracker.get_total_time("decoder"),
            }
            trial_metrics.append(trial_data)
            
        results.extend(trial_metrics)
        
    # 1. 匯出時間實驗結果給畫圖用
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiment_results.csv", index=False)
    
    # 2. 匯出預測文字結果供離線評估用
    with open("trial_predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)
        
    print("\nExperiment finished!")
    print("- Latency results saved to experiment_results.csv")
    print("- Predictions saved to trial_predictions.json")

if __name__ == "__main__":
    run_batch_observation()