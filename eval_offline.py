import json
from data.nurec_loader import NuRecDataset
from utils.metrics import TextImageVIEScoreEvaluator

def run_offline_evaluation():
    print("Loading trial predictions...")
    with open("trial_predictions.json", "r", encoding="utf-8") as f:
        predictions_data = json.load(f)
        
    # 初始化資料集來抓回原始圖片 (確保 max_samples 與原本一致)
    dataset = NuRecDataset(max_samples=160)
    
    # 初始化 VIEScore 裁判
    evaluator = TextImageVIEScoreEvaluator()
    
    evaluated_results = []
    
    print(f"Starting offline evaluation for {len(predictions_data)} samples...")
    for item in predictions_data:
        idx = item["frame_idx"]
        image = dataset.frames[idx]  # 透過 frame_idx 完美對應回原本的 PIL Image
        
        # 呼叫單張評估
        res = evaluator.evaluate_single(
            image=image,
            prediction=item["prediction"],
            ground_truth=item["ground_truth"],
            prompt=item["prompt"]
        )
        
        evaluated_results.append({
            "batch_size": item["batch_size"],
            "trial": item["trial"],
            "frame_idx": idx,
            "overall_score": res["overall_score"],
            "sub_scores": res["sub_scores"],
            "reasoning": res["reasoning"]
        })
        
    # 計算每個 batch size 的平均 ViE Score，準備拿去畫第二張圖
    # (可依據 batch_size 進行 groupby 平均)
    print("Offline evaluation completed!")
    # 後續可將 evaluated_results 存成 vie_score_results.csv 進行視覺化

if __name__ == "__main__":
    run_offline_evaluation()