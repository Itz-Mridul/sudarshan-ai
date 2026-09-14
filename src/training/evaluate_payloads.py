import os
import sys
import tempfile
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training.config import Config
from data_pipeline.dataset import split_stems_three_way, get_dataloaders
from data_pipeline.lsb_embedder import create_stego_dataset
from training.evaluate import load_model, run_inference, compute_metrics

def main():
    cfg = Config()
    cfg.mode = "fusion"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating multi-payload generalisation on: {device}")
    
    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.best_model_name)
    if not os.path.exists(ckpt_path):
        print(f"Error: No trained model found at {ckpt_path}")
        print("Run full training first!")
        sys.exit(1)
        
    model = load_model(ckpt_path, cfg.mode, cfg).to(device)
    
    # 1. Get TEST stems ONLY
    _, _, test_stems = split_stems_three_way(
        cfg.clean_dir, cfg.stego_dir, val_split=cfg.val_split, test_split=cfg.test_split, 
        max_images=cfg.max_images, seed=cfg.seed
    )
    test_stems_list = list(test_stems)
    
    payload_rates = [0.05, 0.10, 0.20, 0.40]
    results = {}
    
    print(f"\nEvaluating on {len(test_stems)} test pairs across {len(payload_rates)} payload rates...")
    print("-" * 65)
    print(f"{'Payload (bpp)':<15} | {'Accuracy':<10} | {'AUC':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    # 2. For each rate, embed test set & evaluate
    for rate in payload_rates:
        with tempfile.TemporaryDirectory() as temp_stego_dir:
            # Generate temporary stego images for the TEST set
            from data_pipeline.lsb_embedder import embed_lsb
            from PIL import Image
            import numpy as np
            
            for stem in test_stems:
                src_path = os.path.join(cfg.clean_dir, f"{stem}.png")
                if not os.path.exists(src_path):
                    src_path = os.path.join(cfg.clean_dir, f"{stem}.pgm")
                
                # Deterministic seed per image based on its stem
                import hashlib
                sha_seed = int(hashlib.sha256(stem.encode()).hexdigest(), 16) % (2**31)
                
                img_array = np.array(Image.open(src_path).convert("L"))
                payload_size = int((img_array.size * rate) // 8)
                payload = bytes([0] * payload_size)  # dummy payload for testing
                stego_array = embed_lsb(img_array, payload, seed=sha_seed)
                
                out_path = os.path.join(temp_stego_dir, f"{stem}.png")
                Image.fromarray(stego_array).save(out_path, format="PNG")
            
            # Create dataloader using SteganalysisDataset directly
            from data_pipeline.dataset import SteganalysisDataset, get_transforms
            from torch.utils.data import DataLoader
            
            _, val_transform = get_transforms(cfg.image_size)
            
            test_dataset = SteganalysisDataset(
                cfg.clean_dir,
                temp_stego_dir,
                allowed_stems=test_stems,
                transform=val_transform,
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
            
            labels, preds, probs = run_inference(model, test_loader, device)
            metrics = compute_metrics(labels, preds, probs)
            results[rate] = metrics
            
            print(f"{rate:<15.2f} | {metrics['accuracy']:<10.2f} | {metrics['auc']:<10.2f} | {metrics['f1']:<10.2f}")
            
    print("-" * 65)
    print("Add this table to your paper's robustness evaluation section.")

if __name__ == "__main__":
    main()
