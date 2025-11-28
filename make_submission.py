"""
Create ensemble submission by combining basic and advanced predictions.
Usage: python make_submission.py [--weight_basic 0.5] [--weight_advanced 0.5]
"""
import pandas as pd
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create ensemble submission")
    parser.add_argument("--weight_basic", type=float, default=0.5, help="Weight for basic submission (0-1)")
    parser.add_argument("--weight_advanced", type=float, default=0.5, help="Weight for advanced submission (0-1)")
    parser.add_argument("--output", type=str, default="submission_ensemble.csv", help="Output filename")
    args = parser.parse_args()
    
    # Validate weights
    total_weight = args.weight_basic + args.weight_advanced
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: weights sum to {total_weight}, normalizing...")
        args.weight_basic /= total_weight
        args.weight_advanced /= total_weight
    
    print("\n" + "=" * 70)
    print("ENSEMBLE SUBMISSION GENERATOR")
    print("=" * 70)
    
    # Load submissions
    print("\n[1] Loading submissions...")
    try:
        basic = pd.read_csv("submission.csv")
        print(f"✓ Basic: {basic.shape}")
    except FileNotFoundError:
        print("✗ submission.csv not found. Run: python main.py")
        return
    
    try:
        advanced = pd.read_csv("submission_advanced.csv")
        print(f"✓ Advanced: {advanced.shape}")
    except FileNotFoundError:
        print("✗ submission_advanced.csv not found. Run: python advanced_pipeline.py")
        return
    
    # Validate
    if basic.shape != advanced.shape:
        print(f"✗ Shape mismatch: {basic.shape} vs {advanced.shape}")
        return
    
    if not (basic['id'] == advanced['id']).all():
        print("✗ ID mismatch between submissions")
        return
    
    print(f"✓ Validation passed")
    
    # Create ensemble
    print("\n[2] Creating ensemble...")
    ensemble = basic.copy()
    ensemble['prediction'] = (
        args.weight_basic * basic['prediction'] + 
        args.weight_advanced * advanced['prediction']
    )
    
    print(f"Weights: Basic={args.weight_basic:.2f}, Advanced={args.weight_advanced:.2f}")
    print(f"\nBasic stats:")
    print(f"  Range: [{basic['prediction'].min():.6f}, {basic['prediction'].max():.6f}]")
    print(f"  Mean: {basic['prediction'].mean():.6f}")
    
    print(f"\nAdvanced stats:")
    print(f"  Range: [{advanced['prediction'].min():.6f}, {advanced['prediction'].max():.6f}]")
    print(f"  Mean: {advanced['prediction'].mean():.6f}")
    
    print(f"\nEnsemble stats:")
    print(f"  Range: [{ensemble['prediction'].min():.6f}, {ensemble['prediction'].max():.6f}]")
    print(f"  Mean: {ensemble['prediction'].mean():.6f}")
    
    # Save
    print(f"\n[3] Saving to {args.output}...")
    ensemble.to_csv(args.output, index=False)
    print(f"✓ Saved {len(ensemble)} predictions")
    
    # Correlation
    corr = basic['prediction'].corr(advanced['prediction'])
    print(f"\nCorrelation (Basic vs Advanced): {corr:.3f}")
    
    print("\n" + "=" * 70)
    print(f"✓ {args.output} created successfully")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
