"""
Summarize OOF metrics from reports directory.
Reads all walkforward_oof_fin_*.json files and displays them in a table.
"""
import json
from pathlib import Path
from datetime import datetime


def main():
    reports_dir = Path("reports")
    pattern = "walkforward_oof_fin_*.json"
    
    files = sorted(reports_dir.glob(pattern), reverse=True)
    
    if not files:
        print("No OOF metrics found in reports/")
        return
    
    print("\n" + "=" * 80)
    print("WALK-FORWARD OOF METRICS HISTORY")
    print("=" * 80)
    print(f"{'Timestamp':<20} {'Sharpe (2bps)':<15} {'Max Drawdown':<15} {'File'}")
    print("-" * 80)
    
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        
        sharpe = data.get('oof_sharpe_2bps', 'N/A')
        maxdd = data.get('oof_max_drawdown', 'N/A')
        
        # Extract timestamp from filename
        fname = fpath.name
        ts_str = fname.replace('walkforward_oof_fin_', '').replace('.json', '')
        try:
            ts = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
            timestamp = ts.strftime('%Y-%m-%d %H:%M:%S')
        except:
            timestamp = ts_str
        
        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        maxdd_str = f"{maxdd:.2f}" if isinstance(maxdd, (int, float)) else str(maxdd)
        
        print(f"{timestamp:<20} {sharpe_str:<15} {maxdd_str:<15} {fpath.name}")
    
    print("=" * 80)
    print(f"\nTotal runs: {len(files)}")
    
    # Show latest
    if files:
        latest = files[0]
        with open(latest) as f:
            data = json.load(f)
        print(f"\nLatest run: {latest.name}")
        print(f"  - Sharpe (2bps): {data.get('oof_sharpe_2bps', 'N/A')}")
        print(f"  - Max Drawdown: {data.get('oof_max_drawdown', 'N/A')}")
    
    print()


if __name__ == "__main__":
    main()
