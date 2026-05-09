import polars as pl
import numpy as np
from pathlib import Path
import config
from train_tcn import build_labels

def main():
    csv_path = Path(config.FEATURE_DUMP_PATH)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # 1. Use a LazyFrame to scan the file without loading it
    # This just creates a pointer to the file.
    q = pl.scan_csv(csv_path)
    
    total_shocks = 0
    total_rows = 0
    chunk_size = 1_000_000  # 1 million rows at a time (~200MB of RAM)
    
    print(f"Streaming {csv_path.name} in chunks of {chunk_size}...")

    # 2. Iterate through the file in chunks to keep RAM low
    # We use collect(streaming=True) to tell Polars to use its low-memory engine.
    df_full = q.collect(streaming=True)
    
    # Slice the large dataframe manually to stay compatible with build_labels
    for i in range(0, len(df_full), chunk_size):
        chunk = df_full.slice(i, chunk_size)
        
        # Convert ONLY this chunk to the format build_labels expects
        data_chunk = {col: chunk[col].to_numpy() for col in chunk.columns}
        
        # Run the detection on this small slice
        _, events = build_labels(data_chunk)
        
        total_shocks += len(events)
        total_rows += len(chunk)
        
        print(f"  Processed {total_rows:,} rows... Found {total_shocks} shocks so far.")

    print(f"\nFinal Results for {config.SYMBOL}:")
    print(f"  Total Rows:    {total_rows:,}")
    print(f"  Total Shocks:  {total_shocks}")
    print(f"  Threshold:     {config.MIN_CALIBRATION_EVENTS}")
    
    if total_shocks >= config.MIN_CALIBRATION_EVENTS:
        print("  Ready to train! Run: python train_tcn.py")
    else:
        print(f"  Still need {config.MIN_CALIBRATION_EVENTS - total_shocks} more shocks.")

if __name__ == "__main__":
    main()