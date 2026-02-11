import argparse
import os
from datasets import load_dataset


def process_narrativeqa_dataset(output_dir: str):
    
    # Load dataset
    print("\nLoading dataset: deepmind/narrativeqa")
    ds = load_dataset("deepmind/narrativeqa")
    print(f"Dataset loaded. Splits: {list(ds.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save raw data for each split
    for split_name in ds.keys():
        print(f"\n{'='*60}")
        print(f"Processing split: {split_name}")
        print(f"{'='*60}")
        
        dataset = ds[split_name]
        print(f"Sample count: {len(dataset)}")
        
        # Inspect schema
        if len(dataset) > 0:
            print(f"Columns: {dataset.column_names}")
        
        # Save raw data as parquet
        print("\nSaving raw data as parquet...")
        output_path = os.path.join(output_dir, f"{split_name}.parquet")
        dataset.to_parquet(output_path)
        
        print(f"[OK] Saved to: {output_path}")
        print(f"     File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    print(f"\n{'='*60}")
    print("All splits processed.")
    print(f"{'='*60}")
    print(f"Data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and save the raw NarrativeQA dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/narrativeqa",
        help="Output directory path (default: data/narrativeqa)",
    )
    args = parser.parse_args()
    
    process_narrativeqa_dataset(args.output_dir)


if __name__ == "__main__":
    main()
