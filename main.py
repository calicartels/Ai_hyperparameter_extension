import os
import time
from data_collection.bigquery_collector import run_tensorflow_query
from config.settings import LOCAL_OUTPUT_DIR

def main():
    """Main entry point to collect TensorFlow code samples with hyperparameters."""
    start_time = time.time()
    print("🚀 Starting TensorFlow code collection...")
    
    # Set sample limit (adjust based on your needs)
    sample_limit = 1000
    
    # Collect TensorFlow code samples from GitHub via BigQuery
    code_samples = run_tensorflow_query(limit=sample_limit)
    
    # Print summary
    print("\n✅ Data Collection Summary:")
    print(f"  - Total samples collected: {len(code_samples)}")
    print(f"  - Output directory: {LOCAL_OUTPUT_DIR}")
    print(f"  - Execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()