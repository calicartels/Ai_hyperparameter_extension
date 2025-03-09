import os
import json
import time
from google.cloud import storage
from data_collection.bigquery_collector import run_tensorflow_query
from data_processing.extract_params import process_code_file, transform_to_target_format
from ai_enrichment.gemini_enrichment import batch_enrich_hyperparameters
from config.settings import LOCAL_OUTPUT_DIR, OUTPUT_BUCKET, PROJECT_ID

def process_and_enrich_data(sample_limit=100, process_limit=20):
    """Main processing pipeline to extract and enrich hyperparameters"""
    start_time = time.time()
    print("🚀 Starting hyperparameter extraction and enrichment pipeline...")
    
    # Step 1: Initialize documentation extractor
    print("\n📚 Initializing documentation database...")
    from data_processing.documentation_extractor import DocumentationExtractor
    doc_extractor = DocumentationExtractor()
    
    # Step 2: Collect TensorFlow code samples from GitHub via BigQuery
    print("\n📊 Collecting data from BigQuery...")
    code_samples = run_tensorflow_query(limit=sample_limit)
    print(f"Retrieved {len(code_samples)} code samples")
    
    # Step 3: Extract hyperparameters using AST parsing
    print("\n🔍 Extracting hyperparameters...")
    metadata_list = []
    
    for i, sample in enumerate(code_samples[:process_limit]):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{min(len(code_samples), process_limit)}")
        
        # Process the code file to extract hyperparameters
        metadata = process_code_file(
            sample["content"], 
            sample["repo_name"], 
            sample["file_path"]
        )
        
        if metadata and metadata["hyperparameters"]:
            metadata_list.append(metadata)
    
    print(f"Extracted metadata from {len(metadata_list)} samples with {sum(len(m['hyperparameters']) for m in metadata_list)} hyperparameters")
    
    # Step 4: Enrich hyperparameters using Vertex AI Gemini
    print("\n🧠 Enriching hyperparameters with Gemini...")
    enriched_metadata_list = batch_enrich_hyperparameters(metadata_list)
    
    # Step 5: Transform to target format with one hyperparameter per entry
    print("\n🔄 Transforming to target format...")
    target_format_list = transform_to_target_format(enriched_metadata_list)
    
    print(f"Created {len(target_format_list)} entries in target format")
    
    # Step 6: Save the results locally and to Google Cloud Storage
    print("\n💾 Saving transformed hyperparameter data...")
    
    # Save locally
    output_file = os.path.join(LOCAL_OUTPUT_DIR, "hyperparameters_target_format.json")
    with open(output_file, 'w') as f:
        json.dump(target_format_list, f, indent=2)
    
    # Save to Google Cloud Storage
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(OUTPUT_BUCKET)
        blob = bucket.blob("hyperparameters_target_format.json")
        blob.upload_from_filename(output_file)
        print(f"Uploaded to gs://{OUTPUT_BUCKET}/hyperparameters_target_format.json")
    except Exception as e:
        print(f"Error uploading to Cloud Storage: {str(e)}")
    
    # Print summary
    print("\n✅ Hyperparameter Extraction and Enrichment Summary:")
    print(f"  - Raw samples collected: {len(code_samples)}")
    print(f"  - Samples with hyperparameters: {len(metadata_list)}")
    print(f"  - Total hyperparameters extracted: {sum(len(m['hyperparameters']) for m in metadata_list)}")
    print(f"  - Total target format entries: {len(target_format_list)}")
    print(f"  - Execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    process_and_enrich_data()