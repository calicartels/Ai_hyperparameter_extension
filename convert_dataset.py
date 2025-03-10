import json
import os
from google.cloud import storage
from config.settings import LOCAL_OUTPUT_DIR, OUTPUT_BUCKET, PROJECT_ID

def convert_to_gemini_format(input_jsonl, output_jsonl):
    """
    Convert dataset from VertexTextBison format to Gemini chat format with proper contents field.
    
    Args:
        input_jsonl: Path to input JSONL file with input_text/output_text format
        output_jsonl: Path to output JSONL file with contents format for Gemini
    """
    print(f"Converting dataset format from {input_jsonl} to {output_jsonl}")
    
    # Read the input file
    examples = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Read {len(examples)} examples from input file")
    
    # Convert to proper Gemini fine-tuning format
    gemini_examples = []
    for example in examples:
        # Extract user input and assistant output
        user_input = example.get('input_text', '')
        assistant_output = example.get('output_text', '')
        
        # Create correct format with contents and parts
        gemini_example = {
            "contents": [
                {"role": "user", "parts": [{"text": user_input}]},
                {"role": "model", "parts": [{"text": assistant_output}]}
            ]
        }
        
        gemini_examples.append(gemini_example)
    
    # Write to output file
    with open(output_jsonl, 'w') as f:
        for example in gemini_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Converted {len(gemini_examples)} examples to Gemini chat format with contents field")
    return gemini_examples

def convert_and_upload():
    """Convert both training and validation datasets and upload to GCS"""
    # Set paths
    train_input = os.path.join(LOCAL_OUTPUT_DIR, "finetuning", "train.jsonl")
    val_input = os.path.join(LOCAL_OUTPUT_DIR, "finetuning", "validation.jsonl")
    
    train_output = os.path.join(LOCAL_OUTPUT_DIR, "finetuning", "train_gemini_fixed.jsonl")
    val_output = os.path.join(LOCAL_OUTPUT_DIR, "finetuning", "validation_gemini_fixed.jsonl")
    
    # Convert files
    convert_to_gemini_format(train_input, train_output)
    convert_to_gemini_format(val_input, val_output)
    
    # Upload to GCS
    try:
        print("Uploading converted files to Google Cloud Storage...")
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(OUTPUT_BUCKET)
        
        # Upload training file
        blob = bucket.blob("finetuning/train_gemini_fixed.jsonl")
        blob.upload_from_filename(train_output)
        
        # Upload validation file
        blob = bucket.blob("finetuning/validation_gemini_fixed.jsonl")
        blob.upload_from_filename(val_output)
        
        # Print GCS paths
        train_gcs = f"gs://{OUTPUT_BUCKET}/finetuning/train_gemini_fixed.jsonl"
        val_gcs = f"gs://{OUTPUT_BUCKET}/finetuning/validation_gemini_fixed.jsonl"
        
        print(f"Training data uploaded to: {train_gcs}")
        print(f"Validation data uploaded to: {val_gcs}")
        
        print("\nTo fine-tune using the web console:")
        print("1. Go to Google Cloud Console -> Vertex AI -> Foundation Models")
        print("2. Select 'Gemini 1.5 Flash' and click 'Tune'")
        print("3. Use these paths for training and validation data:")
        print(f"   - Training data: {train_gcs}")
        print(f"   - Validation data: {val_gcs}")
        
        return {
            "train_gcs": train_gcs,
            "validation_gcs": val_gcs
        }
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        print("Continue with local files:")
        print(f"Training data: {train_output}")
        print(f"Validation data: {val_output}")
        
        return {
            "train_local": train_output,
            "validation_local": val_output
        }

# For manual upload to GCS
def show_manual_upload_instructions():
    """Show instructions for manually uploading files to GCS"""
    train_path = os.path.join(LOCAL_OUTPUT_DIR, "finetuning", "train_gemini_fixed.jsonl")
    val_path = os.path.join(LOCAL_OUTPUT_DIR, "finetuning", "validation_gemini_fixed.jsonl")
    
    print("\n===== MANUAL UPLOAD INSTRUCTIONS =====")
    print("1. Use the Google Cloud Console Storage browser:")
    print("   - Go to: https://console.cloud.google.com/storage/browser")
    print("   - Navigate to your bucket: hyperparameter_deep_learning")
    print("   - Create/navigate to folder: finetuning")
    print("   - Upload these files:")
    print(f"     * {train_path}")
    print(f"     * {val_path}")
    print("\n2. Or use gsutil commands:")
    print(f"   gsutil cp {train_path} gs://{OUTPUT_BUCKET}/finetuning/")
    print(f"   gsutil cp {val_path} gs://{OUTPUT_BUCKET}/finetuning/")
    print("\nAfter uploading, use these paths for fine-tuning:")
    print(f"   - Training data: gs://{OUTPUT_BUCKET}/finetuning/train_gemini_fixed.jsonl")
    print(f"   - Validation data: gs://{OUTPUT_BUCKET}/finetuning/validation_gemini_fixed.jsonl")
    print("=========================================")

if __name__ == "__main__":
    # Convert the files
    result = convert_and_upload()
    
    # Also show manual upload instructions if automatic upload failed
    if "train_local" in result:
        show_manual_upload_instructions()