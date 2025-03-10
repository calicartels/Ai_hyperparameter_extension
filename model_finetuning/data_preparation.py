import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
from config.settings import LOCAL_OUTPUT_DIR, PROJECT_ID, OUTPUT_BUCKET

def prepare_finetuning_dataset(input_file=None, output_dir=None, gcs_upload=True):
    """
    Prepare dataset for fine-tuning Gemini 1.0 Pro on Vertex AI.
    
    Args:
        input_file: Path to hyperparameters JSON file, defaults to hyperparameters_target_format.json
        output_dir: Directory to save prepared datasets, defaults to LOCAL_OUTPUT_DIR
        gcs_upload: Whether to upload datasets to GCS bucket
        
    Returns:
        Dictionary with paths to the created files
    """
    # Set default paths
    if input_file is None:
        input_file = os.path.join(LOCAL_OUTPUT_DIR, "hyperparameters_target_format.json")
    
    if output_dir is None:
        output_dir = os.path.join(LOCAL_OUTPUT_DIR, "finetuning")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Preparing fine-tuning dataset from {input_file}...")
    
    # Load hyperparameter data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} hyperparameter entries")
    
    # Create training examples in Vertex AI tuning format
    examples = []
    skipped_count = 0
    
    for entry in data:
        context = entry['code_snippet']
        framework = entry.get('framework', 'unknown')
        model_type = entry.get('model_type', 'Neural Network')
        task = entry.get('task', 'general_ml_task')
        
        for param in entry['hyperparameters']:
            try:
                # Safely get impact values with defaults if missing
                impact = param.get('impact', {})
                convergence_speed = impact.get('convergence_speed', 'medium')
                generalization = impact.get('generalization', 'good')
                stability = impact.get('stability', 'medium')
                
                # Format for Gemini tuning
                input_text = f"CODE:\n```python\n{context}\n```\n\nANALYZE HYPERPARAMETER: {param['name']} = {param['value']}"
                
                # Create the standard structured output format
                alternatives = param.get('alternatives', [])
                if not alternatives:
                    alternatives = [
                        {"value": "default", "scenario": "Standard use case"},
                        {"value": "alternative", "scenario": "Alternative case"}
                    ]
                
                alternatives_text = "\n".join([f"- {alt.get('value', 'value')}: {alt.get('scenario', 'use case')}" 
                                            for alt in alternatives])
                
                explanation = param.get('explanation', f"Controls the {param['name']} parameter in the model.")
                typical_range = param.get('typical_range', "Varies based on model architecture")
                
                output_text = f"""EXPLANATION: {explanation}
TYPICAL_RANGE: {typical_range}
ALTERNATIVES:
{alternatives_text}
IMPACT:
Convergence Speed: {convergence_speed}
Generalization: {generalization}
Stability: {stability}
FRAMEWORK: {framework}
MODEL_TYPE: {model_type}
TASK: {task}"""
                
                # Create example dictionary in the format expected by Vertex AI
                example = {
                    "input_text": input_text,
                    "output_text": output_text
                }
                
                examples.append(example)
                
            except KeyError as e:
                skipped_count += 1
                print(f"Warning: Skipping example due to missing key: {e}")
                continue
    
    print(f"Created {len(examples)} fine-tuning examples (skipped {skipped_count} with missing data)")
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(examples)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    # Split into training and validation sets (90% train, 10% validation)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    print(f"Split into {len(train_df)} training examples and {len(val_df)} validation examples")
    
    # Save as JSONL files (required format for Vertex AI)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "validation.jsonl")
    
    train_df.to_json(train_path, orient='records', lines=True)
    val_df.to_json(val_path, orient='records', lines=True)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved validation data to {val_path}")
    
    # Upload to Google Cloud Storage if requested
    if gcs_upload:
        try:
            gcs_train_path = f"gs://{OUTPUT_BUCKET}/finetuning/train.jsonl"
            gcs_val_path = f"gs://{OUTPUT_BUCKET}/finetuning/validation.jsonl"
            
            storage_client = storage.Client(project=PROJECT_ID)
            bucket = storage_client.bucket(OUTPUT_BUCKET)
            
            # Upload training file
            blob = bucket.blob("finetuning/train.jsonl")
            blob.upload_from_filename(train_path)
            
            # Upload validation file
            blob = bucket.blob("finetuning/validation.jsonl")
            blob.upload_from_filename(val_path)
            
            print(f"Uploaded training data to {gcs_train_path}")
            print(f"Uploaded validation data to {gcs_val_path}")
            
            return {
                "train_local": train_path,
                "validation_local": val_path,
                "train_gcs": gcs_train_path,
                "validation_gcs": gcs_val_path
            }
        except Exception as e:
            print(f"Warning: Failed to upload to GCS: {e}")
            print("Continuing with local files only")
    
    return {
        "train_local": train_path,
        "validation_local": val_path
    }

if __name__ == "__main__":
    # Run the data preparation as a standalone script
    prepare_finetuning_dataset()