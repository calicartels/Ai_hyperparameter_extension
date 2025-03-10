import os
import time
import argparse
import json
from google.cloud import aiplatform
import vertexai
from config.settings import PROJECT_ID, LOCATION, OUTPUT_BUCKET

def finetune_gemini(
    train_data_path=None,
    validation_data_path=None,
    model_name="gemini-1.0-pro",
    model_display_name="hyperparameter-explainer",
    epochs=5,
    batch_size=8,
    learning_rate=1e-4,
    tuning_job_location=LOCATION,
    prepare_only=False
):
    """
    Fine-tune Gemini 1.0 Pro on Vertex AI with hyperparameter data.
    
    Args:
        train_data_path: GCS path to training data JSONL
        validation_data_path: GCS path to validation data JSONL
        model_name: Base model to fine-tune (gemini-1.0-pro)
        model_display_name: Display name for the tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        tuning_job_location: Google Cloud region for tuning job
        prepare_only: If True, just prepare the config file and don't start training
        
    Returns:
        Tuned model resource name or config file path if prepare_only=True
    """
    # Set default paths if not provided
    if train_data_path is None:
        train_data_path = f"gs://{OUTPUT_BUCKET}/finetuning/train.jsonl"
    
    if validation_data_path is None:
        validation_data_path = f"gs://{OUTPUT_BUCKET}/finetuning/validation.jsonl"
    
    # Get timestamp for unique job naming
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_name = f"{model_display_name}_{timestamp}"
    
    print(f"🚀 Preparing Gemini fine-tuning job: {job_name}")
    print(f"Training data: {train_data_path}")
    print(f"Validation data: {validation_data_path}")
    
    # Create a tuning job configuration for Vertex AI
    tuning_config = {
        "tuningJobSpec": {
            "baseModelId": model_name,
            "tuningJobId": job_name,
            "displayName": job_name,
            "trainingDatasetUri": train_data_path,
            "validationDatasetUri": validation_data_path,
            "hyperParameters": {
                "batchSize": batch_size,
                "learningRate": learning_rate,
                "epochs": epochs
            }
        },
        "project": PROJECT_ID,
        "location": tuning_job_location
    }
    
    # Save configuration to file
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"tuning_config_{timestamp}.json")
    
    with open(config_path, 'w') as config_file:
        json.dump(tuning_config, config_file, indent=2)
    
    print(f"✅ Saved tuning configuration to {config_path}")
    
    # Create model information file
    model_info_path = os.path.join(os.path.dirname(__file__), "model_info.txt")
    with open(model_info_path, "w") as f:
        f.write(f"TUNING_JOB_NAME={job_name}\n")
        f.write(f"PROJECT_ID={PROJECT_ID}\n")
        f.write(f"LOCATION={tuning_job_location}\n")
        f.write(f"CONFIG_PATH={config_path}\n")
    
    # If prepare_only, just output instructions
    if prepare_only:
        print("\n✨ Fine-tuning configuration prepared!")
        print("\nTo start fine-tuning using Google Cloud CLI, run:")
        print("--------------------------------------------------")
        print(f"gcloud ai tuning-jobs create \\")
        print(f"  --region={tuning_job_location} \\")
        print(f"  --display-name={job_name} \\")
        print(f"  --base-model={model_name} \\")
        print(f"  --training-data={train_data_path} \\")
        print(f"  --validation-data={validation_data_path}")
        print("--------------------------------------------------")
        print("\nOr alternatively:")
        print("--------------------------------------------------")
        print(f"gcloud ai tuning-jobs create \\")
        print(f"  --region={tuning_job_location} \\")
        print(f"  --config-file={config_path}")
        print("--------------------------------------------------")
        print("\nAfter fine-tuning completes, you can deploy the model with:")
        print("--------------------------------------------------")
        print("gcloud ai models deploy [MODEL_ID] \\")
        print(f"  --region={tuning_job_location} \\")
        print("  --display-name=hyperparameter-explainer-deployed")
        print("--------------------------------------------------")
        
        return config_path
    
    # Otherwise, attempt to launch the job
    try:
        print("\n⚠️ Fine-tuning Gemini models requires:")
        print("  - Proper GCP project setup and quotas")
        print("  - Payment method configured for your GCP account")
        print("  - Authentication set up correctly")
        print("  - Patience (typically 3-5 hours to complete)")
        
        print("\nWe'll provide instructions for both Python SDK and command-line approaches.")
        print("The command-line approach is often more reliable.")
        
        # Instructions for command-line
        print("\n📋 To fine-tune using Google Cloud CLI (recommended):")
        print("--------------------------------------------------")
        print(f"gcloud ai tuning-jobs create \\")
        print(f"  --region={tuning_job_location} \\")
        print(f"  --display-name={job_name} \\")
        print(f"  --base-model={model_name} \\")
        print(f"  --training-data={train_data_path} \\")
        print(f"  --validation-data={validation_data_path}")
        print("--------------------------------------------------")
        
        # Attempting Python SDK approach
        print("\n🔄 Attempting to start the job via Python SDK...")
        print("(This may fail if authentication isn't properly set up)")
        
        # Check if credentials are available
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            print("⚠️ GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
            print("Please set up authentication for Google Cloud:")
            print("export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json")
            raise EnvironmentError("Google Cloud credentials not found")
        
        # Try initializing SDK
        vertexai.init(project=PROJECT_ID, location=tuning_job_location)
        
        # Try submitting job using appropriate API
        from vertexai.preview.tuning import TuningJob
        
        tuning_job = TuningJob.create(
            model=model_name,
            training_data=train_data_path,
            validation_data=validation_data_path,
            tuning_task_type="text_generation",
            hyperparameters={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs
            }
        )
        
        print(f"✅ Fine-tuning job started successfully via SDK!")
        print(f"Job resource name: {tuning_job.resource_name}")
        
        # Update model info
        with open(model_info_path, "a") as f:
            f.write(f"JOB_RESOURCE_NAME={tuning_job.resource_name}\n")
        
        return tuning_job.resource_name
    
    except Exception as e:
        print(f"\n❌ Error starting job via SDK: {str(e)}")
        print("Please use the command-line approach outlined above.")
        print(f"Configuration has been saved to {config_path}")
        
        # Return the config path so the pipeline knows where to find it
        return config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemini 1.0 Pro on hyperparameter data")
    parser.add_argument("--train", help="GCS path to training data")
    parser.add_argument("--validation", help="GCS path to validation data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_name", default="gemini-1.0-pro", help="Base model to fine-tune")
    parser.add_argument("--display_name", default="hyperparameter-explainer", help="Display name for tuned model")
    parser.add_argument("--prepare_only", action="store_true", help="Just prepare config, don't start training")
    
    args = parser.parse_args()
    
    finetune_gemini(
        train_data_path=args.train,
        validation_data_path=args.validation,
        model_name=args.model_name,
        model_display_name=args.display_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        prepare_only=args.prepare_only
    )