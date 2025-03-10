import os
import argparse
import time
from model_finetuning.data_preparation import prepare_finetuning_dataset
from model_finetuning.finetune_gemini import finetune_gemini
from model_finetuning.deploy_model import deploy_model

def run_complete_pipeline(
    input_file=None,
    model_name="gemini-1.0-pro",
    model_display_name="hyperparameter-explainer",
    epochs=5,
    batch_size=8,
    learning_rate=1e-4,
    deploy=True,
    machine_type="n1-standard-4",
    prepare_only=False
):
    """
    Run the complete fine-tuning pipeline from data preparation to deployment.
    
    Args:
        input_file: Path to hyperparameters JSON file
        model_name: Base model to fine-tune
        model_display_name: Display name for the tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        deploy: Whether to deploy the model after fine-tuning
        machine_type: VM type for serving
        prepare_only: If True, just prepare the config file and don't start training
    """
    start_time = time.time()
    print("🚀 Starting HyperParam Explainer Model Fine-tuning Pipeline")
    
    # Step 1: Prepare fine-tuning dataset
    print("\n📊 Step 1: Preparing fine-tuning dataset...")
    data_paths = prepare_finetuning_dataset(input_file=input_file)
    
    # Step 2: Fine-tune Gemini model
    print("\n🧠 Step 2: Fine-tuning Gemini model...")
    result = finetune_gemini(
        train_data_path=data_paths.get("train_gcs"),
        validation_data_path=data_paths.get("validation_gcs"),
        model_name=model_name,
        model_display_name=model_display_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        prepare_only=prepare_only
    )
    
    # If we're just preparing, no need to continue
    if prepare_only:
        print("\n✅ Preparation complete! Use the commands above to start fine-tuning.")
        execution_time = time.time() - start_time
        print(f"Total preparation time: {execution_time:.2f} seconds")
        return {
            "preparation_time": execution_time,
            "config_path": result if isinstance(result, str) else None
        }
    
    # Step 3: Deploy the model if requested
    if deploy and not prepare_only:
        print("\n🚀 Step 3: Deploying fine-tuned model...")
        try:
            deployment_info = deploy_model(
                model_resource_name=result,
                machine_type=machine_type
            )
            
            # Print deployment information
            print("\n✅ Model successfully deployed:")
            print(f"Endpoint ID: {deployment_info['endpoint_id']}")
            print(f"Deployed Model ID: {deployment_info['deployed_model_id']}")
            
            # Print instructions for starting the API server
            print("\n📋 To start the API server, run:")
            print(f"python -m api.serve_model --endpoint {deployment_info['endpoint_id']}")
        except Exception as e:
            print(f"\n❌ Error deploying model: {str(e)}")
            print("You can deploy the model manually after fine-tuning completes using:")
            print("python -m model_finetuning.deploy_model --model [MODEL_RESOURCE_NAME]")
            deployment_info = None
    else:
        deployment_info = None
    
    # Print summary and execution time
    execution_time = time.time() - start_time
    print(f"\n✅ Fine-tuning pipeline completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    return {
        "model_resource_name": result if not isinstance(result, str) or "projects/" in result else None,
        "config_path": result if isinstance(result, str) and "projects/" not in result else None,
        "deployment_info": deployment_info,
        "execution_time": execution_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete fine-tuning pipeline")
    parser.add_argument("--input_file", help="Path to hyperparameters JSON file")
    parser.add_argument("--model_name", default="gemini-1.0-pro", help="Base model to fine-tune")
    parser.add_argument("--display_name", default="hyperparameter-explainer", help="Display name for tuned model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--no_deploy", action="store_true", help="Skip model deployment")
    parser.add_argument("--machine_type", default="n1-standard-4", help="VM type for serving")
    parser.add_argument("--prepare_only", action="store_true", help="Just prepare config, don't start training")
    
    args = parser.parse_args()
    
    run_complete_pipeline(
        input_file=args.input_file,
        model_name=args.model_name,
        model_display_name=args.display_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        deploy=not args.no_deploy,
        machine_type=args.machine_type,
        prepare_only=args.prepare_only
    )