import os
import argparse
import time
from google.cloud import aiplatform
import vertexai  
from config.settings import PROJECT_ID, LOCATION

def deploy_model(
    model_resource_name=None,
    endpoint_name="hyperparameter-explainer-endpoint",
    machine_type="n1-standard-4",
    min_replicas=1,
    max_replicas=2,
    deployment_location=LOCATION
):
    """
    Deploy a fine-tuned Gemini model to Vertex AI Endpoint.
    
    Args:
        model_resource_name: Resource name of the fine-tuned model
        endpoint_name: Name for the endpoint
        machine_type: VM type for serving
        min_replicas: Minimum number of serving instances
        max_replicas: Maximum number of serving instances
        deployment_location: Google Cloud region for deployment
        
    Returns:
        Endpoint and deployed model IDs
    """
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=deployment_location)
    
    # If model_resource_name not provided, try to read from model_info.txt
    if model_resource_name is None:
        model_info_path = os.path.join(os.path.dirname(__file__), "model_info.txt")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                for line in f:
                    if line.startswith("MODEL_RESOURCE_NAME="):
                        model_resource_name = line.strip().split("=")[1]
                        break
    
    if model_resource_name is None:
        raise ValueError(
            "Model resource name not provided and model_info.txt not found. "
            "Please specify a model resource name."
        )
    
    # Get timestamp for unique endpoint naming
    timestamp = time.strftime("%Y%m%d%H%M%S")
    display_name = f"{endpoint_name}-{timestamp}"
    
    print(f"🚀 Deploying fine-tuned model to endpoint: {display_name}")
    print(f"Model resource name: {model_resource_name}")
    
    # Get the model
    model = aiplatform.Model(model_resource_name)
    
    # Create or get an endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"',
        order_by="create_time desc",
        project=PROJECT_ID,
        location=deployment_location
    )
    
    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            project=PROJECT_ID,
            location=deployment_location
        )
        print(f"Created new endpoint: {endpoint.display_name}")
    
    # Deploy the model to the endpoint
    deployed_model = endpoint.deploy(
        model=model,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        deploy_request_timeout=1800,  # 30 minutes
        service_account=None  # Will use default service account
    )
    
    print(f"✅ Model deployed successfully!")
    print(f"Endpoint ID: {endpoint.name}")
    print(f"Deployed model ID: {deployed_model.id}")
    
    # Save endpoint information to file
    endpoint_info_path = os.path.join(os.path.dirname(__file__), "endpoint_info.txt")
    with open(endpoint_info_path, "w") as f:
        f.write(f"ENDPOINT_ID={endpoint.name}\n")
        f.write(f"DEPLOYED_MODEL_ID={deployed_model.id}\n")
        f.write(f"PROJECT_ID={PROJECT_ID}\n")
        f.write(f"LOCATION={deployment_location}\n")
    
    print(f"Endpoint information saved to {endpoint_info_path}")
    
    # Test endpoint with a sample request
    test_prediction(endpoint, model_resource_name)
    
    return {
        "endpoint_id": endpoint.name,
        "deployed_model_id": deployed_model.id
    }

def test_prediction(endpoint, model_name):
    """Test the deployed endpoint with a sample request"""
    print("\n🧪 Testing endpoint with a sample request...")
    
    sample_input = {
        "instances": [{
            "input_text": "CODE:\n```python\nmodel.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')\n```\n\nANALYZE HYPERPARAMETER: learning_rate = 0.001"
        }]
    }
    
    try:
        response = endpoint.predict(instances=sample_input["instances"])
        print("✅ Test prediction successful!")
        print("Response:")
        print(response.predictions[0])
    except Exception as e:
        print(f"❌ Test prediction failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a fine-tuned Gemini model")
    parser.add_argument("--model", help="Resource name of the fine-tuned model")
    parser.add_argument("--endpoint", default="hyperparameter-explainer-endpoint", help="Name for the endpoint")
    parser.add_argument("--machine_type", default="n1-standard-4", help="VM type for serving")
    parser.add_argument("--min_replicas", type=int, default=1, help="Minimum number of replicas")
    parser.add_argument("--max_replicas", type=int, default=2, help="Maximum number of replicas")
    
    args = parser.parse_args()
    
    deploy_model(
        model_resource_name=args.model,
        endpoint_name=args.endpoint,
        machine_type=args.machine_type,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas
    )