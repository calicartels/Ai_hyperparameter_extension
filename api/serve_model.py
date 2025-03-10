import os
import re
import json
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import aiplatform
from config.settings import PROJECT_ID, LOCATION

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for endpoint configuration
endpoint_id = None
endpoint = None
project_id = PROJECT_ID
location = LOCATION

def initialize_endpoint():
    """Initialize the endpoint connection"""
    global endpoint, endpoint_id
    
    # Try to read endpoint info from file
    endpoint_info_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model_finetuning",
        "endpoint_info.txt"
    )
    
    if os.path.exists(endpoint_info_path):
        with open(endpoint_info_path, "r") as f:
            for line in f:
                if line.startswith("ENDPOINT_ID="):
                    endpoint_id = line.strip().split("=")[1]
                    break
    
    if not endpoint_id:
        raise ValueError(
            "Endpoint ID not found. Please deploy the model first or set ENDPOINT_ID environment variable."
        )
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Get the endpoint
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    print(f"✓ Connected to endpoint: {endpoint_id}")
    return endpoint

def parse_model_response(text):
    """Parse the structured response from the model"""
    # Extract explanation
    explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?:\n|$)', text)
    explanation = explanation_match.group(1) if explanation_match else ""
    
    # Extract typical range
    range_match = re.search(r'TYPICAL_RANGE:\s*(.*?)(?:\n|$)', text)
    typical_range = range_match.group(1) if range_match else ""
    
    # Extract alternatives
    alternatives = []
    alternatives_block = re.search(r'ALTERNATIVES:([\s\S]*?)(?:IMPACT:|$)', text)
    if alternatives_block:
        alternatives_text = alternatives_block.group(1).strip()
        alternatives_items = re.findall(r'-\s*([^:]+):\s*([^\n]+)', alternatives_text)
        
        for value, scenario in alternatives_items:
            alternatives.append({
                "value": value.strip(),
                "scenario": scenario.strip()
            })
    
    # Extract impact
    impact = {
        "convergence_speed": "medium",
        "generalization": "good",
        "stability": "medium"
    }
    
    impact_block = re.search(r'IMPACT:([\s\S]*?)(?:FRAMEWORK:|$)', text)
    if impact_block:
        impact_text = impact_block.group(1).strip()
        
        convergence_match = re.search(r'Convergence Speed:\s*([^\n]+)', impact_text)
        if convergence_match:
            impact["convergence_speed"] = convergence_match.group(1).strip()
        
        generalization_match = re.search(r'Generalization:\s*([^\n]+)', impact_text)
        if generalization_match:
            impact["generalization"] = generalization_match.group(1).strip()
        
        stability_match = re.search(r'Stability:\s*([^\n]+)', impact_text)
        if stability_match:
            impact["stability"] = stability_match.group(1).strip()
    
    # Extract framework, model_type, task
    framework_match = re.search(r'FRAMEWORK:\s*([^\n]+)', text)
    framework = framework_match.group(1).strip() if framework_match else "unknown"
    
    model_type_match = re.search(r'MODEL_TYPE:\s*([^\n]+)', text)
    model_type = model_type_match.group(1).strip() if model_type_match else "Neural Network"
    
    task_match = re.search(r'TASK:\s*([^\n]+)', text)
    task = task_match.group(1).strip() if task_match else "general_ml_task"
    
    return {
        "explanation": explanation,
        "typical_range": typical_range,
        "alternatives": alternatives,
        "impact": impact,
        "framework": framework,
        "model_type": model_type,
        "task": task
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "endpoint_id": endpoint_id})

@app.route('/analyze', methods=['POST'])
def analyze_parameter():
    """Analyze a hyperparameter with the fine-tuned model"""
    try:
        data = request.json
        
        # Extract data from request
        code = data.get('code', '')
        param_name = data.get('param_name', '')
        param_value = data.get('param_value', '')
        line_number = data.get('line_number')
        
        # Validate inputs
        if not code or not param_name or not param_value:
            return jsonify({
                "error": "Missing required parameters. Please provide 'code', 'param_name', and 'param_value'."
            }), 400
        
        # Format input for the model
        input_text = f"CODE:\n```python\n{code}\n```\n\nANALYZE HYPERPARAMETER: {param_name} = {param_value}"
        
        # Call the model
        response = endpoint.predict(
            instances=[{
                "input_text": input_text
            }]
        )
        
        # Parse the model response
        raw_response = response.predictions[0]
        parsed_response = parse_model_response(raw_response)
        
        # Add parameter details
        result = {
            "name": param_name,
            "value": param_value,
            "line_number": line_number,
            **parsed_response
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in analyze_parameter: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_parameters():
    """Detect hyperparameters in code"""
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
        
        # Simple regex-based parameter detection
        # In production, you might want to use your AST parser for better accuracy
        param_patterns = [
            (r'learning_rate\s*=\s*([^,\)\n]+)', 'learning_rate', 'optimizer'),
            (r'lr\s*=\s*([^,\)\n]+)', 'lr', 'optimizer'),
            (r'batch_size\s*=\s*([^,\)\n]+)', 'batch_size', 'training'),
            (r'epochs\s*=\s*([^,\)\n]+)', 'epochs', 'training'),
            (r'activation\s*=\s*([^,\)\n]+)', 'activation', 'activation_function'),
            (r'dropout\s*=\s*([^,\)\n]+)', 'dropout', 'regularization'),
            (r'optimizer\s*=\s*([^,\)\n]+)', 'optimizer', 'optimizer'),
            (r'kernel_size\s*=\s*([^,\)\n]+)', 'kernel_size', 'architecture'),
            (r'filters\s*=\s*([^,\)\n]+)', 'filters', 'architecture'),
            (r'units\s*=\s*([^,\)\n]+)', 'units', 'architecture')
        ]
        
        detected = []
        lines = code.split('\n')
        
        for pattern, param_name, param_type in param_patterns:
            for match in re.finditer(pattern, code):
                value = match.group(1).strip()
                
                # Get line number
                code_before = code[:match.start()]
                line_number = code_before.count('\n') + 1
                
                detected.append({
                    "name": param_name,
                    "value": value,
                    "param_type": param_type,
                    "line_number": line_number,
                    "char_start": match.start(),
                    "char_end": match.end()
                })
        
        return jsonify(detected)
    
    except Exception as e:
        print(f"Error in detect_parameters: {str(e)}")
        return jsonify({"error": str(e)}), 500

def main(port=8080):
    """Run the API server"""
    initialize_endpoint()
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the fine-tuned model API")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--endpoint", help="Endpoint ID (overrides endpoint_info.txt)")
    parser.add_argument("--project", help="Project ID (overrides settings.py)")
    parser.add_argument("--location", help="Location (overrides settings.py)")
    
    args = parser.parse_args()
    
    if args.endpoint:
        endpoint_id = args.endpoint
    
    if args.project:
        project_id = args.project
    
    if args.location:
        location = args.location
    
    main(port=args.port)