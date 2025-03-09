from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import GenerativeModel as PreviewGenerativeModel
import json
import time

def initialize_gemini():
    """Initialize Gemini model for hyperparameter enrichment"""
    try:
        # First try the stable version
        model = GenerativeModel("gemini-pro")
        return model
    except:
        # Fall back to preview if needed
        model = PreviewGenerativeModel("gemini-pro")
        return model

def enrich_hyperparameter(model, hyperparameter_data, full_code=None):
    """Use Gemini to enrich hyperparameter data with explanations and alternatives"""
    param_name = hyperparameter_data["name"]
    param_value = hyperparameter_data["value"]
    code_context = hyperparameter_data["code_context"]
    param_type = hyperparameter_data["param_type"]
    
    # Get model_type and task from hyperparameter_data
    model_type = hyperparameter_data.get("model_type", "Neural Network")
    task = hyperparameter_data.get("task", "general_ml_task")
    framework = hyperparameter_data.get("framework", "unknown")
    
    # Get official documentation if available
    official_docs = hyperparameter_data.get("official_docs", {})
    official_description = official_docs.get("description", "")
    typical_range = official_docs.get("typical_range", "")
    
    # Add more complete model analysis
    prompt = f"""
    As an ML expert, analyze this hyperparameter in detail:
    
    Parameter: {param_name} 
    Current Value: {param_value}
    Parameter Type: {param_type}
    Framework: {framework}
    Model Type: {model_type}
    ML Task: {task}
    
    Official Documentation:
    {official_description}
    
    Official Typical Range: {typical_range}
    
    Code Context:
    ```python
    {code_context}
    ```
    
    Respond with a JSON object ONLY (no explanations outside the JSON) containing:
      "framework": "{framework if framework != 'unknown' else 'Infer the likely framework (tensorflow|pytorch|sklearn|other)'}",
      "model_type": "{model_type if model_type != 'Neural Network' else 'Infer the specific model architecture'}",
      "task": "{task if task != 'general_ml_task' else 'Infer the specific ML task'}",
      "param_type": "{param_type if param_type != 'other' else 'Infer specific parameter category'}",
      "explanation": "A concise (1-2 sentences) explanation of what this parameter controls and its practical impact",
      "typical_range": "A specific numerical or categorical range that's practical for this parameter",
      "alternatives": [
        {{"value": "specific_value_1", "scenario": "Concise description of when to use this value (5-10 words)"}},
        {{"value": "specific_value_2", "scenario": "Concise description of when to use this value (5-10 words)"}},
        {{"value": "specific_value_3", "scenario": "Concise description of when to use this value (5-10 words)"}}
      ],
      "impact": {{
        "convergence_speed": "fast|medium|slow",
        "generalization": "poor|good|excellent",
        "stability": "low|medium|high"
      }}
    
    Incorporate information from the official documentation when relevant. Be as specific as possible.
    """
    
    try:
        # Call Gemini and get result
        response = model.generate_content(prompt)
        
        # Parse JSON response
        try:
            # First try to extract JSON directly from the text
            result = json.loads(response.text)
        except:
            # If that fails, try to extract JSON from within markdown code blocks
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Last resort, try to find any JSON-like structure
                json_match = re.search(r'{.*}', response.text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("Could not extract JSON from response")
        
        # Update the hyperparameter data with the enriched information
        hyperparameter_data.update(result)
        return hyperparameter_data
    
    except Exception as e:
        print(f"Error enriching {param_name}: {str(e)}")
        # Provide fallback values if enrichment fails
        hyperparameter_data.update({
            "explanation": f"Controls the {param_name} parameter in the model",
            "typical_range": "Varies depending on model architecture and dataset",
            "alternatives": [
                {"value": "lower", "scenario": "More conservative training"},
                {"value": "higher", "scenario": "Faster training"},
                {"value": "default", "scenario": "General use cases"}
            ],
            "impact": {
                "convergence_speed": "medium",
                "generalization": "good",
                "stability": "medium"
            }
        })
        return hyperparameter_data

def batch_enrich_hyperparameters(metadata_list, batch_size=5):
    """Process a batch of code samples for hyperparameter enrichment"""
    model = initialize_gemini()
    enriched_samples = []
    
    for i, metadata in enumerate(metadata_list):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(metadata_list)}")
        
        # Deep copy to avoid modifying the original
        enriched_metadata = json.loads(json.dumps(metadata))
        
        # Add model_type and task to each hyperparameter for context
        for hyperparam in enriched_metadata["hyperparameters"]:
            hyperparam["model_type"] = enriched_metadata["model_type"]
            hyperparam["task"] = enriched_metadata["task"]
        
        # Enrich each hyperparameter
        for j, hyperparam in enumerate(enriched_metadata["hyperparameters"]):
            # Rate limiting to avoid API throttling
            if j > 0 and j % batch_size == 0:
                time.sleep(2)
                
            enriched_hyperparam = enrich_hyperparameter(model, hyperparam)
            enriched_metadata["hyperparameters"][j] = enriched_hyperparam
        
        enriched_samples.append(enriched_metadata)
    
    return enriched_samples