import ast
import astor  
import os
import json
from typing import Dict, List, Any, Tuple
import uuid
from data_processing.documentation_extractor import DocumentationExtractor


# Define hyperparam context extractor using AST parsing
class HyperparameterExtractor(ast.NodeVisitor):
    """Extract hyperparameters from Python code using AST parsing"""
    
    def __init__(self):
        self.hyperparameters = []
        self.source_lines = []
        self.framework = "unknown"
    
    def visit_Call(self, node):
        """Detect function calls that might contain hyperparameters"""
        # Framework detection
        class_name = None
        if hasattr(node, 'func') and isinstance(node.func, ast.Attribute):
            if node.func.attr in ['compile', 'fit', 'Sequential', 'Model']:
                self.framework = "tensorflow"
            elif node.func.attr in ['SGD', 'Adam', 'RMSprop']:
                # Try to detect PyTorch
                if hasattr(node.func, 'value') and hasattr(node.func.value, 'attr'):
                    if node.func.value.attr == 'optim':
                        self.framework = "pytorch"
            
            # Extract class name for documentation lookup
            class_name = node.func.attr
        
        # Extract keyword arguments which could be hyperparameters
        for kw in node.keywords:
            if kw.arg in ['learning_rate', 'batch_size', 'epochs', 'optimizer', 
                        'dropout', 'activation', 'momentum', 'beta_1', 'beta_2',
                        'weight_decay', 'filters', 'kernel_size', 'units',
                        'num_layers', 'hidden_size', 'lr']:
                
                # Get the value as a string
                if isinstance(kw.value, ast.Num):
                    value = str(kw.value.n)
                elif isinstance(kw.value, ast.Str):
                    value = kw.value.s
                else:
                    # For complex values, convert AST back to code
                    value = astor.to_source(kw.value).strip()
                
                # Get line and column info
                line_num = kw.value.lineno if hasattr(kw.value, 'lineno') else node.lineno
                col_start = kw.value.col_offset if hasattr(kw.value, 'col_offset') else 0
                
                # Calculate end column based on value length
                col_end = col_start + len(str(value))
                
                # Extract context (surrounding lines)
                context_start = max(0, line_num - 5)
                context_end = min(len(self.source_lines), line_num + 5)
                context_code = '\n'.join(self.source_lines[context_start:context_end])
                
                # Create hyperparameter entry
                hyperparam = {
                    "name": kw.arg,
                    "value": value,
                    "line_number": line_num,
                    "char_start": col_start,
                    "char_end": col_end,
                    "code_context": context_code,
                    "param_type": self._infer_param_type(kw.arg),
                    "class_name": class_name  # Add class context
                }
                
                # Enrich with documentation if available
                if class_name and self.framework != "unknown":
                    hyperparam = enrich_with_docs(hyperparam, self.framework, class_name)
                
                self.hyperparameters.append(hyperparam)
        
        # Continue traversing the tree
        self.generic_visit(node)
    
    def _infer_param_type(self, param_name):
        """Infer parameter type based on name"""
        optimizer_params = ['learning_rate', 'lr', 'momentum', 'beta_1', 'beta_2']
        architecture_params = ['units', 'filters', 'kernel_size', 'hidden_size', 'num_layers']
        regularization_params = ['dropout', 'weight_decay', 'l1', 'l2']
        training_params = ['batch_size', 'epochs']
        
        if param_name in optimizer_params:
            return "optimizer"
        elif param_name in architecture_params:
            return "architecture"
        elif param_name in regularization_params:
            return "regularization"
        elif param_name in training_params:
            return "training"
        else:
            return "other"
    
    def infer_model_type(self, code):
        """Infer model type from code with more specificity"""
        code_lower = code.lower()
        
        # Check for CNN indicators
        if 'conv2d' in code_lower or 'conv1d' in code_lower or 'convolution' in code_lower:
            return 'CNN'
        # Check for RNN indicators
        elif 'lstm' in code_lower:
            return 'LSTM'
        elif 'gru' in code_lower:
            return 'GRU'
        elif 'rnn' in code_lower:
            return 'RNN'
        # Check for Transformer indicators
        elif 'transformer' in code_lower or 'attention' in code_lower:
            return 'Transformer'
        # Check for other specific architectures
        elif 'densenet' in code_lower:
            return 'DenseNet'
        elif 'resnet' in code_lower:
            return 'ResNet'
        elif 'mobilenet' in code_lower:
            return 'MobileNet'
        elif 'dense' in code_lower or 'fully connected' in code_lower:
            return 'Dense Neural Network'
        else:
            return 'Neural Network'
    
    def infer_task(self, code):
        """Infer ML task with more specificity"""
        code_lower = code.lower()
        
        # Classification tasks
        if 'imagenet' in code_lower or 'cifar' in code_lower or 'mnist' in code_lower:
            return 'image_classification'
        elif 'object detection' in code_lower or 'yolo' in code_lower or 'ssd' in code_lower:
            return 'object_detection'
        elif 'sentiment' in code_lower or 'text classification' in code_lower:
            return 'text_classification'
        elif 'categorical_crossentropy' in code_lower or 'classification' in code_lower:
            return 'classification'
        
        # Regression tasks
        elif 'mean_squared_error' in code_lower or 'mse' in code_lower or 'regression' in code_lower:
            return 'regression'
        
        # Generation tasks
        elif 'generator' in code_lower or 'gan' in code_lower:
            return 'image_generation'
        elif 'text generation' in code_lower or 'language model' in code_lower:
            return 'text_generation'
        
        # Sequence tasks
        elif 'sequence' in code_lower or 'time series' in code_lower:
            return 'sequence_prediction'
        
        # Default case
        else:
            return 'general_ml_task'

def process_code_file(content, repo_name, file_path):
    """Process a Python code file to extract hyperparameters"""
    try:
        # Parse the code into an AST
        tree = ast.parse(content)
        
        # Create source lines for context extraction
        source_lines = content.splitlines()
        
        # Create an extractor and visit the AST
        extractor = HyperparameterExtractor()
        extractor.source_lines = source_lines
        extractor.visit(tree)
        
        # Generate a unique ID
        framework_prefix = extractor.framework[:2] if extractor.framework != "unknown" else "ml"
        unique_id = f"{framework_prefix}_{uuid.uuid4().hex[:8]}"
        
        # Create source URL
        source_url = f"https://github.com/{repo_name}/blob/master/{file_path}"
        
        # Infer model type and task
        model_type = extractor.infer_model_type(content)
        task = extractor.infer_task(content)
        
        # Create the metadata for the code sample
        metadata = {
            "id": unique_id,
            "framework": extractor.framework,
            "source_url": source_url,
            "hyperparameters": extractor.hyperparameters,
            "model_type": model_type,
            "task": task,
            "code_snippet": content
        }
        
        return metadata
    except SyntaxError:
        # Handle files with syntax errors
        return None
def enrich_with_docs(hyperparameter, framework, class_name):
    """Enrich hyperparameter with documentation if available"""
    doc_extractor = DocumentationExtractor()
    
    param_info = doc_extractor.get_param_info(framework, class_name, hyperparameter["name"])
    
    if param_info:
        # Add documentation info
        hyperparameter["official_docs"] = {
            "description": param_info.get("description", ""),
            "default": param_info.get("default", ""),
            "type": param_info.get("type", ""),
            "common_values": param_info.get("common_values", []),
            "typical_range": param_info.get("typical_range", "")
        }
        
        # If there's a typical range in the docs, use it
        if "typical_range" in param_info:
            hyperparameter["typical_range"] = param_info["typical_range"]
    
    return hyperparameter

def get_parameter_line(code, param):
    """Extract just the line(s) relevant to the hyperparameter"""
    lines = code.splitlines()
    try:
        # Get line where parameter is defined
        param_line = lines[param["line_number"] - 1]
        
        # If line is too short, get more context
        if len(param_line.strip()) < 20:
            # Get a few lines before and after
            start = max(0, param["line_number"] - 2)
            end = min(len(lines), param["line_number"] + 1)
            param_line = "\n".join(lines[start:end])
        
        return param_line
    except (IndexError, KeyError):
        # Fall back to using code_context if available
        if "code_context" in param:
            return param["code_context"]
        
        # Last resort, return a reasonable snippet
        return f"{param['name']}={param['value']}"

def infer_dataset_size(code):
    """Try to infer dataset size from code context"""
    code_lower = code.lower()
    
    # Look for common indicators
    if "imagenet" in code_lower or "large" in code_lower or "big" in code_lower:
        return "large"
    elif "mnist" in code_lower or "small" in code_lower:
        return "small"
    elif "medium" in code_lower:
        return "medium"
    else:
        return "unspecified"

def transform_to_target_format(metadata_list):
    """Transform enriched metadata to target format with one hyperparameter per entry"""
    target_format_list = []
    counter_map = {}  # To keep track of sequential IDs by framework and model type
    
    for metadata in metadata_list:
        framework = metadata["framework"]
        model_type = metadata["model_type"]
        
        # Create a more specific prefix
        if framework == "tensorflow" or framework == "keras":
            framework_prefix = "tf"
        elif framework == "pytorch":
            framework_prefix = "pt"
        elif framework == "sklearn":
            framework_prefix = "skl"
        else:
            framework_prefix = "ml"
            
        # Create a model type short code
        if model_type == "CNN":
            model_code = "cnn"
        elif model_type in ["RNN", "LSTM", "GRU"]:
            model_code = "rnn"
        elif "Transformer" in model_type:
            model_code = "tfm"
        elif model_type == "Dense Neural Network":
            model_code = "dnn"
        elif model_type == "Neural Network":
            model_code = "nn"
        else:
            model_code = "gen"  # generic
            
        prefix = f"{framework_prefix}_{model_code}"
        
        # Initialize counter if needed
        if prefix not in counter_map:
            counter_map[prefix] = 1
        
        # Transform each hyperparameter into its own entry
        for param in metadata["hyperparameters"]:
            # Skip if we couldn't extract key information
            if not param.get("name") or not param.get("value"):
                continue
                
            # Get the line containing this hyperparameter
            param_line = get_parameter_line(metadata["code_snippet"], param)
            
            # Generate sequential ID
            seq_id = f"{prefix}_{counter_map[prefix]:03d}"
            counter_map[prefix] += 1
            
            # Add dataset size (inferring if possible)
            dataset_size = infer_dataset_size(metadata["code_snippet"])
            
            # Determine explanation and typical range, prioritizing official docs
            explanation = ""
            typical_range = ""
            alternatives = []
            impact = {
                "convergence_speed": "medium",
                "generalization": "good",
                "stability": "medium"
            }
            
            # Use official docs if available
            if "official_docs" in param:
                if param["official_docs"].get("description"):
                    explanation = param["official_docs"]["description"]
                if param["official_docs"].get("typical_range"):
                    typical_range = param["official_docs"]["typical_range"]
                if param["official_docs"].get("common_values"):
                    # Create basic alternatives from common values
                    for val in param["official_docs"]["common_values"]:
                        alternatives.append({
                            "value": val,
                            "scenario": f"Common value for {param['name']}"
                        })
            
            # If we have LLM-enriched data, use that to fill gaps or enhance
            if "explanation" in param and (not explanation or len(explanation) < 20):
                explanation = param.get("explanation", "")
            
            if "typical_range" in param and not typical_range:
                typical_range = param.get("typical_range", "")
                
            if "alternatives" in param and param["alternatives"] and (not alternatives or len(alternatives) < 2):
                alternatives = param.get("alternatives", [])
                
            if "impact" in param:
                impact = param.get("impact", impact)
                
            # Ensure we have at least some explanation
            if not explanation:
                explanation = f"Controls the {param['name']} parameter in the model."
                
            # Ensure we have some range
            if not typical_range:
                if param["name"] in ["learning_rate", "lr"]:
                    typical_range = "0.0001 to 0.1"
                elif param["name"] == "batch_size":
                    typical_range = "8 to 512"
                elif param["name"] == "epochs":
                    typical_range = "10 to 200"
                elif param["name"] == "dropout":
                    typical_range = "0.1 to 0.5"
                else:
                    typical_range = "Varies based on model architecture and dataset"
            
            # Ensure we have some alternatives
            if not alternatives or len(alternatives) < 2:
                if param["name"] in ["learning_rate", "lr"]:
                    alternatives = [
                        {"value": "0.1", "scenario": "Initial training with simple models"},
                        {"value": "0.01", "scenario": "Standard starting point for most models"},
                        {"value": "0.001", "scenario": "Fine-tuning or complex models"}
                    ]
                elif param["name"] == "batch_size":
                    alternatives = [
                        {"value": "16", "scenario": "Limited GPU memory or small datasets"},
                        {"value": "64", "scenario": "Balance between speed and generalization"},
                        {"value": "256", "scenario": "Faster training with large datasets"}
                    ]
                elif param["name"] == "dropout":
                    alternatives = [
                        {"value": "0.1", "scenario": "Minimal regularization needed"},
                        {"value": "0.3", "scenario": "Moderate regularization for most models"},
                        {"value": "0.5", "scenario": "Strong regularization for overfitting"}
                    ]
            
            # Create the target format entry
            target_entry = {
                "id": seq_id,
                "framework": metadata["framework"],
                "source_url": metadata["source_url"],
                "code_snippet": param_line,
                "hyperparameters": [
                    {
                        "name": param["name"],
                        "value": param["value"],
                        "line_number": param["line_number"],
                        "char_start": param["char_start"],
                        "char_end": param["char_end"],
                        "param_type": param["param_type"],
                        "explanation": explanation,
                        "typical_range": typical_range,
                        "alternatives": alternatives,
                        "impact": impact
                    }
                ],
                "model_type": metadata["model_type"],
                "task": metadata["task"] if metadata["task"] != "unknown" else "general_ml_task",
                "dataset_size": dataset_size
            }
            
            target_format_list.append(target_entry)
    
    return target_format_list