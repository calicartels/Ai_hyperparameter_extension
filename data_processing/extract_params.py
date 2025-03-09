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
        self.imports = set()  # Track imports for better framework detection
    
    def visit_Import(self, node):
        """Detect framework imports"""
        for name in node.names:
            self.imports.add(name.name)
            if name.name == 'tensorflow' or name.name.startswith('tensorflow.'):
                self.framework = "tensorflow"
            elif name.name == 'torch' or name.name.startswith('torch.'):
                self.framework = "pytorch"
            elif name.name == 'sklearn' or name.name.startswith('sklearn.'):
                self.framework = "sklearn"
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Detect from-imports of frameworks"""
        if node.module:
            self.imports.add(node.module)
            if node.module == 'tensorflow' or node.module.startswith('tensorflow.'):
                self.framework = "tensorflow"
            elif node.module == 'torch' or node.module.startswith('torch.'):
                self.framework = "pytorch"
            elif node.module == 'sklearn' or node.module.startswith('sklearn.'):
                self.framework = "sklearn"
            # Check for keras imports as they indicate TensorFlow
            elif node.module == 'keras' or node.module.startswith('keras.'):
                self.framework = "tensorflow"
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Detect function calls that might contain hyperparameters"""
        # Framework detection from API calls
        class_name = None
        if hasattr(node, 'func') and isinstance(node.func, ast.Attribute):
            # Store potential TensorFlow indicators
            if node.func.attr in ['compile', 'fit', 'Sequential', 'Model']:
                self.framework = "tensorflow"
            # Store potential PyTorch indicators
            elif node.func.attr in ['SGD', 'Adam', 'RMSprop']:
                if hasattr(node.func, 'value') and hasattr(node.func.value, 'id'):
                    if node.func.value.id == 'optim':
                        self.framework = "pytorch"
                elif hasattr(node.func, 'value') and hasattr(node.func.value, 'attr'):
                    if node.func.value.attr == 'optim':
                        self.framework = "pytorch"
            
            # Extract class name for documentation lookup
            class_name = node.func.attr
        
        # Extract keyword arguments which could be hyperparameters
        for kw in node.keywords:
            if kw.arg in ['learning_rate', 'batch_size', 'epochs', 'optimizer', 
                        'dropout', 'activation', 'momentum', 'beta_1', 'beta_2',
                        'weight_decay', 'filters', 'kernel_size', 'units',
                        'num_layers', 'hidden_size', 'lr', 'epsilon', 'l1_regularization',
                        'l2_regularization', 'num_heads', 'max_length', 'padding',
                        'num_filters', 'num_epochs', 'n_estimators', 'max_depth']:
                
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
        """Infer parameter type based on name with more specificity"""
        type_mappings = {
            # Optimizer params
            'learning_rate': 'optimizer', 'lr': 'optimizer', 'momentum': 'optimizer',
            'beta_1': 'optimizer', 'beta_2': 'optimizer', 'epsilon': 'optimizer',
            'weight_decay': 'optimizer', 'rho': 'optimizer', 'betas': 'optimizer',
            
            # Architecture params
            'units': 'architecture', 'filters': 'architecture', 'kernel_size': 'architecture',
            'hidden_size': 'architecture', 'num_layers': 'architecture', 
            'num_heads': 'architecture', 'embedding_dim': 'architecture',
            'num_filters': 'architecture', 'hidden_dim': 'architecture',
            'n_estimators': 'architecture', 'max_depth': 'architecture',
            
            # Activation functions
            'activation': 'activation_function', 'activation_fn': 'activation_function',
            
            # Regularization params
            'dropout': 'regularization', 'l1': 'regularization', 'l2': 'regularization',
            'l1_regularization': 'regularization', 'l2_regularization': 'regularization',
            
            # Training params
            'batch_size': 'training', 'epochs': 'training', 'num_epochs': 'training',
            'steps_per_epoch': 'training', 'max_epochs': 'training',
            
            # Input processing
            'padding': 'preprocessing', 'max_length': 'preprocessing'
        }
        
        return type_mappings.get(param_name, 'other')
    
    def infer_model_type(self, code):
        """Infer model type from code with more specificity"""
        code_lower = code.lower()
        
        # More comprehensive checks for model types
        # CNN indicators
        if any(term in code_lower for term in ['conv2d', 'conv1d', 'convolution', 'cnn']):
            return 'CNN'
        
        # RNN variants
        if 'lstm' in code_lower:
            return 'LSTM'
        elif 'gru' in code_lower:
            return 'GRU'
        elif any(term in code_lower for term in ['rnn', 'recurrent']):
            return 'RNN'
        
        # Transformer models
        if any(term in code_lower for term in ['transformer', 'attention', 'multihead']):
            return 'Transformer'
        
        # Specific architectures
        architecture_indicators = {
            'resnet': 'ResNet',
            'densenet': 'DenseNet',
            'vgg': 'VGG',
            'inception': 'InceptionNet',
            'efficientnet': 'EfficientNet',
            'mobilenet': 'MobileNet',
            'yolo': 'YOLO',
            'ssd': 'SSD',
            'unet': 'UNet',
            'facenet': 'FaceNet',
            'bert': 'BERT',
            'gpt': 'GPT'
        }
        
        for indicator, model_name in architecture_indicators.items():
            if indicator in code_lower:
                return model_name
        
        # Detect feedforward networks
        if any(term in code_lower for term in ['dense', 'fully connected', 'feedforward', 'mlp']):
            return 'Dense Neural Network'
        
        # Check imports for model hints
        if 'sklearn' in self.imports:
            if any(term in code_lower for term in ['randomforest', 'random_forest']):
                return 'RandomForest'
            elif 'gradient' in code_lower and 'boost' in code_lower:
                return 'GradientBoosting'
            elif 'svm' in code_lower or 'support vector' in code_lower:
                return 'SVM'
            return 'ML Algorithm'  # Generic sklearn algorithm
            
        return 'Neural Network'  # Default
    
    def infer_task(self, code):
        """Infer ML task with more specificity"""
        code_lower = code.lower()
        
        # Classification tasks
        if any(dataset in code_lower for dataset in ['imagenet', 'cifar', 'mnist']):
            return 'image_classification'
        elif any(term in code_lower for term in ['object detection', 'yolo', 'ssd', 'rcnn']):
            return 'object_detection'
        elif any(term in code_lower for term in ['sentiment', 'text classification', 'document classification']):
            return 'text_classification'
        elif any(term in code_lower for term in ['categorical_crossentropy', 'crossentropy', 'accuracy']):
            return 'classification'
        
        # Regression tasks
        elif any(term in code_lower for term in ['mean_squared_error', 'mse', 'mae', 'regression']):
            return 'regression'
        
        # Generation tasks
        elif any(term in code_lower for term in ['generator', 'gan', 'generative']):
            return 'image_generation'
        elif any(term in code_lower for term in ['text generation', 'language model', 'gpt', 'completion']):
            return 'text_generation'
        
        # Sequence tasks
        elif any(term in code_lower for term in ['sequence', 'time series', 'forecasting', 'prediction']):
            return 'sequence_prediction'
        
        # NLP tasks
        elif any(term in code_lower for term in ['nlp', 'token', 'embedding', 'bert', 'transformer', 'attention']):
            if 'translation' in code_lower:
                return 'machine_translation'
            elif any(term in code_lower for term in ['question', 'answer']):
                return 'question_answering'
            return 'nlp_task'
        
        # Reinforcement learning
        elif any(term in code_lower for term in ['reinforcement', 'agent', 'environment', 'reward', 'action']):
            return 'reinforcement_learning'
        
        # Unsupervised learning
        elif any(term in code_lower for term in ['cluster', 'kmeans']):
            return 'clustering'
        elif any(term in code_lower for term in ['autoencoder', 'dimension reduction', 'pca']):
            return 'dimensionality_reduction'
        
        # Default case
        else:
            return 'general_ml_task'
    
    def infer_dataset_size(self, code):
        """Infer dataset size from code context"""
        code_lower = code.lower()
        
        # Dataset size indicators
        large_indicators = ['imagenet', 'large', 'big', 'million', '1000000', '10000000']
        medium_indicators = ['medium', 'moderate', '10000', '100000', 'thousand', 'cifar100']
        small_indicators = ['mnist', 'small', 'tiny', 'few', '1000', 'hundred', 'cifar10']
        
        # Check for explicit mentions
        for indicator in large_indicators:
            if indicator in code_lower:
                return 'large'
                
        for indicator in medium_indicators:
            if indicator in code_lower:
                return 'medium'
                
        for indicator in small_indicators:
            if indicator in code_lower:
                return 'small'
        
        # Check for batch size as a hint (large batch sizes often indicate larger datasets)
        batch_size_indicators = [
            r'batch_size\s*=\s*(\d+)',
            r'batch_size:\s*(\d+)',
            r'batch_size\s+(\d+)'
        ]
        
        import re
        for pattern in batch_size_indicators:
            match = re.search(pattern, code_lower)
            if match:
                try:
                    batch_size = int(match.group(1))
                    if batch_size >= 128:
                        return 'large'
                    elif batch_size >= 32:
                        return 'medium'
                    else:
                        return 'small'
                except (ValueError, IndexError):
                    pass
        
        return 'unspecified'  # Default if no clear indicators


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
        
        # Try additional framework detection if still unknown
        if extractor.framework == "unknown":
            if "tf." in content or "tensorflow" in content.lower():
                extractor.framework = "tensorflow"
            elif "torch." in content or "nn." in content.lower():
                extractor.framework = "pytorch"
            elif "sklearn" in content.lower() or "scikit-learn" in content.lower():
                extractor.framework = "sklearn"
        
        # Generate a unique ID
        framework_prefix = extractor.framework[:2] if extractor.framework != "unknown" else "ml"
        unique_id = f"{framework_prefix}_{uuid.uuid4().hex[:8]}"
        
        # Create source URL
        source_url = f"https://github.com/{repo_name}/blob/master/{file_path}"
        
        # Infer model type and task
        model_type = extractor.infer_model_type(content)
        task = extractor.infer_task(content)
        dataset_size = extractor.infer_dataset_size(content)
        
        # Create the metadata for the code sample
        metadata = {
            "id": unique_id,
            "framework": extractor.framework,
            "source_url": source_url,
            "hyperparameters": extractor.hyperparameters,
            "model_type": model_type,
            "task": task,
            "dataset_size": dataset_size,
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
def postprocess_metadata(metadata_list):
    """Post-process metadata to improve framework and parameter detection"""
    processed_list = []
    
    for metadata in metadata_list:
        # Skip empty metadata
        if not metadata or not metadata.get("hyperparameters"):
            continue
            
        # Make a copy to avoid modifying the original
        item = json.loads(json.dumps(metadata))
        
        # Fix framework detection if still unknown
        if item['framework'] == "unknown":
            code = item['code_snippet'].lower()
            if 'tensorflow' in code or 'tf.' in code or 'keras' in code:
                item['framework'] = "tensorflow"
            elif 'torch' in code or 'nn.' in code:
                item['framework'] = "pytorch"
            elif 'sklearn' in code:
                item['framework'] = "sklearn"
                
        # Improve model type detection
        if item['model_type'] == "Neural Network":
            code = item['code_snippet'].lower()
            if 'conv' in code or 'cnn' in code:
                item['model_type'] = "CNN"
            elif 'lstm' in code:
                item['model_type'] = "LSTM"
            elif 'gru' in code:
                item['model_type'] = "GRU"
            elif 'rnn' in code:
                item['model_type'] = "RNN"
            elif 'transformer' in code or 'attention' in code:
                item['model_type'] = "Transformer"
        
        # Improve task detection
        if item['task'] == "general_ml_task":
            code = item['code_snippet'].lower()
            if 'classifier' in code or 'classification' in code:
                item['task'] = "classification"
            elif 'regressor' in code or 'regression' in code:
                item['task'] = "regression"
        
        # Fix activation parameter types
        for i, param in enumerate(item['hyperparameters']):
            if param['name'] == 'activation' and param['param_type'] == 'other':
                item['hyperparameters'][i]['param_type'] = 'activation_function'
        
        processed_list.append(item)
    
    return processed_list