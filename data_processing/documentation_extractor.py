import inspect
import json
import os
import re

class DocumentationExtractor:
    """Extract documentation from ML frameworks using Python's introspection"""
    
    def __init__(self, cache_dir="./data/documentation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.doc_db = {}
        self._load_or_extract_all()
    
    def _load_or_extract_all(self):
        """Load existing documentation database or extract if it doesn't exist"""
        cache_file = os.path.join(self.cache_dir, "documentation_db.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.doc_db = json.load(f)
                print(f"✓ Loaded documentation database from {cache_file}")
        else:
            print("Documentation database not found. Extracting...")
            self.extract_and_save_all()
    
    def extract_and_save_all(self):
        """Extract docs from all supported frameworks and save to disk"""
        # Check which frameworks are available
        frameworks = []
        
        try:
            import torch
            frameworks.append("pytorch")
        except ImportError:
            print("PyTorch not found. Skipping PyTorch documentation extraction.")
        
        try:
            import tensorflow as tf
            frameworks.append("tensorflow")
        except ImportError:
            print("TensorFlow not found. Skipping TensorFlow documentation extraction.")
        
        try:
            import sklearn
            frameworks.append("sklearn")
        except ImportError:
            print("scikit-learn not found. Skipping scikit-learn documentation extraction.")
        
        # Initialize empty database for each framework
        for framework in frameworks:
            self.doc_db[framework] = {}
        
        # Extract documentation for each available framework
        if "pytorch" in frameworks:
            self.extract_torch_docs()
        
        if "tensorflow" in frameworks:
            self.extract_tf_docs()
        
        if "sklearn" in frameworks:
            self.extract_sklearn_docs()
        
        # Save to disk
        cache_file = os.path.join(self.cache_dir, "documentation_db.json")
        with open(cache_file, 'w') as f:
            json.dump(self.doc_db, f, indent=2)
        
        print(f"✓ Saved documentation database to {cache_file}")
        return self.doc_db
    
    def extract_torch_docs(self):
        """Extract documentation from PyTorch"""
        print("Extracting PyTorch documentation...")
        import torch
        
        # Common PyTorch modules to document
        module_classes = [
            (torch.nn, ["Conv1d", "Conv2d", "Conv3d", "Linear", "LSTM", "GRU", "RNN",
                        "Dropout", "BatchNorm1d", "BatchNorm2d"]),
            (torch.optim, ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"])
        ]
        
        for module, classes in module_classes:
            for class_name in classes:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    self.doc_db["pytorch"][class_name] = {
                        "description": inspect.getdoc(cls),
                        "parameters": self._get_params_info(cls)
                    }
                    print(f"  - Extracted docs for {class_name}")
        
        # Extract common hyperparameters
        self._extract_common_pytorch_hyperparams()
        
        return self.doc_db["pytorch"]
    
    def _extract_common_pytorch_hyperparams(self):
        """Extract common hyperparameters from PyTorch"""
        # Common hyperparameters with their documentation
        hyperparam_info = {
            "lr": {
                "description": "Learning rate. Controls how much to adjust model weights during training.",
                "common_values": ["0.1", "0.01", "0.001", "0.0001"],
                "typical_range": "0.0001 to 0.1"
            },
            "weight_decay": {
                "description": "Weight decay (L2 penalty). Adds regularization by penalizing large weights.",
                "common_values": ["0.0", "0.0001", "0.001", "0.01"],
                "typical_range": "0.0 to 0.01" 
            },
            "batch_size": {
                "description": "Number of samples processed in each training batch.",
                "common_values": ["16", "32", "64", "128", "256"],
                "typical_range": "8 to 512"
            }
        }
        
        self.doc_db["pytorch"]["hyperparameters"] = hyperparam_info
    
    def extract_tf_docs(self):
        """Extract documentation from TensorFlow/Keras"""
        print("Extracting TensorFlow documentation...")
        import tensorflow as tf
        from tensorflow import keras
        
        # Common TensorFlow/Keras modules to document
        module_classes = [
            (keras.layers, ["Conv1D", "Conv2D", "Dense", "LSTM", "GRU",
                           "Dropout", "BatchNormalization"]),
            (keras.optimizers, ["SGD", "Adam", "RMSprop"])
        ]
        
        for module, classes in module_classes:
            for class_name in classes:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    self.doc_db["tensorflow"][class_name] = {
                        "description": inspect.getdoc(cls),
                        "parameters": self._get_params_info(cls)
                    }
                    print(f"  - Extracted docs for {class_name}")
        
        # Extract common hyperparameters
        self._extract_common_tf_hyperparams()
        
        return self.doc_db["tensorflow"]
    
    def _extract_common_tf_hyperparams(self):
        """Extract common hyperparameters from TensorFlow"""
        hyperparam_info = {
            "learning_rate": {
                "description": "Learning rate. Controls how rapidly the model weights are adjusted.",
                "common_values": ["0.1", "0.01", "0.001", "0.0001"],
                "typical_range": "0.0001 to 0.1"
            },
            "batch_size": {
                "description": "Number of samples processed before the model is updated.",
                "common_values": ["16", "32", "64", "128", "256"],
                "typical_range": "8 to 512"
            },
            "epochs": {
                "description": "Number of complete passes through the training dataset.",
                "common_values": ["10", "50", "100", "200"],
                "typical_range": "10 to 500"
            }
        }
        
        self.doc_db["tensorflow"]["hyperparameters"] = hyperparam_info
    
    def extract_sklearn_docs(self):
        """Extract documentation from scikit-learn"""
        print("Extracting scikit-learn documentation...")
        import sklearn
        from sklearn import linear_model, ensemble, svm
        
        module_classes = [
            (linear_model, ["LinearRegression", "LogisticRegression"]),
            (ensemble, ["RandomForestClassifier", "GradientBoostingClassifier"]),
            (svm, ["SVC"])
        ]
        
        for module, classes in module_classes:
            for class_name in classes:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    self.doc_db["sklearn"][class_name] = {
                        "description": inspect.getdoc(cls),
                        "parameters": self._get_params_info(cls)
                    }
                    print(f"  - Extracted docs for {class_name}")
        
        return self.doc_db["sklearn"]
    
    def _get_params_info(self, cls):
        """Extract parameter information from a class"""
        params = {}
        
        # Get signature information
        try:
            sig = inspect.signature(cls)
            
            for name, param in sig.parameters.items():
                # Skip self parameter
                if name == 'self':
                    continue
                    
                params[name] = {
                    "default": str(param.default) if param.default is not inspect.Parameter.empty else None,
                    "type": str(param.annotation) if param.annotation is not inspect.Parameter.empty else "unspecified",
                }
                
                # Extract parameter description from docstring
                docstring = inspect.getdoc(cls)
                if docstring:
                    # Try different docstring formats
                    param_patterns = [
                        rf"{re.escape(name)}\s*[:]\s*.*?(?:\n\s+.*?)+",  # Format: param_name : type
                        rf"{re.escape(name)}\s*--\s*.*?(?:\n\s+.*?)+",   # Format: param_name -- description
                        rf"Parameters.*?{re.escape(name)}.*?:(.*?)(?:\n\s*\S+:|$)"  # Format: Parameters section
                    ]
                    
                    for pattern in param_patterns:
                        param_match = re.search(pattern, docstring, re.DOTALL)
                        if param_match:
                            params[name]["description"] = param_match.group(0).strip()
                            break
                            
        except Exception as e:
            print(f"  ⚠️ Could not inspect {cls.__name__}: {e}")
        
        return params
    
    def get_param_info(self, framework, class_name, param_name):
        """Get documentation for a specific parameter"""
        if framework not in self.doc_db:
            return None
        
        # First try class-specific parameters
        if class_name in self.doc_db[framework]:
            params = self.doc_db[framework][class_name].get("parameters", {})
            if param_name in params:
                return params[param_name]
        
        # Then try common hyperparameters
        if "hyperparameters" in self.doc_db[framework] and param_name in self.doc_db[framework]["hyperparameters"]:
            return self.doc_db[framework]["hyperparameters"][param_name]
        
        # Check for synonyms
        synonyms = {
            "learning_rate": ["lr"],
            "lr": ["learning_rate"]
        }
        
        if param_name in synonyms:
            for synonym in synonyms[param_name]:
                # Check if synonym exists in class parameters
                if class_name in self.doc_db[framework]:
                    params = self.doc_db[framework][class_name].get("parameters", {})
                    if synonym in params:
                        return params[synonym]
                
                # Check if synonym exists in common hyperparameters
                if "hyperparameters" in self.doc_db[framework] and synonym in self.doc_db[framework]["hyperparameters"]:
                    return self.doc_db[framework]["hyperparameters"][synonym]
        
        return None