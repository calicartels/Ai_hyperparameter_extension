import os

# Google Cloud settings
PROJECT_ID = "capstone-449418"
LOCATION = "us-central1" 
SERVICE_ACCOUNT_KEY = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   "credentials", "695116221974-hqg9ap5bh4ok0nc23bbbkh8h30nhqa8d.apps.googleusercontent.com")
CREDENTIALS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials")

# Storage settings
OUTPUT_BUCKET = "hyperparameter_deep_learning"  
LOCAL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Ensure directories exist
os.makedirs(CREDENTIALS_DIR, exist_ok=True)
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)