import os
from google.oauth2.service_account import Credentials
from config.settings import PROJECT_ID, SERVICE_ACCOUNT_KEY

def get_bigquery_credentials():
    """
    Get credentials for Google BigQuery using service account.
    
    Returns:
        Credentials: The service account credentials
    """
    if os.path.exists(SERVICE_ACCOUNT_KEY):
        credentials = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_KEY,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        print(f"✓ Loaded BigQuery credentials from {SERVICE_ACCOUNT_KEY}")
        return credentials
    else:
        print("⚠️ Service account key not found, using application default credentials")
        return None