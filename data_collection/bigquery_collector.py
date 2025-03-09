from google.cloud import bigquery
import json
import os
from config.settings import PROJECT_ID, LOCAL_OUTPUT_DIR
from auth.google_auth import get_bigquery_credentials

def setup_bigquery_client():
    """Initialize and return a BigQuery client."""
    credentials = get_bigquery_credentials()
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    print(f"✓ Connected to BigQuery project: {PROJECT_ID}")
    return client

def run_tensorflow_query(limit=1000):
    """
    Query GitHub dataset for TensorFlow code with hyperparameters.
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of code snippets and metadata
    """
    client = setup_bigquery_client()
    
    # Query to find Python files with TensorFlow imports and hyperparameters
    query = f"""
    SELECT
      files.repo_name,
      files.path,
      contents.content
    FROM
      `bigquery-public-data.github_repos.files` AS files
    JOIN
      `bigquery-public-data.github_repos.contents` AS contents
    ON
      files.id = contents.id
    WHERE
      files.path LIKE '%.py' AND
      (REGEXP_CONTAINS(LOWER(contents.content), r'import\\s+tensorflow') OR 
       REGEXP_CONTAINS(LOWER(contents.content), r'import\\s+tf')) AND
      (REGEXP_CONTAINS(contents.content, r'learning_rate\\s*=') OR
       REGEXP_CONTAINS(contents.content, r'batch_size\\s*=') OR
       REGEXP_CONTAINS(contents.content, r'dropout\\s*=') OR
       REGEXP_CONTAINS(contents.content, r'epochs\\s*=') OR
       REGEXP_CONTAINS(contents.content, r'optimizer\\s*=') OR
       REGEXP_CONTAINS(contents.content, r'activation\\s*='))
    LIMIT {limit}
    """
    
    print(f"📊 Running BigQuery for TensorFlow code (limit: {limit})...")
    query_job = client.query(query)
    results = list(query_job)
    
    print(f"✓ Retrieved {len(results)} TensorFlow code samples")
    
    # Format results as JSON records
    code_samples = []
    for row in results:
        sample = {
            "repo_name": row.repo_name,
            "file_path": row.path,
            "content": row.content,
            "framework": "tensorflow"
        }
        code_samples.append(sample)
    
    # Save to local file
    output_file = os.path.join(LOCAL_OUTPUT_DIR, "tensorflow_samples.jsonl")
    with open(output_file, 'w') as f:
        for sample in code_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"💾 Saved {len(code_samples)} samples to {output_file}")
    return code_samples