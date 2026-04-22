import os
import json


def load_secrets():
    """Load secrets from AWS Secrets Manager if AWS_SECRETS_NAME is set,
    otherwise rely on .env loaded by the caller."""
    secret_name = os.environ.get("AWS_SECRETS_NAME")
    if not secret_name:
        return

    import boto3
    client = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "eu-central-1"))
    response = client.get_secret_value(SecretId=secret_name)
    secrets = json.loads(response["SecretString"])

    for key, value in secrets.items():
        os.environ[key] = value
