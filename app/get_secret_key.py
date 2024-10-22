import boto3
import boto3.session
import json

def get_secret(key : str):
    secret_name = "askkenai.io"
    region_name = "us-east-1"

    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )

    secret = get_secret_value_response['SecretString']
    # Convert JSON string to dict
    secret_dict = json.loads(secret)

    return secret_dict[key]