import boto3
from strands.models import BedrockModel
from dotenv import load_dotenv

import os
load_dotenv()

# Create a custom boto3 session
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-northeast-3",
)

# Create a Bedrock model with the custom session
llm = BedrockModel(
    model_id="jp.anthropic.claude-haiku-4-5-20251001-v1:0",
    boto_session=session
)
