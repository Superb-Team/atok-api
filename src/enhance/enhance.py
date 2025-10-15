from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatBedrockConverse(
    model_id="deepseek.v3-v1:0",
    region_name="ap-northeast-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    additional_model_request_fields={
        "temperature": 0.0,
        "max_tokens": 2048,
        "top_p": 0.5,
        "reasoning_effort": "high",
    },
    
)

system_prompt = SystemMessage(
    content="""
    Your job is to enhance the transcription results. 
    the transcription results may be inaccurate due audio issues or mispredict. 
    so you need to analize the transcription result and enhance it. 
    you need to find pattern from the text. 
    YOU MUST GIVE OUTPUT WITH CLEAN UNBIASED TRANSCRIPTION OUTPUT
    """
)

# Example usage (commented out - will be called from API endpoint)
# input_text = HumanMessage(
#     content="""
#     The following is a transcription of a podcast. 
#     The podcast is about a man who is trying to find a way to live a better life. 
#     He is exploring different strategies and techniques to improve his well-being and happiness.
#     """
# )
# 
# response = llm.invoke([system_prompt, input_text])
# print(response.content)