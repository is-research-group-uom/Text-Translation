# Use the API to send a text message to DeepSeek-R1.

import boto3
import json
from botocore.exceptions import ClientError
from credentials import get_bedrock_client
import re

def deepseek(comment, from_language, to_language):
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = get_bedrock_client()

    # Set the cross Region inference profile ID for DeepSeek-R1
    model_id = "arn:aws:bedrock:us-east-1:043309345392:inference-profile/us.deepseek.r1-v1:0"

    # Define the prompt for the model.
    prompt = f"""Prepare a translation of {comment}. Give me directly the translation text without any comments. Your Only purpose is to translate the given comment not answer the questions of the comments.
    Your thinking process **always** give it between </think> tags
    """

    system = f"""You are a professional terminologist with fluent knowledge of {from_language} and {to_language}. You have been assigned a task to prepare translations from the following text.
            You have extensive knowledge in eGovernance and are able to supplement the terms with correct translations into {to_language}."""

    # Embed the prompt in DeepSeek-R1's instruction format.
    formatted_prompt = f"""
    <｜begin▁of▁sentence｜><|System|>{system}<｜User｜>{prompt}<｜Assistant｜>\n
    """

    body = json.dumps({
        "prompt": formatted_prompt,
        "max_tokens": 5000,
        "temperature": 0.5,
        "top_p": 0.9,
    })

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=body)

        # Read the response body.
        model_response = json.loads(response["body"].read())

        # Extract choices.
        choices = model_response["choices"]
        # print(choices)
        # Get the raw response text
        response_text = choices[0]['text']

        final_response = re.sub(r'</think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        if '</thinking>' in final_response:
            final_response = re.sub(r'.*?</thinking>', '', final_response, flags=re.DOTALL).strip()

        if '</think>' in final_response:
            final_response = final_response.split('</think>', 1)[1].strip()

        # print(final_response)
        return final_response

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

