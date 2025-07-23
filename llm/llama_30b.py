import boto3
import json

from botocore.exceptions import ClientError
from credentials import get_bedrock_client


def llama(comment, from_language, to_language):
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = get_bedrock_client()

    # Set the model ID, e.g., Llama 3 70b Instruct.
    model_id = "arn:aws:bedrock:us-east-1:043309345392:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"

    # Define the prompt for the model.
    prompt = f"""
    Prepare a translation of {comment}. Give me directly the translation text without any comments.
    """

    system = f"""You are a professional terminologist with fluent knowledge of {from_language} and {to_language}. You have been assigned a task to prepare translations from the following text.
            You have extensive knowledge in eGovernance and are able to supplement the terms with correct translations into {to_language}."""

    # Embed the prompt in Llama 3's instruction format.
    formatted_prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system}
    <|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    # Format the request payload using the model's native structure.
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 4096,
        "temperature": 0.5,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["generation"]

    return response_text

