import json
from botocore.exceptions import ClientError
from credentials import get_bedrock_client


def claude3_5(comment, from_language, to_language):
    # Create an Amazon Bedrock Runtime client.
    brt = get_bedrock_client()

    # Set the model ID
    model_id = "arn:aws:bedrock:us-east-1:043309345392:inference-profile/us.anthropic.claude.txt-3-5-sonnet-20240620-v1:0"

    # Define the prompt for the model
    prompt = f"""
    Prepare a translation of {comment}. Give me directly the translation text without any comments.
    """

    # Format the request payload (back to simple text)
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 131072,
        "top_k": 250,
        "stop_sequences": [],
        "temperature": 0.7,
        "top_p": 0.999,
        "system": f"""You are a professional terminologist with fluent knowledge of {from_language} and {to_language}. You have been assigned a task to prepare translations from the following text. 
                    You have extensive knowledge in eGovernance and are able to supplement the terms with correct translations into {to_language}.""",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    # Convert the native request to JSON
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request
        response = brt.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body
    model_response = json.loads(response["body"].read())

    # Extract and print the response text
    response_text = model_response['content'][0]['text']

    return response_text