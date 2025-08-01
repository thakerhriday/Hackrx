import json
import os

# Import your lambda handler function
from lambda_function import lambda_handler

# Load your test event data
with open("test_event.json", "r") as f:
    event = json.load(f)

# Run the lambda handler
result = lambda_handler(event, None)

# Pretty-print the result
print(json.dumps(result, indent=2))
