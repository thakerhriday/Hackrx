# Use a lightweight official Python base image (you can choose a version e.g. 3.10, 3.11)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy your requirements file
COPY requirements.txt .

# Install CPU-only torch manually
RUN pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install dependencies including awslambdaric
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt awslambdaric

# Copy your lambda function code
COPY lambda_function.py .

# Set the entry point to the AWS Lambda Runtime Interface Client to simulate Lambda environment
ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]

# Command specifies your lambda handler (module.function)
CMD ["lambda_function.lambda_handler"]
