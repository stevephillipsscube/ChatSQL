# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some python packages like chromadb/duckdb sometimes)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose port 8501 for Streamlit
EXPOSE 8501

# Define environment variable for running in non-interactive mode
ENV HEADLESS=true

# Run streamlit when the container launches
CMD ["streamlit", "run", "underwriting_ui_UseThis.py", "--server.port=8501", "--server.address=0.0.0.0"]
