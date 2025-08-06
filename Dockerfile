# Use an official lightweight Python image as a base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application's source code into the container
COPY ./src /app/src

# For this assignment, we copy the local mlruns directory into the container
# so the API can load the model. In a real production setup, the container
# would connect to a remote MLflow tracking server instead.
COPY ./mlruns /app/mlruns

# Make port 5000 available to the outside world
EXPOSE 5001

# The command to run when the container starts
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]