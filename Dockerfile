# Use an official Python 3.12 runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install python-multipart (if not already in requirements.txt)
RUN pip install python-multipart

# Copy the rest of the project files into the container
COPY . .

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
