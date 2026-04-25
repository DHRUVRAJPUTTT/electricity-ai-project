# Use Python slim image
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file from the backend folder to the current directory
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else (frontend and backend folders)
COPY . .

# Expose the port Hugging Face uses
EXPOSE 7860

# Move into the backend folder to run the script
WORKDIR /app/backend

# Run the application
CMD ["python", "main.py"]
