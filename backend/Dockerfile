# 1. Use the exact Python version requested (slim variant for smaller size)
FROM python:3.11.9-slim

# 2. Prevent Python from writing .pyc files and keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy ONLY the requirements file first to leverage Docker cache
COPY requirements.txt .

# 5. Install the dependencies (keeping the image size down)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your backend files (main.py, models, scalers, etc.)
COPY . .

# 7. Expose the port (adjust if your app uses a different port like 5000)
EXPOSE 8000

# 8. The command to start your app
CMD ["python", "main.py"]
