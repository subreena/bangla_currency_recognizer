# Use official Python image with supported version
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all source code
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
