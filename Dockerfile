# Base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . .

# Expose the port
EXPOSE 8082

# Start the app
CMD [ "python", "app.py" ]
