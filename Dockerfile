# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application code into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the trained model files into the container
COPY app/Crop_Disease_Prediction_model.h5 /app/
COPY app/DecisionTree.pkl /app/
COPY app/class_indices.json /app/




# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]