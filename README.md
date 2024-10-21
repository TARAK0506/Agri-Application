# Streamlined Data Science Deployment: FastAPI, Docker, and GitHub Actions.


This repository contains an end-to-end project that demonstrates how to deploy both machine learning (ML) and deep learning (DL) models as an **API endpoints** using **FastAPI** and **Docker**, integrated with **GitHub Actions** for CI/CD automation. The solution addresses challenges related to deploying large models on platforms with file size limitations (**Heroku**) by optimizing the model deployment process.

## Problem Statement

Farmers need a reliable, AI-driven solution to help them:

- **Predict crop diseases** based on image inputs to minimize loss and improve yields.
- **Recommend optimal crops** based on environmental and soil conditions, ensuring more efficient farming practices.
- **Provide fertilizer recommendations** for maximizing soil potential and improving crop health.
- **Real-time weather forecasting**, enabling better planning and decision-making for farming operations.


## Solution Overview

To address the problem:
1. We developed three AI models:
   - **Plant Disease Prediction Model**: A CNN-based deep learning model that classifies diseases in crops from images, helping farmers take timely action to minimize crop loss.
   - **Crop Recommendation Model**: A Decision Tree-based machine learning model that recommends the most suitable crop based on environmental factors such as soil conditions, temperature, and rainfall.
   - **Fertlizer Recommendation Model**: A machine learning model that recommends the optimal fertilizer based on soil nutrients and crop requirements, ensuring balanced nutrition for healthy crop growth.
   
2. The models were **quantized** and **pruned** to reduce their size for deployment on Heroku, overcoming the platform’s 500 MB slug size limitation.

3. We used **FastAPI** to build a scalable API to serve both models, with well-documented endpoints accessible and testing via Postman and Swagger UI.

4. The entire application, along with its dependencies, was **Dockerized** and deployed on **Heroku** using a **GitHub Actions** CI/CD pipeline for seamless integration and automated deployment.


## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Details](#model-details)
- [Deployment Strategy](#deployment-strategy)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [CI/CD with GitHub Actions](#ci-cd-with-github-actions) 
- [Optimizations](#optimizations)

## Project Overview

This project is designed to demonstrate the deployment of both machine learning and deep learning models using FastAPI as the web framework and Docker for containerization. The deployment pipeline uses GitHub Actions for Continuous Integration/Continuous Deployment (CI/CD) and automates the deployment process. The main goal is to handle the deployment of models that exceed size limits by optimizing model loading strategies.

## Features
- FastAPI-based API for serving ML and DL models.
- Dockerized application for easy environment replication and deployment.
- GitHub Actions for CI/CD: automate testing, building, and deployment.
- Model optimization to handle deployment constraints (e.g., Heroku file size limit).
- Separate endpoints for machine learning (Decision Tree) and deep learning (CNN) models.
  
## Tech Stack
- **FastAPI**: Web framework for serving the models.
- **Docker**: Containerization of the application for streamlined deployment.
- **GitHub Actions**: CI/CD pipeline for automatic build, test, and deployment.
- **Heroku**: Platform as a Service (PaaS) for deployment.
  
## Model Details
1. **Crop Disease Prediction Model (Deep Learning)**:
   - Architecture: CNN-based model saved as `Plant_Disease_Prediction_model.h5` and weights as `Keras_model.weights.weights`.
   - Usage: Predicts crop diseases from input images.
2. **Crop Recommendation Model (Machine Learning)**:
   - Algorithm: Decision Tree saved as `DecisionTree.pkl`.
   - Usage: Recommends crops based on environmental conditions.

## Deployment Strategy
The deployment utilizes FastAPI to serve both models as API endpoints. The models are loaded dynamically to handle large file sizes, ensuring a balance between performance and platform limitations.

## Key Considerations:
- **Model Size Optimization**: Models exceeding platform file limits are loaded dynamically during runtime to minimize initial deployment size.
- **CI/CD Pipeline**: Automated testing, building, and deployment are handled through GitHub Actions.


## Project Structure

```
.
├── .github
│   └── workflows
│       └── ci-cd.yml
├── app
│   ├── __init__.py
│   ├── app.py
|  
├── data
│   ├── raw
│   │   └── class_indices.json
│   ├── processed
│       └── crop_recommendation.csv
|       └── Fertlizer_recommendation.csv
|       
├── models
│    └── Fertlizer_recommendation.pkl
|    └── Keras_model.weights.weights
|    └── Crop_recommendation.pkl
|   
├── notebooks
│   └── Crop Recommendation.ipynb
|   └── Plant Disease Prediction.ipynb
|   └── Fertilizer Recommendation.ipynb
|
├── templates
│   └── index.html
|
├── static
|   └── style.css
|   └── scripts.js
|   └── favicon.ico
|
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .slugignore
├── Procfile
├── runtime.txt
├── test_images
|    └── image.jpg
|    └── image2.jpg
├── requirements.txt
├── README.md
└── .env

```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

2. **Install dependencies: Install the required Python packages using**:
   ```bash
   pip install -r requirements.txt

3. **Set up Docker: Build the Docker image**:
    ```bash
    docker build -t your_image_name .
    
## Running Locally

1. **Start the FastAPI server**:
    ```bash 
    uvicorn main:app --reload

2. **Access the API**: The API can be accessed at http://127.0.0.1:8000/docs for API documentation (Swagger UI).

3. **Postman for API Testing**: API endpoints were thoroughly tested using Postman to ensure they are functioning correctly.
   
4. **Using Docker Compose**: To run the application with Docker Compose, use:
   ```bash 
   uvicorn main:app --reload

## CI/CD with Github Actions
This repository is configured to automatically:
- Run tests on each commit.
- Build and push the Docker image.
- Deploy the app to Heroku or another platform.
  
The CI/CD pipeline is defined in .github/workflows/cicd.yml.

## How to Set Up GitHub Actions
- Ensure you have configured your Docker and Heroku API key and other credentials as GitHub secrets.
- Modify the cicd.yml file to customize the deployment steps based on your platform.

## Challenges
- This repository demonstrates an **end-to-end deployment** of machine learning and deep learning models for agriculture, including **Plant disease prediction** ,**Crop recommendation** and **Fertlizer recommendation**.
- The solution is optimized for deployment on **Heroku**, with seamless integration through **GitHub Actions** for CI/CD and containerization via **Docker**. This project also leverages model size reduction techniques like **quantization** and **pruning** to ensure deployment feasibility within Heroku’s constraints.
  
## Optimizations
- Model Loading Strategy: Models are loaded dynamically at runtime to reduce the initial file size, making it easier to deploy on Heroku (which has a 500 MB slug size limit).
- Model Size Reduction: Various techniques like model quantization can be applied to further reduce the model size if necessary.




## Future Enhancements

The current version of the application provides essential crop disease prediction and crop recommendation functionalities. However, we have exciting plans to expand and enhance the features in future iterations:

1. **Real-Time Weather Forecasting Integration**:
   - Incorporate real-time weather data from APIs (e.g., OpenWeatherMap) to provide context-aware crop recommendations. This will enable users to make more informed decisions based on current weather conditions in their region.

2. **AI-Based Custom Chatbot for Agriculture**:
   - Develop an AI-powered chatbot that can assist farmers with questions about crop diseases, fertilizer suggestions, and optimal farming practices.
   - The chatbot will integrate with the existing FastAPI application to provide real-time advice based on data and predictions from the models.
     
3. **Seamless User Interface (UI)**:
   - Develop a more intuitive and user-friendly interface to ensure a better user experience for both farmers and agricultural experts.
   - Implement responsive design to make the application more accessible across devices, including desktops, tablets, and mobile phones.
  




   
   

    
   




