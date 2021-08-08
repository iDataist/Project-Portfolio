## Table of Content
- [Full Stack Web Development](#1)
- [Cloud Native Application Architecture](#2)
- [Cloud Development with Microsoft Azure](#3)
- [Cloud DevOps with Microsoft Azure](#4)
- [Machine Learning Engineering with Microsoft Azure](#5)
- [Cloud Development with AWS](#6)
- [Cloud DevOps with AWS](#7)
- [Machine Learning Engineering with AWS Sagemaker](#8)
- [Machine Learning DevOps](#16)
- [Data Architecture](#9)
- [Data Engineering](#10)
- [Data Streaming](#17)
- [Data Science](#11)
- [Deep Learning](#12)
- [Deep Reinforcement Learning](#13)
- [Natural Language Processing](#14)
- [AI for Healthcare](#15)

<a id='1'></a>
## Full Stack Web Development

### [Data Modeling for Music Event App](https://github.com/iDataist/Data-Modeling-for-Music-Event-App)
Built the data models to power the API endpoints for the music event app Fyyur.

Key Skills Demonstrated:
- Building the data models upon which the website relies
- Architecting and populating the Postgresql database

### [API Development and Documentation for the Trivia App](https://github.com/iDataist/API-Development-and-Documentation-for-the-Trivia-App)
Implemented, tested and documented the APIs for the Trivia app.

Key Skills Demonstrated:
- Using APIs to control and manage a web application with existing data models
- Implementing tests and documentations for APIs

### [Identity and Access Management for the Coffee Shop App](https://github.com/iDataist/Identity-and-Access-Management-for-the-Coffee-Shop-App)
Built the backend for a coffee shop application and enabled role-based access.

Key Skills Demonstrated:
- Implementing authentication and authorization in Flask
- Implementing role-based control design patterns
- Applying software system risk and compliance principles

### [Deploying a Flask App to Amazon Elastic Kubernetes Service with a CI/CD pipeline](https://github.com/iDataist/Deploying-a-Flask-App-to-Amazon-Elastic-Kubernetes-Service-with-a-CI-CD-pipeline)
Created a container for a Flask web app using Docker and deployed the container to Amazon Elastic Kubernetes Service with AWS CloudFormation, AWS CodePipeline and AWS CodeBuild.

Key Skills Demonstrated:
- Defining a container environment using a Dockerfile
- Deploying a Docker container to a Kubernetes cluster using Amazon EKS and the AWS command line interface (CLI)
- Implementing Continuous Integration (CI) and Continuous Delivery (CD) with AWS CloudFormation, AWS CodePipeline and AWS CodeBuild

<a id='2'></a>
## Cloud Native Application Architecture

### [Deploying the TechTrends App to Kubernetes with a CI/CD Pipeline](https://github.com/iDataist/Deploying-the-TechTrends-App-to-Kubernetes-with-a-CI-CD-Pipeline)
Packaged and deployed a news-sharing application to Kubernetes with GitHub Actions and ArgoCD.

Key Skills Demonstrated:
- Packaging an application with Docker and distributing it through DockerHub
- Deploying a docker image to a cluster using Kubernetes resources
- Using template configuration managers, such as Helm, to implement the parameterization of Kubernetes declarative manifests
- Automating the release process for an application by implementing a CI/CD pipeline with GitHub Actions and ArgoCD

<a id='3'></a>
## Cloud Development with Microsoft Azure
### [Deploying a Content Management System to Azure](https://github.com/iDataist/Deploying-a-Content-Management-System-to-Azure)
Deployed an article content management system (CMS) application to Microsoft Azure. The CMS system lets a user log in, view published articles, and post new articles.

Key Skills Demonstrated:
- Deploying storage solutions for the application to interact with, including a SQL database that contains a user table and an article table for the web app to query, and a Blob Storage container where images are stored
- Configuring “Sign in with Microsoft” for authentication using OAuth 2.0 and Azure Active Directory, in addition to a simple username/ password login
- Adding logging to the cloud application to track successful or unsuccessful login attempts

### [Deploying the Neighborly App with Azure Functions](https://github.com/iDataist/Deploying-the-Neighborly-App-with-Azure-Functions)
Implemented a serverless microservice backend architecture for a social networking web application called Neighborly, a service for neighbors to exchange helpful information, goods, and services.

Key Skills Demonstrated:
- Building the backend services that leverage an API to communicate with a MongoDB database
- Integrating the client-side application and server-side API endpoints
- Deploying and managing the service with Azure Kubernetes Service for CI/CD integration

### [Migrating Tech Conference App to Azure](https://github.com/iDataist/Migrating-Tech-Conference-App-to-Azure)
Migrated a pre-existing conference registration system to Azure and architected a resilient and scalable system in Azure.

Key Skills Demonstrated:
- Migrating and deploying pre-existing web apps to an Azure App Service
- Migrating PostgreSQL database backups to Azure Postgres database instances
- Refactoring the notification logic to Azure Function via service bus queue messages

### [Enhancing Applications](https://github.com/iDataist/Enhancing-Applications)
Collected and displayed performance and health data about an application post-migration to Azure.

Key Skills Demonstrated:
- Setting up Application Insights monitoring on virtual machine scale set (VMSS)
- Implementing monitoring in applications to collect telemetry data
- Creating auto-scaling for VMSS and using RunBook to automate the resolution of performance issues
- Creating alerts to trigger auto-scaling on an Azure Kubernetes Service (AKS) cluster

<a id='4'></a>

## Cloud DevOps with Microsoft Azure

### [Deploying a Scalable IaaS Web Server in Azure](https://github.com/iDataist/Deploying-a-Scalable-IaaS-Web-Server-in-Azure)
Wrote infrastructure as code using Packer and Terraform to deploy a customizable, scalable web server in Azure.

Key Skills Demonstrated:
- Creating tagging policies
- Creating and deploying customized web server images with Packer
- Writing the infrastructure configuration with Terraform and creating load-balanced web servers

### [Deploying a Flask Machine Learning App to Azure App Services with a CI/CD Pipeline](https://github.com/iDataist/Deploying-a-Flask-Machine-Learning-App-to-Azure-App-Services-with-a-CI-CD-Pipeline)
Built a CI/CD pipeline that deploys a Flask Machine Learning application to Azure App Services with GitHub Actions and Azure DevOps.

Key Skills Demonstrated:
- Leveraging GitHub actions to implement continuous integration that includes install, lint, and test steps
- Using the Azure pipeline for continuous delivery that deploys a Flask Machine Learning application
- Testing the prediction capability of the machine learning application by giving it a JSON payload

### [Ensuring Quality Releases](https://github.com/iDataist/Ensuring-Quality-Releases)
Created a disposable test environment and ran a variety of automated tests.

Key Skills Demonstrated:
- Building a CI/CD pipeline, which creates a test environment using Terraform, deploys an application to the test environment, and executes load, integration, and functional tests
- Ingesting logs and data from tests into Azure Log Analytics to determine where failures may have occurred

<a id='5'></a>

## Machine Learning Engineering with Microsoft Azure

### [Operationalizing Machine Learning](https://github.com/iDataist/Operationalizing-Machine-Learning)
Leveraged Azure AutoML to build classifiers that predict whether the client will subscribe to a term deposit with the bank, and deployed the best model to an Azure Container Instance (ACI).

Key Skills Demonstrated:
- Shipping machine learning models into production in a reliable, reproducible, and automated way
- Enhancing observability by enabling Application Insights and logging

<a id='6'></a>
## Cloud Development with AWS

<a id='7'></a>
## Cloud DevOps with AWS

<a id='8'></a>

## Machine Learning Engineering with AWS Sagemaker

### [Deploy a Sentiment Analysis Model with SageMaker](https://github.com/iDataist/Deploy-a-Sentiment-Analysis-Model-with-SageMaker)
Built a simple web page which a user can use to enter a movie review. The web page will then send the review off to the deployed recurrent neural network which will predict the sentiment of the entered review.

Key Skills Demonstrated:
- Text analysis
- Model deployment via SageMaker
- APIs for web deployment

### [Create a Plagiarism Detector with SageMaker](https://github.com/iDataist/Create-a-Plagiarism-Detector-with-SageMaker)
Built a plagiarism detector that examines a text file and performs binary classification, labeling that file as either plagiarized or not depending on how similar that text file is to a provided source text.

Key Skills Demonstrated:
- Feature engineering
- Model design and evaluation
- Model deployment via SageMaker

<a id='16'></a>

## Machine Learning DevOps

### [Applying Clean Code Principles](https://github.com/iDataist/Applying-Clean-Code-Principles)

I applied the clean code principles (modular, documented, and tested) to implement a classifier that predicts customer churn.

Key Skills Demonstrated:
- Write production-ready code

### [Build an ML Pipeline for Short-Term Rental Prices in NYC](https://github.com/iDataist/Building-a-Reproducible-Model-Workflow)

A property management company rents rooms and properties for short periods on various rental platforms. The company needs to estimate the typical price for a given property based on the price of similar properties. New data arrives in bulk every week. I built a pipeline covering data fetching, validation, segregation, train and validation, test, and release. The end-to-end pipeline enables the model retraining with the same cadence.

Key Skills Demonstrated:
- Create a clean, organized, reproducible, end-to-end machine learning pipeline using MLflow
- Validate the data using pytest
- Track experiments, code, and results using GitHub and Weights & Biases
- Select the best-performing model for production
- Deploy a model using MLflow

### [Deploying a Scalable ML Pipeline in Production](https://github.com/iDataist/Deploying-a-Scalable-ML-Pipeline-in-Production)
I developed a CI/CD pipeline to predict salary range based on publicly available Census Bureau data. I created tests to monitor the machine learning pipeline. Then, I deployed the model using the FastAPI package and create API tests. The tests were incorporated into the CI/CD framework using GitHub Actions.

Key Skills Demonstrated:
- Version control the data and models using Data Version Control (DVC)
- Implement a CI/CD pipeline using GitHub Actions and Heroku
- Develop a type-checked and auto-documented API using FastAPI

### [ML Model Scoring and Monitoring](https://github.com/iDataist/ML-Model-Scoring-and-Monitoring)
I built an end-to-end, automated ML pipeline that predicts customer attrition risks. First, I set up processes to ingest data and score, retrain and re-deploy ML models that predict attrition risk. Second, I implemented automatically check for new data and model drift. Lastly, I set up APIs that allow users to access model results, metrics, and diagnostics.

Key Skills Demonstrated:
- Set up scoring processes
- Assess model drift and determine whether models need to be re-trained and re-deployed
- Diagnose operational issues with models, including data integrity and stability problems, timing problems, and dependency issues
- Set up automated reporting with APIs
- Automate the DevOps processes required to score and re-deploy ML models

<a id='9'></a>

## Data Architecture

### [Human Resources Database Design](https://github.com/iDataist/Human-Resources-Database-Design)
Designed, built, and populated a relational database for the Human Resources (HR) Department.

Key Skills Demonstrated:
- Build conceptual, logical and physical entity relationship diagrams (ERDs)
- Architect a physical database in PostGreSQL

### [Data Warehouse Design](https://github.com/iDataist/Data-Warehouse-Design)
Architected, designed and staged a data warehouse to assess if weather has an any effect on customer reviews of restaurants.

Key Skills Demonstrated:
- Transform data from transactional systems into an operational data store
- Create a data warehouse system using dimensional data models
- Architect a physical data warehouse with Snowflake

### [Data Lake Design](https://github.com/iDataist/Data-Lake-Design)
Developed a cloud data lake solution to replace the on-premise legacy system for a medical data processing company.

Key Skills Demonstrated:
- Use appropriate storage and processing frameworks to manage big data
- Design end-to-end batch and stream processing architecture in AWS

### [Data Governance](https://github.com/iDataist/Data-Governance)
Created foundational data management tools and artifacts, including documenting data systems, setting up a data catalog, designing better data quality and master data management processes, and formalizing data governance roles.

Key Skills Demonstrated:
- Establish data governance best practices including metadata management, master data management and data quality management

<a id='10'></a>
## Data Engineering

### [Build Data Pipelines](https://github.com/iDataist/Build-ETL-Pipelines)
Created a database and ETL pipeline in Postgres and Apache Cassandra, which enabled analysis of user activity data residing in a directory of JSON logs and metadata, then moved the data warehouse to the cloud, and eventually created and automated a set of data pipelines with Apache Airflow.

Key Skills Demonstrated:
- Data Modeling with Postgres and Apache Cassandra
- Build the Data Warehouse with S3 and Redshift
- Build the Data Lake with Spark
- Automate the Data Pipeline with Apache Airflow

<a id='17'></a>
## Data Streaming
### [Data Ingestion with Kafka and Kafka Streaming](https://github.com/iDataist/Data-Ingestion-with-Kafka-and-Kafka-Streaming)
I streamed public transit status from [Chicago Transit Authority](https://www.transitchicago.com/data/) using Kafka and the Kafka ecosystem and built a stream processing application that shows the status of trains in real-time. I used Confluent Kafka to produce events, REST Proxy to send events over HTTP, and Kafka Connect to collect data from a Postgres database, all of which are sources into Kafka. Then, I used KSQL to combine related data models into a single topic ready for consumption, and built a simple Python application that ingests data from the Kafka topics for analysis. Finally, I leveraged the Faust Python Stream Processing library to further transform train station data into a more streamlined representation. Using stateful processing, I showed whether passenger volume is increasing, decreasing, or staying steady.

Key Skills Demonstrated:
- Produce and consume topics with Confluent Kafka
- Collect data from databases with Kafka Connect
- Producing data over REST with Kafka REST Proxy
- Stream Processing with Faust Python Library and KSQL

### [Streaming API Development and Documentation](https://github.com/iDataist/Streaming-API-Development-and-Documentation)
The Step Trending Electronic Data Interface (STEDI) application assesses fall risk for seniors. When a senior takes a test, they are scored using an index that reflects the likelihood of falling and potentially sustaining an injury in the course of walking. STEDI uses a Redis datastore for risk scores and other data. Using Spark, I aggregated Kafka Connect Redis Source events and Business Events to create a Kafka topic containing anonymized risk scores of seniors in the clinic and generated a population risk graph.

Key Skills Demonstrated:
- Consuming and processing data from Apache Kafka with Spark Structured Streaming
- Creating a DataFrame as an aggregation of source DataFrames
- Sinking a composite DataFrame to Kafka

<a id='11'></a>
## Data Science

### [Find Donors for Charity with Supervised Learning Algorithms](https://github.com/iDataist/Find-Donors-for-Charity)
Evaluated and optimized several different supervised learning algorithms to determine which algorithm will provide the highest donation yield while under some marketing constraints.

Key Skills Demonstrated:
- Supervised learning
- Model evaluation and comparison
- Tuning models according to constraints

### [Create Customer Segments with Unsupervised Learning Techniques](https://github.com/iDataist/Create-Customer-Segments)
Applied unsupervised learning techniques to organize the general population into clusters, then used those clusters to determine which of them were most likely to be purchasers for a mailout campaign.

Key Skills Demonstrated:
- Data cleaning
- Dimensionality reduction with PCA
- Unsupervised clustering

### [Create an Image Classifier Using a Deep Neural Network](https://github.com/iDataist/Create-an-Image-Classifier)
Created an image classification application, which trains a deep learning model on a dataset of images and then uses the trained model to classify new images.

Key Skills Demonstrated:
- PyTorch and neural networks
- Model validation and evaluation

### [Build a Machine Learning Pipeline to Categorize Emergency Text Messages](https://github.com/iDataist/Build-Pipelines-to-Classify-Messages)
Built a data pipeline to prepare the message data from major natural disasters around the world and a machine learning pipeline to categorize emergency text messages based on the need communicated by the sender, and deployed the pipelines to create a website app that classify messages.

Key Skills Demonstrated:
- ETL Pipeline
- Machine Learning Pipeline
- Flask Web App

### [Design a Recommendation Engine](https://github.com/iDataist/Design-a-Recommendation-Engine)
Analyzed the interactions that users had with articles on the IBM Watson Studio platform, and designed a recommendation engine on new articles that users might like.

Key Skills Demonstrated:
- Exploratory Data Analysis
- Rank Based Recommendations
- User-User Based Collaborative Filtering
- Matrix Factorization

<a id='12'></a>
## Deep Learning

### [Build a Dog Identification App with Convolutional Neural Networks](https://github.com/iDataist/Build-a-Dog-Identification-App)
Developed an algorithm that could be used as part of a mobile or web app, which accepts any user-supplied image as input and provides an estimate of the dog's breed.

Key Skills Demonstrated:
- Process the image data
- Build and train a convolutional neural network
- Use Transfer Learning

### [Generate Seinfeld Scripts with Recurrent Neural Networks](https://github.com/iDataist/Generate-TV-Scripts-with-Recurrent-Neural-Networks)
Built a LSTM Recurrent Neural Network (RNN) with PyTorch, which can generate a new, "fake" TV script, based on patterns it recognizes in the Seinfeld dataset of scripts from 9 seasons.

Key Skills Demonstrated:
- Process the text data
- Build and train a Recurrent Neural Network with PyTorch
- Optimize hyperparameters

### [Generate Faces with Generative Adversarial Networks](https://github.com/iDataist/Generate-Faces-with-Generative-Adversarial-Networks)
Built a Deep Convolutional Generative Adversarial Network (DCGAN), which is made of a pair of multi-layer neural networks that compete against each other until one learns to generate realistic images of faces.

Key Skills Demonstrated:
- Process the image data
- Build and train a Deep Convolutional Generative Adversarial Network with PyTorch
- Optimize hyperparameters

<a id='13'></a>
## Deep Reinforcement Learning

### [Navigation with Deep Q-Network](https://github.com/iDataist/Navigation-with-Deep-Q-Network)
Leveraged Deep Q-Networks to train an agent that learns to navigate and collect bananas in a large, square world.

Key Skills Demonstrated:
- Implement a Value-based Deep Reinforcement Learning Algorithm
- Build and train neural networks with PyTorch

### [Continuous Control with Deep Deterministic Policy Gradient](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient)
Leveraged neural networks to train twenty double-jointed arms to move to target locations.

Key Skills Demonstrated:
- Implement a Policy-based Deep Reinforcement Learning Algorithm
- Build and train neural networks with PyTorch

### [Tennis With Multi Agent Reinforcement](https://github.com/iDataist/Tennis-With-Multi-Agent-Reinforcement)
Applied reinforcement learning methods to trained two agents to play tennis.

Key Skills Demonstrated:
- Implement a Multi-Agent Reinforcement Learning Algorithm
- Build and train neural networks with PyTorch

<a id='14'></a>
## Natural Language Processing

### [Part of Speech Tagging](https://github.com/iDataist/Part-of-Speech-Tagging)
Used the Pomegranate library to build a hidden Markov model for part of speech tagging with a universal tagset, and achieved >96% tag accuracy.

Key Skills Demonstrated:
- Process the text data
- Build and train a hidden Markov model

### [Machine Translation](https://github.com/iDataist/Machine-Translation)
Built a deep neural network that functions as part of an end-to-end machine translation pipeline, which accepts English text as input and returns the French translation.

Key Skills Demonstrated:
- Process the text data
- Build and train a deep neural network with Keras

### [Speech Recognizer](https://github.com/iDataist/Speech-Recognizer)
Built a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline, which converts raw audio into feature representations, and then turns them into transcribed text.

Key Skills Demonstrated:
- Extract feature representations from raw audio
- Build deep learning models to obtain a probability distribution over all potential transcriptions

<a id='15'></a>

## AI for Healthcare

### [Pneumonia Detection from Chest X-Rays](https://github.com/iDataist/Pneumonia-Detection-from-Chest-Xray)
Analyzed data from the NIH Chest X-ray Dataset and trained a Convolutional Neural Network with Keras to classify a given chest X-ray for the presence or absence of pneumonia.

Key Skills Demonstrated:
- Perform exploratory data analysis (EDA) on medical imaging data to inform model training and explain model performance
- Extract images from a DICOM dataset
- Train common CNN architectures to classify 2D medical images
- Translate outputs of medical imaging models for use by a clinician

### [Patient Selection for Diabetes Drug Testing](https://github.com/iDataist/Patient-Selection-for-Diabetes-Drug-Testing)
Designed and implemented a regression model with TensorFlow to predict the estimated hospitalization time to help select patients for clinical trials.

Key Skills Demonstrated:
- Create categorical features from Key Industry Code Sets (ICD, CPT, NDC) and reduce dimensionality for high cardinality features
- Use TensorFlow feature columns on both continuous and categorical input features to create derived features (bucketing, cross-features, embeddings)
- Analyze and determine biases for a model for key demographic groups

### [Motion Compensated Pulse Rate Estimation](https://github.com/iDataist/Motion-Compensated-Pulse-Rate-Estimation)
Created an algorithm that combines information from the Inertial Measurement Unit (IMU) and Photoplethysmography (PPG) sensors and estimates the wearer’s pulse rate in the presence of motion.

Key Skills Demonstrated:
- Preprocess data (eliminate “noise”) collected by IMU, PPG, and ECG sensors based on mechanical,
physiology and environmental effects on the signal.
- Evaluate algorithm performance without ground truth labels
- Generate a pulse rate algorithm that combines information from the PPG and IMU sensor streams