# project_6
# Bank Risk Controller Systems
# Project Title: Bank Risk Controller Systems
# Skills & Takeaways:
Python for Data Processing and Machine Learning
Analytics and Statistics for Data Exploration
Plotting for Data Visualization
Streamlit for Building Interactive Web Applications
Machine Learning (Classification Models)
Deep Learning (YOLOv5 for Human Prediction)
Generative AI for Predictions
# Domain:
Banking and Financial Services
# Problem Statement:
Given historical loan data, predict whether a customer will default on a loan (TARGET column in the dataset). The objective is to assess the risk of loan default, which helps financial institutions make informed decisions regarding loan approvals.

# Business Use Cases:
Risk Management:

Banks and financial institutions can use the predictive model to assess the risk of potential borrowers defaulting on loans, thus improving decision-making for loan approvals.
Customer Segmentation:

Segment customers based on their risk profiles, tailoring financial products and services to different segments.
Credit Scoring:

Enhance traditional credit scoring models with predictive analytics to improve loan approval accuracy.
Fraud Detection:

Identify patterns in loan applications that could indicate potential fraud.
Approach:
Data Collection:

Gather historical loan data, including customer details, loan information, and loan repayment records.
Data Preprocessing:

Clean and preprocess the data by handling missing values, outliers, and encoding categorical variables.
Exploratory Data Analysis (EDA):

Perform EDA to understand data distributions and relationships between various features, identifying key factors that influence loan defaults.
Feature Engineering:

Create new features that could improve the predictive power of the model, such as deriving new columns from existing ones.
# Model Selection: Various machine learning algorithms were evaluated to determine the best-performing model for loan default prediction:

Logistic Regression
Decision Trees
Random Forest
Gradient Boosting
ExtraTreesClassifier
# Model Training:

Train the selected machine learning models using the training dataset to identify the most suitable algorithm for loan default prediction.
Model Evaluation:

Evaluate model performance using various metrics, including accuracy, precision, recall, and F1-score.
Hyperparameter Tuning:

Optimize model parameters to enhance predictive performance.
# Model Deployment:

Deploy the trained model for real-time predictions and integrate it with business systems to allow decision-makers to use it effectively.
# YOLOv5 for Human Prediction:

The app also includes a YOLOv5 model to provide human detection and prediction. This can be used to enhance security and authentication processes, ensuring that only authorized personnel can access sensitive bank data.
Technology Stack:
Python (for Data Preprocessing, Machine Learning)
Streamlit (for Web Application Deployment)
YOLOv5 (for Human Prediction)
Pandas & Numpy (for Data Handling)
Scikit-learn (for Machine Learning Models)
Matplotlib & Seaborn (for Data Visualization)
Pickle (for Model Serialization)
# Streamlit App - Features and Login:
The Streamlit app allows users to interact with the model and predict whether a loan applicant will default.
The app provides a user-friendly interface to input data and receive predictions in real-time.
Login functionality is implemented to ensure that only authorized personnel can access sensitive bank customer information. This adds a layer of security to the system.
The user interface is designed to ensure that predictions and sensitive data are handled securely, with necessary data transformations and model predictions displayed on the web interface.
How to Access the Streamlit App:
# Login:

Upon accessing the app, users are prompted to log in with credentials (username and password).
Only users with valid credentials can access the prediction features and view customer-related information.
Prediction Interface:

After logging in, users can interact with the model by inputting details of a customer, such as credit amount, income, employment history, etc.
The model will output whether the customer is likely to default or not.
Human Prediction System with YOLOv5:

The YOLOv5 model is integrated into the app for human prediction and detection.
This can be used to enhance security measures and ensure that only authorized users access the prediction system.

# Model Deployment and Access Instructions:
Setting up the Streamlit app:

Clone the repository to your local machine.
Install required libraries using:
bash
Copy code
pip install -r requirements.txt
Running the App:

Start the Streamlit application using the following command:
bash
Copy code
streamlit run app.py
Login Authentication:

Use the provided login interface to authenticate yourself.

Sample Images
![image](https://github.com/user-attachments/assets/3e422739-1491-4295-9034-5d91d2d59f28)
![image](https://github.com/user-attachments/assets/67cbbec9-9f73-48e0-8e6e-a31214fe59e8)
![image](https://github.com/user-attachments/assets/1310ed73-74c5-45e5-a4de-86859e8937d8)
![image](https://github.com/user-attachments/assets/8e566896-c75d-4df5-b811-2fb2683fba79)
![image](https://github.com/user-attachments/assets/e1357c55-de32-425b-bc93-5c1100d7648a)






License:
This project is licensed under the MIT License - see the LICENSE file for details.
