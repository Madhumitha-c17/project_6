import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import re
import itertools
import pdfplumber

# Model imports
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import plotly.express as px

import torch
from PIL import Image

from PIL import Image
import torch
import seaborn as sns


# Streamlit setup
st.set_page_config(page_title="Bank Risk Controller Systems", layout="wide")

# Placeholder for user authentication
USER_CREDENTIALS = {"admin": "password123", "user1": "userpass"}  

# Login function
def login():
    """Login page for user authentication."""
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")


# Check authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    # Main App Content (Unchanged)
    def load_pickle(file_name):
        """Load a pickle file."""
        try:
            with open(file_name, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            st.error(f"File not found: {file_name}")
            return None
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            return None

    # Load datasets
    sample = pd.read_csv(r"C:\Users\LENOVO\Desktop\Data_science_guvi\projects\Bank Risk Controller Systems\eda_data.csv")
    model_data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Data_science_guvi\projects\Bank Risk Controller Systems\model_data.csv")

    def load_pickle(file_name):
        """Load a pickle file."""
        try:
            with open(file_name, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            st.error(f"File not found: {file_name}")
            return None
        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            return None

    # Load encoders
    encoders = load_pickle('label_encoders.pkl')
    if encoders is None:
        st.stop()

    # Retrieve classes from encoders
    ORGANIZATION_TYPE = encoders['ORGANIZATION_TYPE'].classes_.tolist()
    OCCUPATION_TYPE = encoders['OCCUPATION_TYPE'].classes_.tolist()

    def get_user_input():
        """Retrieve user inputs."""
        st.subheader(":blue[Fill all the details below and press the button **Predict** to know if the customer is Defaulter / Non-defaulter]")
        cc1, cc2 = st.columns([2, 2])
        with cc1:
            BIRTH_YEAR = st.number_input("Birth Year (YYYY):", min_value=1950, max_value=2024)
            AMT_CREDIT = st.number_input("Credit Amount of loan:")
            AMT_ANNUITY = st.number_input("Loan Annuity:")
            AMT_INCOME_TOTAL = st.number_input("Income of the user:")
            ORGANIZATION_TYPE_input = st.selectbox("Organization Type:", ORGANIZATION_TYPE)
        with cc2:
            OCCUPATION_TYPE_input = st.selectbox("Occupation Type:", OCCUPATION_TYPE)
            EXT_SOURCE_2 = st.number_input("Score from External-2 data source:")
            EXT_SOURCE_3 = st.number_input("Score from External-3 data source:")
            REGION_POPULATION_RELATIVE = st.number_input("Population of the region:")
            HOUR_APPR_PROCESS_START = st.number_input("Hour user applied for the loan:")
            EMPLOYMENT_START_YEAR = st.number_input("Employment Start Year:", min_value=1950, max_value=2024)

        user_input_data = {
            'BIRTH_YEAR': BIRTH_YEAR,
            'AMT_CREDIT': AMT_CREDIT,
            'AMT_ANNUITY': AMT_ANNUITY,
            'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
            'ORGANIZATION_TYPE': ORGANIZATION_TYPE_input,
            'OCCUPATION_TYPE': OCCUPATION_TYPE_input,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'EXT_SOURCE_3': EXT_SOURCE_3,
            'REGION_POPULATION_RELATIVE': REGION_POPULATION_RELATIVE,
            'HOUR_APPR_PROCESS_START': HOUR_APPR_PROCESS_START,
            'EMPLOYMENT_START_YEAR': EMPLOYMENT_START_YEAR
        }
        return pd.DataFrame(user_input_data, index=[0])

    def load_model():
        """Load the Extra Trees Classifier model and check its performance."""
        try:
            with open('ET_Classifier_model.pkl', 'rb') as file:
                model = pickle.load(file)
                #st.write(f"Model loaded successfully: {model}")  # Check the model type
                return model
        except FileNotFoundError:
            st.error("Model file not found: ET_Classifier_model.pkl")
            return None
        except Exception as e:
            st.error(f"Error loading model file: {e}")
            return None

    def data_transformation_for_the_model(df):
        """Transform data using encoders and check the transformation."""
        df = df.copy()
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
                #st.write(f"Transformed {col}: {df[col].values}")  # Debug transformation
        return df
    def plot(sample, col, title, pie_colors):
        """Generate a distribution plot."""
        bar_color = '#7B68EE'
        plt.figure(figsize=(10, 5))
        value_counts = sample[col].value_counts()

        plt.subplot(121)
        value_counts.plot.pie(
            autopct="%1.0f%%",
            colors=pie_colors[:len(value_counts)],
            startangle=60,
            wedgeprops={"linewidth": 2, "edgecolor": "k"},
            explode=[0.1] * len(value_counts),
            shadow=True
        )
        plt.title(f"Distribution of {title}")
        plt.subplot(122)
        ax = sample[col].value_counts().plot(kind="barh", color=bar_color)
        for i, (value, label) in enumerate(zip(value_counts.values, value_counts.index)):
            ax.text(value, i, f' {value}', weight="bold", fontsize=12, color='black')
        plt.title(f"Count of {title}")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    def main():
        """Main Streamlit app."""
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/6153/6153788.png", width=150)
            st.title("Select options")
            choice = st.radio("Navigation", ["Data", "EDA", "Model", "Detect"])
            st.info("The expected outcome of this project is a robust predictive model that identifies loan defaults.")

        if choice == "Data":
            st.title(":blue[Welcome to Bank Risk Controller System Prediction App]")
            st.write('### :blue[Model Performance Metrics]')
            metrics = {
                'Model': ['ExtraTreesClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],
                'Accuracy': [1.0000, 0.9999, 0.9971, 0.7661],
                'Precision': [1.0000, 0.9999, 0.9971, 0.7665],
                'Recall': [1.0000, 0.9999, 0.9971, 0.7661],
                'F1 Score': [1.0000, 0.9999, 0.9971, 0.7660],
                'ROC AUC': [0.9999, 0.9999, 0.9970, 0.7660],
                'Confusion Matrix': [
                    '[[161689, 5], [0, 162520]]',
                    '[[161675, 19], [0, 162520]]',
                    '[[160746, 948], [0, 162520]]',
                    '[[120689, 41005], [34833, 127687]]'
                ]
            }
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df)
            st.write(":blue[Sample Dataset]")
            st.write(model_data.head(11))
            st.write('## :blue[Created by] \n ### C.Madhumitha')

        if choice == "Model":
            st.title(":blue[Bank Risk Controller System]")
            user_input_data = get_user_input()
            
            if st.button("Predict"):
                df = data_transformation_for_the_model(user_input_data)
                #st.write(f"Input data after transformation: {df}")  # Debug input data
                
                model = load_model()
                if model is not None:
                    prediction = model.predict(df)
                    #st.write("Prediction Output:", prediction)
                    #st.write(f"Model Prediction Output: {prediction}")  # Debug model prediction output
                    if prediction[0] == 1:
                        st.success("Prediction: Defaulter")
                    else:
                        st.success("Prediction: Non-defaulter")

        if choice == "Detect":
            st.title(":blue[YOLOv5 Object Detection]")
            model_path = r'C:\Users\LENOVO\Desktop\Data_science_guvi\projects\Bank Risk Controller Systems\yolov5s.pt'
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                results = model(image)
                st.image(results.render()[0], caption='Processed Image', use_column_width=True)
                st.write("Detected objects:")
                for *xyxy, conf, cls in results.xyxy[0].tolist():
                    st.write(f"Class: {cls}, Confidence: {conf}")

        if choice == "EDA":
            st.title(":blue[Exploratory Data Analysis (EDA)]")
            st.write("Below are the distributions of the most relevant variables in the dataset.")

            # Identify numerical and categorical columns
            numerical_cols = sample.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = sample.select_dtypes(include=['object', 'category']).columns

            # Plot distributions for categorical columns
            st.subheader(":blue[Categorical Variables Distribution]")
            pie_colors = sns.color_palette("pastel")

            # Display categorical distributions in a 3-column layout
            cat_columns = st.columns(3)
            for idx, col in enumerate(categorical_cols):
                with cat_columns[idx % 3]:  # Rotate between columns
                    st.write(f"### {col}")
                    plot(sample, col, title=col, pie_colors=pie_colors)

            # Plot distributions for numerical columns
            st.subheader(":blue[Numerical Variables Distribution]")

            # Display numerical distributions in a 3-column layout
            num_columns = st.columns(3)
            for idx, col in enumerate(numerical_cols):
                with num_columns[idx % 3]:  # Rotate between columns
                    st.write(f"### {col}")

                    # Histogram
                    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust size for layout
                    sns.histplot(sample[col], kde=True, bins=30, color='#7B68EE', ax=ax)
                    ax.set_title(f"Distribution of {col}")
                    st.pyplot(fig)
                    plt.close(fig)

                    # Boxplot
                    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust size for layout
                    sns.boxplot(x=sample[col], color='#FFA07A', ax=ax)
                    ax.set_title(f"Boxplot of {col}")
                    st.pyplot(fig)
                    plt.close(fig)

    if __name__ == "__main__":
        main()
