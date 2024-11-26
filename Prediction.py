
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import re
import itertools
import pdfplumber


#model_imports
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st
import torch
from PIL import Image
import io

# Set up Streamlit configuration
st.set_page_config(page_title="Bank Risk Controller Systems", layout="wide")

# Load sample data
sample = pd.read_csv(r"C:\Users\LENOVO\Desktop\Data_science_guvi\projects\Bank Risk Controller Systems\eda_data.csv")
model_data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Data_science_guvi\projects\Bank Risk Controller Systems\model_data.csv")

def load_pickle(file_name):
    """Load a pickle file and return its content."""
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
    """Get user input from Streamlit form."""
    st.subheader(":violet[Fill all the fields and press the button below to view **The user Defaulter or Non-defaulter**:]")
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
    """Load the Extra Trees Classifier model."""
    try:
        with open('ET_Classifier_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found: ET_Classifier_model.pkl")
        return None
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

def data_transformation_for_the_model(df):
    """Transform data using pre-loaded encoders."""
    df = df.copy()  # Avoid modifying the original DataFrame
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    return df


def plot(sample, col, title, pie_colors):
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
    """Main function to control Streamlit app flow."""
    global text_chunks  # Declare text_chunks as global for use in retrieve_relevant_chunks
    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Select options")
        choice = st.radio("Navigation", ["Data", "EDA", "Model", "Detect"])
        st.info("This project application helps you predict defaulters and non-defaulters")
    
    if choice == "Data":
        st.title(":violet[Welcome to the Bank Risk Controller Systems Prediction App]")   
        st.write('### :violet[Model Performance Metrics]')
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
                '[[120689, 41005], [34833, 127687]]']
        }
        metrics_df = pd.DataFrame(metrics) 
        st.dataframe(metrics_df)
        st.write(":violet[Sample Dataset]")
        st.write(model_data.head(11))
        st.write('## :violet[Created by] \n ### C.Madhumitha')
        
    if choice == "Model":
        st.title(":violet[Bank Risk Controller Systems App]")
        user_input_data = get_user_input()

        if st.button("Predict"):
            df = data_transformation_for_the_model(user_input_data)
            model = load_model()
            if model is not None:
                prediction = model.predict(df)
                st.success(f'Prediction: {"Defaulter" if prediction[0] == 1 else "Non-defaulter"}')
            

    
    if choice == "Detect":
        
        st.title(":violet[YOLOv5 Object Detection]")
        
        # Load the YOLOv5 model (provide the local path to your YOLOv5 model)
        model_path = r'C:\Users\LENOVO\Desktop\Data_science_guvi\projects\Bank Risk Controller Systems\yolov5s.pt'

        # Load YOLOv5 model from local .pt file
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        # Streamlit title and description
        #st.title("YOLOv5 Object Detection")
        st.write("Upload an image to detect objects using YOLOv5 model")

        # File uploader widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)

            # Display the image in the app
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Run YOLOv5 detection
            st.write("Running YOLOv5 model on the uploaded image...")

            # Inference on the uploaded image
            results = model(image)

            # Show the results (image with bounding boxes)
            st.image(results.render()[0], caption='Processed Image with Detections', use_column_width=True)

            # Show result data (labels, confidence)
            st.write("Detected objects and their confidence scores:")
            detected_labels = results.names
            predictions = results.xywh[0]  # Bounding boxes (x, y, width, height)
            confidences = results.pred[0][:, -2]  # Confidence scores

            for i, label in enumerate(predictions):
                st.write(f"Prediction {i + 1}: {detected_labels[int(label[5])]} - Confidence: {confidences[i]:.4f}")


                    
    if choice == "EDA":
        
        col1,col2= st.columns(2)
        
        with col1:
            st.title(":violet[Distribution of Gender in dataset]")
            pie_colors = ["#ff006e", "#ffd60a", '#6a0dad', '#ff4500']
            plot(sample, "CODE_GENDER", "Gender", pie_colors)
            st.markdown(
            '''
            <p style="color: #ffee32; font-size: 30px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">67%</span> Female and 
                        <span style="color: #ef233c; font-size: 40px ">33%</span> Male<br>
                </strong>
            </p>''',
            unsafe_allow_html=True)
        with col2:
            st.title(":violet[Distribution of target]")
            pie_colors = ['#d00000', '#ffd500','#f72585', '#ff4500']
            plot(sample, "TARGET", "Target", pie_colors)
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        1 - defaulter,
                        0 - non-defaulter<br>
                        he/she had late payment more than X days are defaulter.<br>
                        <span style="color: #ef233c; font-size: 40px ">8%</span> Defaulter, <span style="color: #ef233c; font-size: 40px ">92%</span> Non-defaulter
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
            
        col1,col2= st.columns(2)
        
        with col1: 
            st.title(":violet[Distribution in Contract types in loan_data] \n - Revolving loan \n - Cash loan")
            
            pie_colors = ['#ff006e', '#d00000','#f72585', '#ff4500']
            plot(sample, "NAME_CONTRACT_TYPE_x", "Loan Type", pie_colors)
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">8%</span>
                        of user in dataset are Revolving loan type its a form of credit allows the user to borrower to withdraw,
                        repay, and withdraw again up to a certain limit.
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
        with col2:
            st.title(":violet[Distribution of loan type by Gender] \n - Female are more loan taked then male \n - Cash loans is always prefered over Revolving loans by both genders")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(x="NAME_CONTRACT_TYPE_x", hue="CODE_GENDER", data=sample, palette=["#00bbf9", "#f15bb5", "#ee964b"], ax=ax)
            ax.set_facecolor("#020202")
            ax.set_title("Distribution of Contract Type by Gender")
            st.pyplot(plt)
            plt.close()
        
        col1,col2= st.columns(2)
        
        with col1: 
            st.title(":violet[Distribution of Own car]")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            
            pie_colors = ['#ff006e', '#d00000','#f72585', '#ff4500']
            plot(sample, "FLAG_OWN_CAR", "Distribution of own car", pie_colors)
          
            
            st.markdown(
            '''
            <p style="color: #ffee32; font-size: 30px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">34%</span>of users own a car. 
                        <span style="color: #ef233c; font-size: 47px ">66%</span> users didn't own a car <br>
                </strong>
            </p>''',
            unsafe_allow_html=True)
            
        with col2:
            st.title(":violet[Distribution Owning a Car by Gender]")
            fig = plt.figure(figsize=(4, 2))
            ax = plt.subplot(121)

            value_counts = sample[sample["FLAG_OWN_CAR"] == "Y"]["CODE_GENDER"].value_counts()
            pie_colors = ['#ff9999', '#66b3ff', '#99ff99']
            value_counts.plot.pie(
                autopct="%1.0f%%",
                colors=pie_colors[:len(value_counts)],
                startangle=60,
                wedgeprops={"linewidth": 2, "edgecolor": "k"},
                explode=[0.1] * len(value_counts),
                shadow=True,
                ax=ax
            )
            ax.set_title("Distribution Owning a Car by Gender")   
            st.pyplot(plt)
            plt.close()
            
            st.markdown(
            '''
            <p style="color: #ffee32; font-size: 30px; text-align: center;">
                    <strong> Out of own car users
                        <span style="color: #ef233c; font-size: 40px ">55%</span> are Male. 
                        <span style="color: #ef233c; font-size: 40px ">45%</span> are Female.
                </strong>
            </p>''',
            unsafe_allow_html=True)
            
        col1,col2= st.columns(2)
        
        with col1: 
            st.title(":violet[Distribution of owning a house or flat]")
            
            pie_colors = ['#ff006e', '#d00000','#f72585', '#ff4500']
            plot(sample, "FLAG_OWN_REALTY", "Distribution of owning a house or flat", pie_colors)
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">72%</span>
                        of users own a flat or house.
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
        with col2:
            st.title(":violet[Distribution of owning a house or flat by gender]")
            
            fig = plt.figure(figsize=(5, 3))
            ax = plt.subplot(121)
            value_counts = sample[sample["FLAG_OWN_REALTY"] == "Y"]["CODE_GENDER"].value_counts()
            pie_colors = ['#ff9999', '#66b3ff', '#99ff99']
            value_counts.plot.pie(
                autopct="%1.0f%%",
                colors=pie_colors[:len(value_counts)],
                startangle=60,
                wedgeprops={"linewidth": 2, "edgecolor": "k"},
                explode=[0.1] * len(value_counts),
                shadow=True,
                ax=ax
            )
            ax.set_title("Distribution of owning a house or flat by gender")
            st.pyplot(plt)
            plt.close()
            
            st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong> Out of own flat users
                        <span style="color: #ef233c; font-size: 40px ">69%</span> are female
                        <span style="color: #ef233c; font-size: 40px ">31%</span> are male
                    </strong>
                </p>
                ''',
                unsafe_allow_html=True)
            
        col1,col2= st.columns(2)
        
        with col1:
            st.title(":violet[Distribution of Number of Children by Repayment Status]")
            
            fig = plt.figure(figsize=(5, 5))
            plt.subplot(211)

            sns.countplot(x="CNT_CHILDREN", hue="TARGET", data=sample, palette="pastel")
            plt.legend(loc="upper right", prop={'size': 20})
            plt.title("Distribution of Number of Children by Repayment Status")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
            st.write("- if the user has less childer they most likely to be Non-defaulter")
            
        with col2:
            st.title(":violet[Distribution of Number of Family Members by Repayment Status]")
            
            fig = plt.figure(figsize=(5, 5))
            plt.subplot(211)
            
            sns.countplot(x="CNT_FAM_MEMBERS", hue="TARGET", data=sample, palette="Set2")
            plt.legend(loc="upper right",prop={'size': 18})
            plt.title("Distribution of Number of Family Members by Repayment Status")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
            st.write("""- if the user has less family members there Repayment status is high.\n - if the user has only family member of 2 Repayment Status is very higy comparing to higher family members.
                     """)
        
        st.title(":violet[Distribution of Defaulter and non-Defaulter] \n - Loan type \n- Gender \n - Own car \n- Own house")

        default = sample[sample["TARGET"]==1][[ 'NAME_CONTRACT_TYPE_x', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
        non_default = sample[sample["TARGET"]==0][[ 'NAME_CONTRACT_TYPE_x', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

        d_cols = ['NAME_CONTRACT_TYPE_x', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        d_length = len(d_cols)

        fig = plt.figure(figsize=(16,4))
        for i,j in itertools.zip_longest(d_cols,range(d_length)):
            plt.subplot(1,4,j+1)
            default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism"),startangle = 90,
                                                wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
            circ = plt.Circle((0,0),.7,color="white")
            plt.gca().add_artist(circ)
            plt.ylabel("")
            plt.title(i+"-Defaulter")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

        fig = plt.figure(figsize=(16,4))
        for i,j in itertools.zip_longest(d_cols,range(d_length)):
            plt.subplot(1,4,j+1)
            non_default[i].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",3),startangle = 90,
                                                wedgeprops={"linewidth":1,"edgecolor":"white"},shadow =True)
            circ = plt.Circle((0,0),.7,color="white")
            plt.gca().add_artist(circ)
            plt.ylabel("")
            plt.title(i+"-Repayer")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.write("""              
        - 3% Percentage of Cash Loans has more defaults than Revolving Loans.
        - 10% Percentage of males more defaults than non defaulters. 
        - 8% Percentage of female are more repayers.
        """)
        
        
        st.title(":violet[Comparing summary statistics between defaulters and non - defaulters for loan amounts]")
       
        cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT_x','AMT_ANNUITY_x', 'AMT_GOODS_PRICE_x']
        df = sample.groupby("TARGET")[cols].describe().transpose().reset_index()
        df = df[df["level_1"].isin(['mean', 'std', 'min', 'max'])] 

        df_x = df[["level_0", "level_1", 0]].rename(columns={'level_0': "amount_type", 'level_1': "statistic", 0: "amount"})
        df_x["type"] = "REPAYER"

        df_y = df[["level_0", "level_1", 1]].rename(columns={'level_0': "amount_type", 'level_1': "statistic", 1: "amount"})
        df_y["type"] = "DEFAULTER"

        df_new = pd.concat([df_x, df_y], axis=0)

        stat = df_new["statistic"].unique().tolist()
        length = len(stat)

        plt.figure(figsize=(8, 8))

        for i, j in itertools.zip_longest(stat, range(length)):
            plt.subplot(2, 2, j + 1)
            sns.barplot(x="amount_type", y="amount", hue="type",
                        data=df_new[df_new["statistic"] == i], palette=["g", "r"])
            plt.title(i + " -- Defaulters vs Non-defaulters")
            plt.xticks(rotation=35)
            plt.subplots_adjust(hspace=0.4)
            plt.gca().set_facecolor("lightgrey")

        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.write("""              
        # Income of users
        - 1 . Average income of users who default and non-defaulter are almost same.

        - 2 . Standard deviation in income of default is very high compared to non-defaulter.

        - 3 . Default also has maximum income earnings

        # Credit amount of the loan credited , Loan annuity, Amount goods price

        - 1 . Statistics between *credit amounts*, *Loan annuity* and *Amount goods price* given in the data the default and non-defaulter are almost similar.
        """)
       
        st.title(":violet[Average Income,credit,annuity & goods_price by gender]")
        
        df1 = sample.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()

        df_f = df1[["index", "F"]].rename(columns={'index': "amt_type", 'F': "amount"})
        df_f["gender"] = "FEMALE"

        df_m = df1[["index", "M"]].rename(columns={'index': "amt_type", 'M': "amount"})
        df_m["gender"] = "MALE"

        df_xna = df1[["index", "XNA"]].rename(columns={'index': "amt_type", 'XNA': "amount"})
        df_xna["gender"] = "XNA"

        df_gen = pd.concat([df_m, df_f, df_xna], axis=0)


        plt.figure(figsize=(6, 3))
        ax = sns.barplot(x="amt_type", y="amount", data=df_gen, hue="gender", palette="Set1")
        plt.title("Average Income, Credit, Annuity & Goods Price by Gender")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
       
        st.title(":violet[Distribution of Suite type]\n - NAME_TYPE_SUITE - Who was accompanying user when he was applying for the loan.")
        
        col1,col2= st.columns(2)
        with col1:
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            sns.countplot(y=sample["NAME_TYPE_SUITE_x"],
                        palette="Set2",
                        order=sample["NAME_TYPE_SUITE_x"].value_counts().index[:5])
            plt.title("Distribution of Suite Type")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        with col2:
            plt.figure(figsize=(10, 3))
            plt.subplot(122)
            sns.countplot(y=sample["NAME_TYPE_SUITE_x"],
                        hue=sample["CODE_GENDER"],
                        palette="Set2",
                        order=sample["NAME_TYPE_SUITE_x"].value_counts().index[:5])
            plt.ylabel("")
            plt.title("Distribution of Suite Type by Gender")
            plt.legend(loc="lower right", prop={'size': 10})
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
        st.title(":violet[Distribution of client income type]\n - NAME_INCOME_TYPE Clients income type (businessman, working, maternity leave...)")
        
        col1,col2= st.columns(2)
        with col1:
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            sns.countplot(y=sample["NAME_INCOME_TYPE"],
                        palette="Set2",
                        order=sample["NAME_INCOME_TYPE"].value_counts().index[:4])
            plt.title("Distribution of Client Income Type")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        with col2:
            plt.figure(figsize=(10, 3))
            plt.subplot(122)
            sns.countplot(y=sample["NAME_INCOME_TYPE"],
                        hue=sample["CODE_GENDER"],
                        palette="Set2",
                        order=sample["NAME_INCOME_TYPE"].value_counts().index[:4])
            plt.ylabel("")
            plt.title("Distribution of Client Income Type by Gender")
            plt.legend(loc="lower right", prop={'size': 10})
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
        
         
        st.title(":violet[Distribution of Education type by loan repayment status]\n - NAME_EDUCATION_TYPE Level of highest education the user achieved..")
        col1,col2= st.columns(2)
        with col1:
            # Plot for Repayers
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            sample[sample["TARGET"]==0]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(
                fontsize=12,
                autopct="%1.0f%%",
                colors=sns.color_palette("inferno"),
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
                shadow=True
            )
            plt.gca().add_artist(plt.Circle((0, 0), .7, color="white"))  # Add a white circle to create a donut-like effect
            plt.title("Distribution of Education Type for Repayers", color="b")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        
        with col2:
            # Plot for Defaulters
            plt.figure(figsize=(10, 3))
            plt.subplot(122)
            sample[sample["TARGET"]==1]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(
                fontsize=12,
                autopct="%1.0f%%",
                colors=sns.color_palette("Set2"),
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
                shadow=True
            )
            plt.gca().add_artist(plt.Circle((0, 0), .7, color="white"))  # Add a white circle to create a donut-like effect
            plt.title("Distribution of Education Type for Defaulters", color="b")
            plt.ylabel("")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong>
                        <span style="color: #ef233c; font-size: 40px ">8%</span> perentage of users with higher education are less defaulter compared to user non-defaulter.
                    </strong>
                </p>
                ''', unsafe_allow_html=True)
        
        st.title(":violet[Distribution of Education type by loan repayment status]")
        
        edu = sample.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index().sort_values(by='AMT_INCOME_TOTAL', ascending=False)
        # Create the bar plot
        fig = plt.figure(figsize=(10, 5))
        ax = sns.barplot(x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', data=edu, hue='NAME_EDUCATION_TYPE', palette="seismic")
        ax.set_facecolor("k")
        plt.title("Average Earnings by Different Professions and Education Types")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.title(":violet[Distribution normalized population of region where client lives by loan repayment status]\n - REGION_POPULATION_RELATIVE - Normalized population of region where client lives (higher number means the client lives in more populated region).")
        fig = plt.figure(figsize=(10,5))

        plt.subplot(121)
        sns.violinplot(y=sample[sample["TARGET"]==0]["REGION_POPULATION_RELATIVE"],
                    x=sample[sample["TARGET"]==0]["NAME_CONTRACT_TYPE_x"],
                    palette="Set1")
        plt.title("Distribution of Region Population for Non-Default Loans", color="b")

        plt.subplot(122)
        sns.violinplot(y=sample[sample["TARGET"]==1]["REGION_POPULATION_RELATIVE"],
                    x=sample[sample["TARGET"]==1]["NAME_CONTRACT_TYPE_x"],
                    palette="Set1")
        plt.title("Distribution of Region Population for Default Loans", color="b")

        plt.subplots_adjust(wspace=.2)
        fig.set_facecolor("lightgrey")
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        st.markdown(
                '''
                <p style="color: #ffee32; font-size: 20px; text-align: center;">
                    <strong> Point to infer from the graph
                    \n- In High population density regions people are less likely to default on loans.
                    </strong>
                </p>
                ''', unsafe_allow_html=True)    

        
                
   


if __name__ == "__main__":
    main()