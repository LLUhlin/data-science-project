
# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import * 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from PIL import Image

# Importing models used
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import streamlit as st
import joblib

# df = pd.read_csv('/Users/nima/Downloads/PROJ25/gruppuppgift/02_dataset/student_dropout_success/data.csv', sep=';')
# df = pd.read_csv('./data.csv', sep=';') # using relative path instead to not be locked to 1 computer

@st.cache_data
def data_csv():
    return pd.read_csv('./data.csv', sep=';')

@st.cache_data
def label_fit_transform(target):
    le = LabelEncoder()
    return le.fit_transform(target)

@st.cache_data
def split_data(X, y):
    X_train, X_te_va, y_train, y_te_va = train_test_split(X, y, test_size = 0.3, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_te_va, y_te_va, test_size = 0.5, stratify=y_te_va)
    return X_train, y_train, X_test, y_test, X_val, y_val

@st.cache_data
def load_model(name):
    return joblib.load(name)

@st.cache_data
def scale_ttv(X_train, X_test, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    X_test_scaled = scaler.transform(X_test.astype(np.float64))
    X_val_scaled = scaler.transform(X_val.astype(np.float64))
    return X_train_scaled, X_test_scaled, X_val_scaled

@st.cache_resource
def correlation(dft):
    corr_table = pd.DataFrame(dft.corr()[dft.columns[-1:]]).sort_values(by=['Target'], ascending=False).head(15)
    cmatrix = dft.corr()
    fig = plt.figure(figsize=(20,15))
    sns.heatmap(cmatrix[((cmatrix >= .3) | (cmatrix <= -.6)) & (cmatrix !=1.000)], annot=True, cmap="Greens")
    return corr_table, fig

df = data_csv()

df2 = df.copy(deep=True)
df2 = df2[df2.Target != 'Enrolled']
dft = df2.copy(deep=True)

label = label_fit_transform(dft['Target'])

dft.drop("Target", axis=1, inplace=True)
dft["Target"] = label


X = dft.drop(['Target'], axis=1)
y = dft['Target']  

X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)

X_train_scaled, X_test_scaled, X_val_scaled = scale_ttv(X_train, X_test, X_val)

stat_sum = X_train.agg(['median', 'min', 'max', 'std'])
X_train_sc = pd.DataFrame(X_train_scaled)
stat_sum_sc = X_train_sc.agg(['median', 'min', 'max', 'std'])
df1stat = stat_sum.transpose()
df2stat = stat_sum_sc.transpose()


nav = st.sidebar.radio("NAVIGATION BAR",["START", "DATA EXPLORATION", "DATA PREPROCESSING", "MODEL DEVELOPMENT", "MODEL EVALUATION", "PROJECT CONCLUSIONS"])


## Load models - Order of creation
LR_RSCV = load_model('Logisitic_Regression_RandomizedSearchCV.joblib')
RFC_RSCV = load_model('Random_Forest_Classifier_RandomizedSearchCV.joblib')
SVC_RSCV = load_model('Support_Vector_Classifier_RandomizedSearchCV.joblib')
GBC_RSCV = load_model('Gradient_Boosting_Classifier_RandomizedSearchCV.joblib')
LDA_RSCV = load_model('Linear_Discriminant_Analysis_RandomizedSearchCV.joblib')
LR_RFC_SVC_GBC_LDA_VC = load_model('LR_RFC_SVC_GBC_LDA_VotingClassifier.joblib')
LR_RFC_SVC_GBC_VC = load_model('LR_RFC_SVC_GBC_VotingClassifier.joblib')
LR_RFC_SVC_LDA_VC = load_model('LR_RFC_SVC_LDA_VotingClassifier.joblib')
LR_RFC_GBC_LDA_VC = load_model('LR_RFC_GBC_LDA_VotingClassifier.joblib')
LR_SVC_GBC_LDA_VC = load_model('LR_SVC_GBC_LDA_VotingClassifier.joblib')
RFC_SVC_GBC_LDA_VC = load_model('RFC_SVC_GBC_LDA_VotingClassifier.joblib')
ENSEMB_FINAL = load_model('ensemb_final.joblib')

if nav == "START":
    st.title("PROJ25: Machine Learning Project of Student Graduate/Dropout dataset")
    
    st.write("""The purpose of this Machine Learning Project is to predict if a student will GRADUATE or DROPOUT using a dataset created by SATDAP - Capacitação da Administração Pública under grant POCI-05-5762-FSE-000191, Portugal. The main purpose of the predictions is to find and help students that are at risk of dropping out.""")
    st.write("""The project has been divided into 5 steps.""")
    st.write("""• DATA EXPLORATION""")
    st.write("""• DATA PREPROCESSING""")
    st.write("""• MODEL DEVELOPMENT""")
    st.write("""• MODEL EVALUATION""")
    st.write("""• PROJECT CONCLUSIONS""")
    
    # image = Image.open('/Users/nima/Downloads/PROJ25/gruppuppgift/04_presentation/dropout.jpg')
    image = Image.open('./dropout.jpg')

    st.image(image, caption='GRADUATE OR DROPOUT?')

if nav == "DATA EXPLORATION":
    st.header("DATA EXPLORATION (EDA)")
    st.write("""As a first step we have been exploring and analyzing the dataset to get a better understanding of it.""")
    st.write("""Note that we are only looking at the data not touching it!""")
    
    #st.write("""The general info about the dataset.""")
    #st.table(df.info())
    
    #st.write("""The features in the dataset.""")
    #st.write(df.columns.tolist())
    
    st.write("""Displaying the 10 first rows in the dataset to better understand what the data looks like.""")
    st.dataframe(df.head(10))
    
    st.write("""Display of the number of rows and columns in the dataset:""")
    df_shape = df.shape
    #df_df = pd.DataFrame({df_shape}, index=['rows', 'columns'])
    st.dataframe(df_shape)
    
    st.write("""Showing all the features in the dataset and their datatypes.""")
    st.table(df.dtypes)
    
    st.write("""Checking the unique values in the Target attribute and also the distribution of the unique values.""")
    st.dataframe(np.unique(df['Target'], return_counts=True))
    
    st.write("""Checking if there is any missing values in the dataset.""")
    st.table(df.isnull().sum())  


if nav == "DATA PREPROCESSING":
    st.header("DATA PREPROCESSING")
    st.write('In this step we will start to perform preprocessing on the dataset.')
        
    st.write("""The decision was made to drop the rows which had 'Enrolled' in the Target attribute, since we are only interested in predicting if a student is going to dropout or graduate.""")
    st.write("""It should be added that an initially 'Enrolled' was not dropped just to see how different models performed on the complete dataset. The result on the 15 models we trained was between 67 and 78 % in accuracy.""")
    st.write("""Further note that 'Enrolled' was by far the smallest part(18%), which made the dataset highly imbalanced.""")    
    
    st.write("""Again checking the unique values in the Target attribute after dropping 'Enrolled'.""")
    # df2 = df.copy(deep=True)
    # df2 = df2[df2.Target != 'Enrolled']
    st.dataframe(np.unique(df2['Target'], return_counts=True))
    
    st.write("""The number of rows and columns in the dataset after dropping 'Enrolled':""")
    st.dataframe(df2.shape)
    
    st.write("""Since the Target's datatype was object, it was transformed to numerical using the method LabelEncoder.""")
    st.write("""The result after transforming the Target to numerical values, Dropout=0, Graduate=1.""")
    # dft = df2.copy(deep=True)
    # le = LabelEncoder()
    # label = le.fit_transform(dft['Target'])
    # dft.drop("Target", axis=1, inplace=True)
    # dft["Target"] = label
    st.dataframe(np.unique(dft['Target'], return_counts=True))
    
    st.write("""Table for the correlation between the target and the features in the dataset, displaying the top 15.""")
    corr_table, cmatrix = correlation(dft)

    st.table(corr_table)
    
    st.write("""Correlation Matrix for the dataset.""")

    st.write(cmatrix)
    # st.pyplot(cmatrix)

    st.write("""Since all data is in one dataset, we need to split it into input variables(X) and 
             output variables(y).""") 
    st.write("""After that the dataset was split into a training, test and validation dataset.""")   
    # X = dft.drop(['Target'], axis=1)
    # y = dft['Target']  
    st.write("""-----------------------""")
    # X_train, X_te_va, y_train, y_te_va = train_test_split(X, y, test_size = 0.3, stratify=y)
    # X_test, X_val, y_test, y_val = train_test_split(X_te_va, y_te_va, test_size = 0.5, stratify=y_te_va)
    st.write('Training dataset part:', "{:.2%}".format((y_train.count()/y.count())))
    st.write('Test dataset part:', "{:.2%}".format((y_test.count()/y.count())))
    st.write('Validation dataset part:', "{:.2%}".format((y_val.count()/y.count())))
    st.write()
    st.write('Distribution in Training dataset: Graduate:', "{:.2%}".format((np.count_nonzero(y_train)/y_train.count())),
     '; Dropout:',"{:.2%}".format(((y_train.count() - np.count_nonzero(y_train))/y_train.count())))
    st.write('Distribution in Test dataset: Graduate:', "{:.2%}".format((np.count_nonzero(y_test)/y_test.count())),
     '; Dropout:',"{:.2%}".format(((y_test.count() - np.count_nonzero(y_test))/y_test.count())))
    st.write('Distribution in Validation dataset: Graduate:', "{:.2%}".format((np.count_nonzero(y_val)/y_val.count())),
     '; Dropout:',"{:.2%}".format(((y_val.count() - np.count_nonzero(y_val))/y_val.count())))
    st.write()
    st.write('Info training dataset target:', np.unique(y_train, return_counts=True))
    st.write('Info test dataset target:', np.unique(y_test, return_counts=True))
    st.write('Info validation dataset target:', np.unique(y_val, return_counts=True))
    st.write("""-----------------------""")
    
    st.write("""The dataset is then scaled with the Standardscaler method. This improves the performance of the machine learning models. Here we also looked at some statistics before and after scaling the data.""")
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    # X_test_scaled = scaler.transform(X_test.astype(np.float64))
    # X_val_scaled = scaler.transform(X_val.astype(np.float64))
    # stat_sum = X_train.agg(['median', 'min', 'max', 'std'])
    # X_train_sc = pd.DataFrame(X_train_scaled)
    # stat_sum_sc = X_train_sc.agg(['median', 'min', 'max', 'std'])
    # df1stat = stat_sum.transpose()
    # df2stat = stat_sum_sc.transpose()
    
    st.write("""-----------------------""")
    st.write("""Statistics for X_train(before the scaling):""")
    st.dataframe(df1stat)
    st.write("""-----------------------""")
    st.write("""Statistics for X_train_scaled(after the scaling):""")
    st.dataframe(df2stat)
    
if nav == "MODEL DEVELOPMENT":
    st.header("MODEL DEVELOPMENT")
    st.write("""This step has been a lot of trial and error, and several times looping back to Data Preprocessing. Below bullet points shows the main steps that has been done.""")
    
    st.subheader("• Training and choosing Models")
    st.write("""In this process we've been exploring different classification models, trying to find the best predictor for our problem. 15 different models was trained on the dataset. Out of these 15 models was 5 with most potential chosen for further investigation. The code for all 15 models has been removed from the notebook since it's redundant.""") 
    st.write()
    st.write("""A summary was done over the performance of the 5 models, also looking at the confusion matrix for the different models to see the difference in predictions between the models. The conclusion is that some are better at predicting 'Dropout' and others at predicting 'Graduate'. But there are no noteworthy differences between them.""")
    
    st.subheader("• K-fold Cross Validation")
    st.write("""K-fold Cross Validation was done on the chosen five models to check for discrepancies.""")
    
    st.subheader("• Hyperparameter Tuning")
    st.write("""Hyperparameter tuning was performed on the 5 models, using the method RandomizedSearchCV. The outcome did however not show any improvements compared to the models default settings for the hyperparameters. Therefor the choice was made to use the default settings.""")
    
    st.subheader("• Balancing the dataset")
    st.write("""Here we looped back to the Data Preprocessing, since the dataset was slighlty unbalanced a test was done with balancing the training dataset with the Undersampling and the Oversampling technique in the imblearn library. The purpose was to see if the results could be improved further, the only model that improved slighlty in accuracy was the Gradient Boosting Classifier, the other models gave no improvements. The decision was made not to balance the training dataset. This code has been removed from the notebook since it's redundant.""")
    
    st.subheader("• Dropping features")
    st.write("""Here we also looped back to the Data Preprocessing, several features was dropped from the dataset to see if this could improve the prediction results. 17 features was dropped and was chosen from those that had the least correlation with the Target attribute. Training the models on the dataset with dropped features had no significane on the result at all. This code has been removed from the notebook since it's redundant.""")
    
    st.subheader("• Ensemble method")
    st.write("""Finally was different combinations of the 5 models used tested in an ensemble method, VotingClassifier. This method gave a more stable outcome on the prediction results and a more satisfactory result.""")

    st.write(df.head())

    acc_test = ENSEMB_FINAL.estimators_
    st.write(acc_test)


    
if nav == "MODEL EVALUATION":
    st.header("MODEL EVALUATION")
    st.write("""In this step we evaluate our chosen model on the untouched test dataset. We also look at some scoring and feature importance.""")
    
    st.write("""theevaluation.""")
    st.write("""theevaluation.""")
    
if nav == "PROJECT CONCLUSIONS":
    st.header("PROJECT CONCLUSIONS")
    st.write("""In this final step we did some simulation using the dropped part of the dataset, that had the Target 'Enrolled'.""")
        
    st.write("""We also tried to see if we could see some correlation between the attribute values in the Training dataset and our simulated Enrolled dataset.""")
    
    st.write("""It was hard to see any clear correlation between the feature values. However the percentage that was predicted to dropout in the Enrolled dataset is in clear correlation with the actual percentage of dropouts taken from the university statistics.""")
    
    st.write("""theconclusions.""")
    st.write("""theconclusions.""")
    