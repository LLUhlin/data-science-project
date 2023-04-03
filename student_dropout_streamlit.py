
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

from model_streamlit import *

# import data
df = pd.read_csv('/Users/nima/Downloads/PROJ25/gruppuppgift/02_dataset/student_dropout_success/data.csv', sep=';')
#df = pd.read_csv('./data.csv', sep=';') # using relative path instead to not be locked to 1 computer

nav = st.sidebar.radio("NAVIGATION BAR",["START", "DATA EXPLORATION", "DATA PREPROCESSING", "MODEL DEVELOPMENT", "MODEL EVALUATION", "PROJECT CONCLUSIONS"])



if nav == "START":
    st.title("PROJ25: Machine Learning Project of Student Graduate/Dropout dataset")
    
    st.write("""litetext.....""")
    
    image = Image.open('/Users/nima/Downloads/PROJ25/gruppuppgift/04_presentation/dropout.jpg')
    st.image(image, caption='GRADUATE OR DROPOUT?')


   
if nav == "DATA EXPLORATION":
    st.header("DATA EXPLORATION (EDA)")
    st.write("""In this section we will explore and analyze the data without touching it.""")
    
    #st.write("""The general info about the dataset.""")
    #st.table(df.info())
    
    #st.write("""The features in the dataset.""")
    #st.write(df.columns.tolist())
    
    st.write("""The first 10 rows in the dataset.""")
    st.dataframe(df.head(10))
    
    st.write("""The number of rows and columns in the dataset:""")
    df_shape = df.shape
    #df_df = pd.DataFrame({df_shape}, index=['rows', 'columns'])
    st.dataframe(df_shape)
    
    st.write("""The features in the dataset and their datatypes.""")
    st.table(df.dtypes)
    
    st.write("""Checking unique values in the Target attribute and the distribution of the unique values.""")
    st.dataframe(np.unique(df['Target'], return_counts=True))
    
    st.write("""Checking if there is any missing values in the dataset.""")
    st.table(df.isnull().sum())  


if nav == "DATA PREPROCESSING":
    st.header("DATA PREPROCESSING")
    st.write('In this section we will perform preprocessing on the dataset.')
        
    st.write("""The decision was made to drop 'Enrolled' in the Target attribute, since we are only interested in predicting if a student is going to dropout or graduate. The main purpose of the predictions is to find and help students that are at risk of dropping out..""")
    df2 = df.copy(deep=True)
    df2 = df2[df2.Target != 'Enrolled']
    st.dataframe(np.unique(df2['Target'], return_counts=True))
    
    st.write("""The number of rows and columns in the dataset after dropping 'Enrolled':""")
    st.dataframe(df2.shape)
    
    st.write("""Transforming the Target to numerical values, Dropout=0, Graduate=1.""")
    dft = df2.copy(deep=True)
    le = LabelEncoder()
    label = le.fit_transform(dft['Target'])
    dft.drop("Target", axis=1, inplace=True)
    dft["Target"] = label
    st.dataframe(np.unique(dft['Target'], return_counts=True))
    
    st.write("""Table for the correlation between the target and the features in the dataset, displaying top 15.""")
    corr_table = pd.DataFrame(dft.corr()[dft.columns[-1:]]).sort_values(by=['Target'], ascending=False).head(15)
    st.table(corr_table)
    
    st.write("""Correlation Matrix for the dataset.""")
    cmatrix = dft.corr()
    plt.figure(figsize=(20,15))
    sns.heatmap(cmatrix[((cmatrix >= .3) | (cmatrix <= -.6)) & (cmatrix !=1.000)], annot=True, cmap="Greens")
    plt.show()
    st.pyplot(plt)

    st.write("""Since all data is in one dataset, we need to split it into input variables(X) and output variables(y).""") 
    X = dft.drop(['Target'], axis=1)
    y = dft['Target']
    
    st.write("""The dataset is split into a training, test and validation dataset.""")
    X_train, X_te_va, y_train, y_te_va = train_test_split(X, y, test_size = 0.3, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_te_va, y_te_va, test_size = 0.5, stratify=y_te_va)

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
    
    st.write("""The dataset is then scaled with the Standardscaler method. This improves the performance of the machine learning models.""")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    X_test_scaled = scaler.transform(X_test.astype(np.float64))
    X_val_scaled = scaler.transform(X_val.astype(np.float64))
    
    
    
    st.write("""We looked at some statistics before and after scaling the data.""")
    
    #from IPython.display import display_html

    stat_sum = X_train.agg(['median', 'min', 'max', 'std'])
    X_train_sc = pd.DataFrame(X_train_scaled)
    stat_sum_sc = X_train_sc.agg(['median', 'min', 'max', 'std'])
    df1stat = stat_sum.transpose()
    df2stat = stat_sum_sc.transpose()

    #df1styler = df1stat.style.set_table_attributes("style='display:inline'").set_caption('Statistics X_train')
    #df2styler = df2stat.style.hide(axis='index').set_table_attributes("style='display:inline'").set_caption('Statistics X_train_scaled')
    #display_html(df1styler._repr_html_() + df2styler._repr_html_(), raw=True)
    st.dataframe(df1stat)
    st.dataframe(df2stat)
    
    #st.dataframe(df1styler)
    #st.dataframe(df2styler)
    

    
if nav == "MODEL DEVELOPMENT":
    st.title("model develop")
    st.write("""
    ##### • Training and choosing Models
    Fifteen different models was trained on the dataset. This code has been removed from the notebook since it's redundant.

    Out of these fifteen models was the five with most potential chosen for further investigation. A summary was done over the performance of the five models, also looking at the confusion matrix for the different models to see the difference in predictions between the models. The conclusion is that some are better at predicting 'Dropout' and others at predicting 'Graduate'. But there are no noteworthy differences between them.
    ##### • K-fold Cross Validation
    K-fold Cross Validation was done on the chosen five models to check for discrepancies.
    ##### • Hyperparameter Tuning
    Hyperparameter tuning was performed on these five models, but the outcome did however not show any siginificant improvements compared to the models default settings for the hyperparameters. Therefor the choice was made to use the default settings. This code has been removed from the notebook since it's redundant.
    ##### • Balancing dataset
    Since the dataset was slighlty unbalanced a test was done with balancing the training dataset with the Undersampling and the Oversampling technique in the imblearn library. The purpose was to see if the results could be improved further, the only model that improved slighlty in accuracy was the Gradient Boosting Classifier, the other models gave no improvements. The decision was made not to balance the training dataset. This code has been removed from the notebook since it's redundant.
    ##### • Dropping features
    As a final test several features was dropped from the dataset to see if this could improve the prediction results. 17 features was dropped and was chosen from those that had the least correlation with the Target attribute. Training the models on the dataset with dropped features had no significane on the result at all. This code has been removed from the notebook since it's redundant. 
    ##### • Ensemble method
    Finally was different combinations of the five models used tested in an ensemble method, VotingClassifier. This method gave a more stable outcome on the prediction result. The most satisfactory result with VotingClassifier was with Logistic Regression, Random Forest Classifier, Support Vector Classifier and Gradient Boosting Classifier combined together. This combination was chosen as the final model and will be evaulated in the next step with the untouched validation dataset.
    """)
    st.write(test.df.head())
    
    
if nav == "MODEL EVALUATION":
    st.title("model eval")
    st.write("""Read the evaluation.""")
    
if nav == "PROJECT CONCLUSIONS":
    st.title("summary")
    st.write("""Read the conclusions.""")