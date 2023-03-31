
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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import streamlit as st

from model_streamlit import *

# import data
# df = pd.read_csv('/Users/nima/Downloads/PROJ25/gruppuppgift/02_dataset/student_dropout_success/data.csv', sep=';')
""" Changes done to use relative path instead of absolute path to ensure it works on multiple machines """
df = pd.read_csv('./data.csv', sep=';') # using relative path instead to not be locked to 1 computer


# Creating a navigation bar with 5 different sections the user can choose from.

nav = st.sidebar.radio("NAVIGATION BAR",["DATA EXPLORATION", "DATA PREPROCESSING", "MODEL DEVELOPMENT", "MODEL EVALUATION", "PROJECT CONCLUSIONS"])
test = Model()

if nav == "DATA EXPLORATION":
    st.title("DATA EXPLORATION")
    st.header("EDA")
    st.write("""The eda bla bla.""")
    
    # EDA
    st.dataframe(df)
    st.write(df.keys())
    st.write("""The shape of the dataset is:""")
    df.shape
    st.write(df.head())
    st.write(df.dtypes)
    st.dataframe(pd.isnull(df).sum())
    
    target_types = Counter(df['Target'])
    st.dataframe(df['Target'].unique())
    st.dataframe(target_types)

if nav == "DATA PREPROCESSING":
    st.title("DATA PREPROCESSING")
    st.write('In this section we will look at the data and preprocess.')
    
    # Transforming Target to numerical values, Dropout=0, Enrolled=1, Graduate=2
    dft = df.copy(deep=True)
    le = LabelEncoder()
    label = le.fit_transform(dft['Target'])
    dft.drop("Target", axis=1, inplace=True)
    dft["Target"] = label
    
    target_types = Counter(dft['Target'])
    st.dataframe(dft['Target'].unique())
    st.dataframe(target_types)
             
    dft.head()
    
    cmatrix = dft.corr()
    plt.figure(figsize=(80,60))
    sns.heatmap(cmatrix[((cmatrix >= .4) | (cmatrix <= -.6)) & (cmatrix !=1.000)], annot=True, cmap="Greens")
    plt.show()
    st.pyplot(plt)
    
    st.table(pd.DataFrame(dft.corr()[dft.columns[-1:]]).sort_values(by=['Target'], ascending=False))
    
    
    
    
    
    
    
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