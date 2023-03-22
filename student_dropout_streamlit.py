
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

# import data
df = pd.read_csv('/Users/nima/Downloads/PROJ25/gruppuppgift/02_dataset/student_dropout_success/data.csv', sep=';')

# Creating a navigation bar with 5 different sections the user can choose from.

nav = st.sidebar.radio("NAVIGATION BAR",["DATA EXPLORATION", "DATA PREPROCESSING", "MODEL DEVELOPMENT", "MODEL EVALUATION", "PROJECT CONCLUSIONS"])

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
    st.write("""Read the model.""")
    
    
if nav == "MODEL EVALUATION":
    st.title("model eval")
    st.write("""Read the evaluation.""")
    
if nav == "PROJECT CONCLUSIONS":
    st.title("summary")
    st.write("""Read the conclusions.""")