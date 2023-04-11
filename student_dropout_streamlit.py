
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
from sklearn.model_selection import KFold
from itertools import combinations

from PIL import Image

# Importing models used
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import streamlit as st
st.set_page_config(layout="wide")

import joblib
import math

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

@st.cache_resource(show_spinner=True)
def correlation(dft):
    corr_table = pd.DataFrame(dft.corr()[dft.columns[-1:]]).sort_values(by=['Target'], ascending=False).head(15)
    cmatrix = dft.corr()
    fig = plt.figure(figsize=(20,8))
    g = sns.heatmap(cmatrix[((cmatrix >= .3) | (cmatrix <= -.6)) & (cmatrix !=1.000)], annot=True, cmap="Greens")
    g.set_yticklabels(g.get_yticklabels(), rotation =0)
    g.set_xticklabels(g.get_yticklabels(), rotation =90)
    return corr_table, fig

@st.cache_data
def rscv_load():
    return {
       'Logisitic Regression': load_model('Logisitic_Regression_RandomizedSearchCV.joblib'),
       'Random Forest Classifier': load_model('Random_Forest_Classifier_RandomizedSearchCV.joblib'),
       'Support Vector Classifier': load_model('Support_Vector_Classifier_RandomizedSearchCV.joblib'),
       'Gradient Boosting Classifier': load_model('Gradient_Boosting_Classifier_RandomizedSearchCV.joblib'),
       'Linear Discriminant Analysis': load_model('Linear_Discriminant_Analysis_RandomizedSearchCV.joblib')
    }

@st.cache_data
def clf_load():
    return {
       'Logisitic Regression': load_model('Logisitic_Regression.joblib'),
       'Random Forest Classifier': load_model('Random_Forest_Classifier.joblib'),
       'Support Vector Classifier': load_model('Support_Vector_Classifier.joblib'),
       'Gradient Boosting Classifier': load_model('Gradient_Boosting_Classifier.joblib'),
       'Linear Discriminant Analysis': load_model('Linear_Discriminant_Analysis.joblib')
    }

@st.cache_data
def vc_load():
    return {
        'LR, RFC, SVC, GBC, LDA, VC': load_model('LR_RFC_SVC_GBC_LDA_VotingClassifier.joblib'),
        'LR, RFC, SVC, GBC, VC': load_model('LR_RFC_SVC_GBC_VotingClassifier.joblib'),
        'LR, RFC, SVC, LDA, VC': load_model('LR_RFC_SVC_LDA_VotingClassifier.joblib'),
        'LR, RFC, GBC, LDA, VC': load_model('LR_RFC_GBC_LDA_VotingClassifier.joblib'),
        'LR, SVC, GBC, LDA, VC': load_model('LR_SVC_GBC_LDA_VotingClassifier.joblib'),
        'RFC, SVC, GBC, LDA, VC': load_model('RFC_SVC_GBC_LDA_VotingClassifier.joblib')
    }


@st.cache_data
def k_fold_cv(_CLF, X_train, X_train_scaled, y_train):
    k = 5
    
    # Define the K-fold cross-validation object
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    kf_result = pd.DataFrame(columns=['Accuracy', 'Variance', 'Bias', 'Overfitting']) 

    # Perform K-fold cross-validation on each model and compute accuracy, variance, bias, and overfitting scores
    for clf_name, clf in _CLF.items():
        accuracies = []
        variances = []
        biases = []
        overfittings = []
        for train_index, test_index in kf.split(X_train):
            X_fold_train, X_fold_test = X_train_scaled[train_index], X_train_scaled[test_index]
            y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
            clf.fit(X_fold_train, y_fold_train)
            y_pred_train = clf.predict(X_fold_train)
            y_pred_test = clf.predict(X_fold_test)
            accuracy = accuracy_score(y_fold_test, y_pred_test)
            variance = np.var(y_pred_test)
            bias = np.mean(y_pred_test) - np.mean(y_fold_test)
            overfitting = accuracy - accuracy_score(y_fold_train, y_pred_train)
            accuracies.append(accuracy)
            variances.append(variance)
            biases.append(bias)
            overfittings.append(overfitting)
        mean_accuracy = np.sum(accuracies) / k
        mean_variance = np.sum(variances) / k
        mean_bias = np.sum(biases) / k
        mean_overfitting = np.sum(overfittings) / k
        new_row = pd.DataFrame({'Accuracy': mean_accuracy, 'Variance': mean_variance, 'Bias': mean_bias, 'Overfitting': mean_overfitting},
                               index=[clf_name])
        kf_result = pd.concat([kf_result, new_row], axis=0)
    return kf_result


@st.cache_resource
def summary_clf(_CLF, X_val_scaled, X_train_scaled, y_train, df2):
    f, axes = plt.subplots(1, 5, figsize=(20, 5), sharey='row')
    target_list = df2['Target'].unique()
    result = pd.DataFrame(columns = ['Train Accuracy','Validate Accuracy','Precision', 'Recall']) 

    for i, (mdl, clf) in enumerate(_CLF.items()):
        y_pred_val = clf.fit(X_train_scaled, y_train).predict(X_val_scaled)

        y_pred_train = clf.predict(X_train_scaled)
        acctrain = round(accuracy_score(y_train, y_pred_train), 4)
        accval = round(accuracy_score(y_val, y_pred_val), 4)
        prec = round(precision_score(y_val, y_pred_val, average='weighted'), 4)
        rec = round(recall_score(y_val, y_pred_val, average='weighted'), 4)
        new_row = pd.DataFrame({'Train Accuracy':acctrain, 'Validate Accuracy':accval, 'Precision':prec, 'Recall':rec},
                            index=[mdl])
        result = pd.concat([result, new_row], axis=0)
        cm = ConfusionMatrixDisplay.from_estimator(clf,
                                                X_val_scaled,
                                                y_val,
                                                display_labels=target_list,
                                                ax=axes[i],
                                                colorbar=False)
        cm.plot(ax=axes[i], xticks_rotation=45, cmap=plt.cm.Blues,values_format='g')
        cm.ax_.set_title(mdl)
        cm.im_.colorbar.remove()
        cm.ax_.set_xlabel('')
        if i!=0:
            cm.ax_.set_ylabel('')

    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    f.colorbar(cm.im_, ax=axes)

    return f, result


@st.cache_data
def tuning_hypterparameters(_CLF):
    res_hyp_par = pd.DataFrame(columns = ['Accuracy', 'Optimized hyperparameters']) 

    for i, (mdl, clf) in enumerate(_CLF.items()): 
        acc_hp = clf.best_score_ 
        opt_hp = str(clf.best_params_) 
        new_row = pd.DataFrame({'Accuracy':f'{round(acc_hp * 100, 2)}%', 'Optimized hyperparameters':opt_hp}, index=[mdl]) 
        res_hyp_par = pd.concat([res_hyp_par, new_row], axis=0) 

    return res_hyp_par

@st.cache_data
def ensemble_all_models(_CLF, X_train_scaled, y_train, X_val_scaled, y_val, option):
    summa = pd.DataFrame(columns = ['Train Accuracy', f'{option} Accuracy']) 

    for comb in _CLF:
        model = _CLF[comb]
        model.fit(X_train_scaled, y_train)
        acc_tr = model.score(X_train_scaled, y_train)
        acc_te = model.score(X_val_scaled, y_val)
        new_row = pd.DataFrame({'Train Accuracy':acc_tr, f'{option} Accuracy':acc_te}, index=[comb[:-4]])
        summa = pd.concat([summa, new_row], axis=0)

    return summa


@st.cache_resource
def roc_comparison(_CLF, X_train_scaled, y_train, X_test_scaled, y_test):
    temp_clf = {key: value for key, value in _CLF.items() if str(value) != 'LinearDiscriminantAnalysis()'}

    f, axes = plt.subplots(1, len(temp_clf), figsize=(20, 5), sharey='row')

    for i, (mdl, clf) in enumerate(_CLF.items()):
        if (str(clf) != 'LinearDiscriminantAnalysis()'):
            est = clf.fit(X_train_scaled, y_train)
            rc = RocCurveDisplay.from_estimator(est, X_test_scaled, y_test, ax=axes[i])
            rc.plot(ax=axes[i])
            rc.ax_.set_title(mdl)
    
    return f

@st.cache_data
def add_to_axis(ax, ax2, X_train, X_train_scaled, y_train, _CLF, labelsize, titlesize, limit):
    for i, value in enumerate(_CLF):
        if str(_CLF[value]) == str(SVC()):
            classifier = SVC(kernel="linear")
        else:
            classifier = _CLF[value]
            
        classifier.fit(X_train_scaled, y_train)
        try:
            imp = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.coef_[0]})).sort_values(by='im')
            imp2 = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.coef_[0]})).sort_values(by='im').tail(limit)
        except AttributeError:
            imp = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.feature_importances_})).sort_values(by='im')
            imp2 = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.feature_importances_})).sort_values(by='im').tail(limit)

        ax2[math.floor(i/2), i%2].barh(imp2['attr'], imp2['im'], align='center', color=(0.2, 0.6, 0.4, 0.4))
        ax2[math.floor(i/2), i%2].set_title(f'{value}', size=titlesize)
        ax2[math.floor(i/2), i%2].tick_params(axis='both', which='major', labelsize=labelsize)

        ax[math.floor(i/2), i%2].barh(imp['attr'], imp['im'], align='center', color=(0.2, 0.6, 0.4, 0.4))
        ax[math.floor(i/2), i%2].set_title(f'{value}', size=titlesize)
        ax[math.floor(i/2), i%2].tick_params(axis='both', which='major', labelsize=labelsize)

@st.cache_resource
def models_important_features(X_train, X_train_scaled, y_train, _CLF, width=10, height=5, labelsize=7, titlesize=9, limit=15):
    clf = {key: value for key, value in _CLF.items() if key != "Linear Discriminant Analysis"}
    fig1, ax1 = plt.subplots(math.ceil(len(clf.keys())/2), 2, figsize=(width, math.ceil(len(clf.keys())/2)*height), tight_layout = True) 
    fig2, ax2 = plt.subplots(math.ceil(len(clf.keys())/2), 2, figsize=(width, math.ceil(len(clf.keys())/2)*height), tight_layout = True) 
    add_to_axis(ax1, ax2, X_train, X_train_scaled, y_train, clf, labelsize, titlesize, limit)


    fig1.suptitle("\n\n".join(['', "Important Features per classifier", '']), y=0.98, fontsize=titlesize*2 )
    fig2.suptitle("\n\n".join(['', f"Top {limit} important Features per classifier", '']), y=0.98, fontsize=titlesize*2)

    return fig1, fig2


@st.cache_data
def enrolled_test(df):
    # Creating dataset with ONLY 'Enrolled'
    df3 = df.copy(deep=True)
    df3 = df3[df3.Target == 'Enrolled']


    # Creating X_enr, dataset without Target + scaling
    X_enr = df3.drop(['Target'], axis=1)
    X_enr_scaled = StandardScaler().fit_transform(X_enr.astype(np.float64))


    return df3, X_enr, X_enr_scaled

@st.cache_data
def enrolled_predict(_ensemb_final, X_train_scaled, y_train, X_enr_scaled):
    _ensemb_final.fit(X_train_scaled, y_train)
    yt = _ensemb_final.predict(X_train_scaled)
    ye = _ensemb_final.predict(X_enr_scaled)

    return yt, ye

@st.cache_resource
def enrolled_scatter_graph(X_train, X_enr, yt, ye):
    xt_2sem_app = X_train['Curricular units 2nd sem (approved)']
    xt_1sem_app = X_train['Curricular units 1st sem (approved)']
    xt_2sem_gra = X_train['Curricular units 2nd sem (grade)']
    xt_1sem_gra = X_train['Curricular units 1st sem (grade)']
    #xt_tui_fees = X_train['Tuition fees up to date']
    xt_2sem_enr = X_train['Curricular units 2nd sem (enrolled)']
    xt_1sem_enr = X_train['Curricular units 1st sem (enrolled)']

    xe_2sem_app = X_enr['Curricular units 2nd sem (approved)']
    xe_1sem_app = X_enr['Curricular units 1st sem (approved)']
    xe_2sem_gra = X_enr['Curricular units 2nd sem (grade)']
    xe_1sem_gra = X_enr['Curricular units 1st sem (grade)']
    #xe_tui_fees = X_enr['Tuition fees up to date']
    xe_2sem_enr = X_enr['Curricular units 2nd sem (enrolled)']
    xe_1sem_enr = X_enr['Curricular units 1st sem (enrolled)']

    fig, ax = plt.subplots(3, 2, figsize=(10,9))

    ax[0, 0].scatter(yt, xt_2sem_app, s=10, c='b', marker="s", label='Training dataset')
    ax[0, 0].scatter(ye, xe_2sem_app, s=10, c='r', marker="o", label='Enrolled dataset')
    ax[0, 0].set_title('Curricular units 2nd sem (approved)', size=9)
    ax[0, 0].legend(loc='upper center', fontsize=8)

    ax[0, 1].scatter(yt, xt_1sem_app, s=10, c='b', marker="s", label='Training dataset')
    ax[0, 1].scatter(ye, xe_1sem_app, s=10, c='r', marker="o", label='Enrolled dataset')
    ax[0, 1].set_title('Curricular units 1st sem (approved)', size=9)
    ax[0, 1].legend(loc='upper center', fontsize=8)

    ax[1, 0].scatter(yt, xt_2sem_gra, s=10, c='b', marker="s", label='Training dataset')
    ax[1, 0].scatter(ye, xe_2sem_gra, s=10, c='r', marker="o", label='Enrolled dataset')
    ax[1, 0].set_title('Curricular units 2nd sem (grade)', size=9)
    ax[1, 0].legend(loc='upper center', fontsize=8)

    ax[1, 1].scatter(yt, xt_1sem_gra, s=10, c='b', marker="s", label='Training dataset')
    ax[1, 1].scatter(ye, xe_1sem_gra, s=10, c='r', marker="o", label='Enrolled dataset')
    ax[1, 1].set_title('Curricular units 1st sem (grade)', size=9)
    ax[1, 1].legend(loc='upper center', fontsize=8)

    ax[2, 0].scatter(yt, xt_2sem_enr, s=10, c='b', marker="s", label='Training dataset')
    ax[2, 0].scatter(ye, xe_2sem_enr, s=10, c='r', marker="o", label='Enrolled dataset')
    ax[2, 0].set_title('Curricular units 2nd sem (enrolled)', size=9)
    ax[2, 0].legend(loc='upper center', fontsize=8)

    ax[2, 1].scatter(yt, xt_1sem_enr, s=10, c='b', marker="s", label='Training dataset')
    ax[2, 1].scatter(ye, xe_1sem_enr, s=10, c='r', marker="o", label='Enrolled dataset')
    ax[2, 1].set_title('Curricular units 1st sem (enrolled)', size=9)
    ax[2, 1].legend(loc='upper center', fontsize=8)
    
    return fig


@st.cache_data
def enrolled_count_percentage(ye):
    counter = Counter(ye)
    total_count = sum(counter.values())

    df = pd.DataFrame(columns = ['Count','Percentage']) 
    for label, count in counter.items():
        new_row = pd.DataFrame({'Count': count, 'Percentage': str(round((count / total_count) * 100, 2)) + '%'}, index=['Graduate' if label == 1 else 'Dropout']) 
        df = pd.concat([df, new_row], axis=0)
    
    return df

""" ------------------------------------------------------------------------------------------------------------------------- """

nav = st.sidebar.radio("NAVIGATION BAR",["START", "DATA EXPLORATION", "DATA PREPROCESSING", "MODEL DEVELOPMENT", "MODEL EVALUATION", "PROJECT CONCLUSIONS"])

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
X_train_sc = pd.DataFrame(X_train_scaled, columns=X_train.columns)
stat_sum_sc = X_train_sc.agg(['median', 'min', 'max', 'std'])
df1stat = stat_sum.transpose()
df2stat = stat_sum_sc.transpose()

## Load models
ENSEMB_FINAL = load_model('ensemb_final.joblib')
RSCV_CLF = rscv_load()
VC_CLF = vc_load()
CLF = clf_load()

# create cached tables / plots
kf_result = k_fold_cv(CLF, X_train, X_train_scaled, y_train)
plt_clf, sum_clf = summary_clf(CLF, X_val_scaled, X_train_scaled, y_train, df2)
hyper_tuning = tuning_hypterparameters(RSCV_CLF)
ensemble_models = ensemble_all_models(VC_CLF, X_train_scaled, y_train, X_val_scaled, y_val, 'Validate')

ensemble_test_models = ensemble_all_models(VC_CLF, X_train_scaled, y_train, X_test_scaled, y_test , 'Test').loc['LR, RFC, SVC, GBC', :]
roc_score = roc_comparison(CLF, X_train_scaled, y_train, X_test_scaled, y_test)
all_features, top_features = models_important_features(X_train, X_train_scaled, y_train, CLF, limit=10)

df3, X_enr, X_enr_scaled = enrolled_test(df)
yt, ye = enrolled_predict(ENSEMB_FINAL, X_train_scaled, y_train, X_enr_scaled)
enrolled_test_fig = enrolled_scatter_graph(X_train, X_enr, yt, ye)
enr_count_perc = enrolled_count_percentage(ye)

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
    st.dataframe(pd.DataFrame(np.array(df.shape).transpose(), index=['Rows', 'Columns'], columns=['Count']))
    
    st.write("""Showing all the features in the dataset and their datatypes.""")
    st.dataframe(pd.DataFrame(df.dtypes, columns=['Datatype']), height=1333, width=800)
    
    st.write("""Checking the unique values in the Target attribute and also the distribution of the unique values.""")
    st.dataframe(pd.DataFrame(np.unique(df['Target'], return_counts=True)[1], index=np.unique(df['Target'], return_counts=True)[0], columns=['Count']), width=200)
    st.write("""Checking if there is any missing values in the dataset.""")
    st.dataframe(pd.DataFrame(map(lambda x: str(x), df.isnull().sum()), columns=['Count'], index=df.isnull().sum().keys()).style.set_properties(**{'text-align': 'left'}), height=1333, width=800)  

if nav == "DATA PREPROCESSING":
    st.header("DATA PREPROCESSING")
    st.write('In this step we will start to perform preprocessing on the dataset.')
        
    st.write("""The decision was made to drop the rows which had 'Enrolled' in the Target attribute, since we are only interested in predicting if a student is going to dropout or graduate.""")
    st.write("""It should be added that an initially 'Enrolled' was not dropped just to see how different models performed on the complete dataset. The result on the 15 models we trained was between 67 and 78 % in accuracy.""")
    st.write("""Further note that 'Enrolled' was by far the smallest part(18%), which made the dataset highly imbalanced.""")    
    
    st.write("""Again checking the unique values in the Target attribute after dropping 'Enrolled'.""")
    st.dataframe(pd.DataFrame(np.unique(df2['Target'], return_counts=True)[1], index=np.unique(df2['Target'], return_counts=True)[0], columns=['Count']), width=200)
    
    st.write("""The number of rows and columns in the dataset after dropping 'Enrolled':""")
    st.dataframe(pd.DataFrame(np.array(df2.shape).transpose(), index=['Rows', 'Columns'], columns=['Count']))
    
    st.write("""Since the Target's datatype was object, it was transformed to numerical using the method LabelEncoder.""")
    st.write("""The result after transforming the Target to numerical values, Dropout=0, Graduate=1.""")

    st.dataframe(pd.DataFrame(np.unique(dft['Target'], return_counts=True)[1], index=np.unique(dft['Target'], return_counts=True)[0], columns=['Count']), width=200)
    
    st.write("""Table for the correlation between the target and the features in the dataset, displaying the top 15.""")
    corr_table, cmatrix = correlation(dft)

    st.dataframe(pd.DataFrame(corr_table, columns=['Target']), height=563, width=800)
    
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
    
    st.pyplot(plt_clf)
    st.dataframe(sum_clf.style.format('{:.2%}'))

    st.subheader("• K-fold Cross Validation")
    st.write("""K-fold Cross Validation was done on the chosen five models to check for discrepancies.""")
    
    st.dataframe(kf_result.style.format('{:.2%}'))

    st.subheader("• Hyperparameter Tuning")
    st.write("""Hyperparameter tuning was performed on the 5 models, using the method RandomizedSearchCV. The outcome did however not show any improvements compared to the models default settings for the hyperparameters. Therefor the choice was made to use the default settings.""")
    
    st.dataframe(hyper_tuning, width=1500)
    
    st.subheader("• Balancing the dataset")
    st.write("""Here we looped back to the Data Preprocessing, since the dataset was slighlty unbalanced a test was done with balancing the training dataset with the Undersampling and the Oversampling technique in the imblearn library. The purpose was to see if the results could be improved further, the only model that improved slighlty in accuracy was the Gradient Boosting Classifier, the other models gave no improvements. The decision was made not to balance the training dataset.""")
    
    st.subheader("• Dropping features")
    st.write("""Here we also looped back to the Data Preprocessing, several features was dropped from the dataset to see if this could improve the prediction results. 17 features was dropped and was chosen from those that had the least correlation with the Target attribute. Training the models on the dataset with dropped features had no significane on the result at all.""")
    
    st.subheader("• Ensemble method")
    st.write("""Finally was different combinations of the 5 models used tested in an ensemble method, VotingClassifier. This method gave a more stable outcome on the prediction results and a more satisfactory result.""")

    st.dataframe(ensemble_models.style.format("{:.2%}"), width=500)


    # res_hyp_par = pd.DataFrame(columns = ['Accuracy', 'Optimized hyperparameters'])

    # for i, (mdl, clf) in enumerate(RSCV_CLF.items()):

    #     acc_hp = clf.best_score_
    #     opt_hp = str(clf.best_params_)
    #     new_row = pd.DataFrame({'Accuracy':acc_hp, 'Optimized hyperparameters':opt_hp},
    #                         index=[mdl])
    #     res_hyp_par = pd.concat([res_hyp_par, new_row], axis=0)
    # st.write(res_hyp_par)


if nav == "MODEL EVALUATION":
    st.header("MODEL EVALUATION")
    st.write("""In this step we evaluate our chosen model on the untouched test dataset. We also look at some scoring and feature importance.""")
    st.dataframe(pd.DataFrame(ensemble_test_models).transpose().style.format("{:.2%}"))
    st.write("""Receiver Operating Characteristic""")
    st.pyplot(roc_score)
    st.write("""All important features.""")
    st.pyplot(all_features)
    st.write("""Top 10 important features.""")
    st.pyplot(top_features)



    
if nav == "PROJECT CONCLUSIONS":
    st.header("PROJECT CONCLUSIONS")
    st.write("""In this final step we did some simulation using the dropped part of the dataset, that had the Target 'Enrolled'.""")
    st.write('Data')
    st.dataframe(pd.DataFrame(np.unique(df3['Target'], return_counts=True)[1], index=np.unique(df3['Target'], return_counts=True)[0], columns=['Count']), width=200)
    st.write('Shape')
    st.dataframe(pd.DataFrame(np.array(df3.shape).transpose(), index=['Rows', 'Columns'], columns=['Count']))
    st.write("""We also tried to see if we could see some correlation between the attribute values in the Training dataset and our simulated Enrolled dataset.""")
    st.pyplot(enrolled_test_fig)
    
    st.write("""It was hard to see any clear correlation between the feature values. However the percentage that was predicted to dropout in the Enrolled dataset is in clear correlation with the actual percentage of dropouts taken from the university statistics.""")
    
    st.write("""Prediction from final model""")
    st.dataframe(enr_count_perc)
    
    st.write('Data gathered from Polytechnic Institute of Portalegre between 2018 and 2020')
    tt = np.array(['61.3%', '46.24%'])
    st.dataframe(pd.DataFrame(tt,  index=["Bachelor", "Master"], columns=['Graduation Rate']))
    # st.write('Bachelor graduation rate: 61.3%')
    # st.write('Masters graduation rate: 46.24%')
    st.write('Assuming we have a student distribution of 80% Bachelor and 20% Master we would get an average graduate rate of 58.44% with our sample size of 794 students')
    # difference = round(abs(float(enr_count_perc['Percentage']['Graduate'][:-1]) - 58.44), 2)

    m_g = np.array(['52.02%', '58.44%'])
    st.dataframe(pd.DataFrame(m_g, index=['Model', 'Calculated'], columns=['Graduation Rate']))
    st.write('Which tells us that our model is underestimating the students but also shows us that the model can be effectively used to predict who is at risk of dropping out.')
    