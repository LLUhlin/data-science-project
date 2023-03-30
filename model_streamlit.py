# Importing models used
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold
from IPython.display import display_html

# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from collections import Counter

from sklearn.metrics import * 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import VotingClassifier
from itertools import combinations


class Model():
    def __init__(self):
        super().__init__()

        # df = pd.read_csv('/Users/nima/Downloads/PROJ25/gruppuppgift/02_dataset/student_dropout_success/data.csv', sep=';')
        self.df = pd.read_csv('./data.csv', sep=';') # Change path to be relative and not dependent on machine
        pd.set_option('display.max_colwidth', None)

        # Checking unique values in the Target attribute and the distribution of the unique values
        self.target_types = Counter(self.df['Target'])
        # Creating dataset without 'Enrolled'
        self.df2 = self.df.copy(deep=True)
        self.df2 = self.df2[self.df2.Target != 'Enrolled']
        # Transforming Target to numerical values, Dropout=0, Graduate=1
        self.dft = self.df2.copy(deep=True)
        self.le = LabelEncoder()
        self.label = self.le.fit_transform(self.dft['Target'])
        self.dft.drop("Target", axis=1, inplace=True)
        self.dft["Target"] = self.label

        # Correlation Matrix over the dataset
        self.cmatrix = self.dft.corr()
        # plt.figure(figsize=(20,15))
        # sns.heatmap(cmatrix[((cmatrix >= .3) | (cmatrix <= -.6)) & (cmatrix !=1.000)], annot=True, cmap="Greens")


        # Creating X, dataset without labels
        self.X = self.dft.drop(['Target'], axis=1)
        # Creating y, the label dataset
        self.y = self.dft['Target']
        # Splitting the data into a training, test and validation dataset
        self.X_train, self.X_te_va, self.y_train, self.y_te_va = train_test_split(self.X, self.y, test_size = 0.3, stratify=self.y)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(self.X_te_va, self.y_te_va, test_size = 0.5, stratify=self.y_te_va)

        # Scaling the data with StandardScaler
        self.X_train_scaled = StandardScaler().fit_transform(self.X_train.astype(np.float64))
        self.X_test_scaled = StandardScaler().fit_transform(self.X_test.astype(np.float64))
        self.X_val_scaled = StandardScaler().fit_transform(self.X_val.astype(np.float64))


        # Train models
        # Training the models and displaying the result
        # Add any classifier that you want to train your model with into below dictionary.
        # Cross-Validation not included for these variables

        self.clf_dict = {
            'Logisitic Regression': LogisticRegression(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Support Vector Classifier': SVC(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
        }

        self.estimators = list(zip(list(map(lambda x: ''.join([c for c in x if c.isupper()]).lower(), list(self.clf_dict.keys()))), list(self.clf_dict.values())))
        self.estimator_order = ''.join(list(map(lambda x: f'{x.upper()}, ', list(zip(*self.estimators[:-1]))[0]))) + self.estimators[-1][0].upper()

        self.lr = LogisticRegression()
        self.rf = RandomForestClassifier()
        self.svc = SVC()
        self.gbc = GradientBoostingClassifier()
        self.lda = LinearDiscriminantAnalysis()
        self.mdl_list = [('lr', self.lr), ('rf', self.rf), ('svc', self.svc), ('gbc', self.gbc), ('lda', self.lda)]
        self.mdl_list2 = ('LR', 'RF', 'SVC', 'GBC', 'LDA')
        self.comb_list = list(combinations(self.mdl_list, 4))
        self.comb_list2 = list(combinations(self.mdl_list2, 4))
        self.comb_list.insert(0, self.mdl_list)
        self.comb_list2.insert(0, self.mdl_list2) 



        # OBS Ska läggas in i koden
        # Run the final model on the part of the dataset that has Target 'Enrolled'

        # Creating dataset with ONLY 'Enrolled'
        self.df3 = self.df.copy(deep=True)
        self.df3 = self.df3[self.df3.Target == 'Enrolled']

        # Creating X_enr, dataset without Target + scaling
        self.X_enr =self.df3.drop(['Target'], axis=1)
        self.X_enr_scaled = StandardScaler().fit_transform(self.X_enr.astype(np.float64))



    def train_models_multiple_clf(self):
        clf = self.clf_dict
        f, axes = plt.subplots(1, 5, figsize=(20, 5), sharey='row')
        target_list = self.df2['Target'].unique()
        result = pd.DataFrame(columns = ['Train Accuracy','Validate Accuracy','Precision', 'Recall']) 

        for i, (mdl, clf) in enumerate(clf.items()):
            y_pred_val = clf.fit(self.X_train_scaled, self.y_train).predict(self.X_val_scaled)
            y_pred_train = clf.predict(self.X_train_scaled)
            acctrain = round(accuracy_score(self.y_train, y_pred_train), 4)
            accval = round(accuracy_score(self.y_val, y_pred_val), 4)
            prec = round(precision_score(self.y_val, y_pred_val, average='weighted'), 4)
            rec = round(recall_score(self.y_val, y_pred_val, average='weighted'), 4)
            new_row = pd.DataFrame({'Train Accuracy':acctrain, 'Validate Accuracy':accval, 'Precision':prec, 'Recall':rec},
                                index=[mdl])
            result = pd.concat([result, new_row], axis=0)
            cm = ConfusionMatrixDisplay.from_estimator(clf,
                                                    self.X_val_scaled,
                                                    self.y_val,
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
        plt.show()

        #result.style.background_gradient(cmap=sns.light_palette("blue", as_cmap=True))
        # display(result.style.format('{:.2%}'))


    def tuning_with_rscv():
        clf = {'Logistic Regression': LogisticRegression(),
        'Random Forest Classifier': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'Gradient Boosting Classifier': GradientBoostingClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()}

        params_lr = {'C':[0.001, 0.01, 0.1, 1, 10, 100],
                    'solver':['lbfgs', 'saga', 'sag', 'liblinear'],
                    'max_iter':[1000]}

        params_rf = {'bootstrap': [True, False],
                    'max_depth': [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    'max_features': ['auto', 'sqrt', 'log2', None],
                    'min_samples_leaf': [1, 2, 4, 10, 30, 60],
                    'min_samples_split': [2, 5, 10],
                    'n_estimators': [10, 50, 100, 200, 500, 1000, 1500, 2000]}

        params_svc = {'kernel':['linear', 'poly', 'rbf'],
                    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'gamma':['scale', 'auto', 0.1, 1, 10, 100],
                    'coef0':[0, 1]}


        params_gbc = {'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500],
                    'max_leaf_nodes': [2, 5, 10, 20, 50, 100],
                    'learning_rate': (0.01, 1)}

        params_lda = {'solver': ['lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.2, 0.3, 0.4, 0.5]}
        # lda has no hyperparameters that you can set
        # found hyperparameters!

        hyp_par_list = (params_lr, params_rf, params_svc, params_gbc, params_lda)

        res_hyp_par = pd.DataFrame(columns = ['Accuracy', 'Optimized hyperparameters'])

        for i, (mdl, clf) in enumerate(clf.items()):
            racv = RandomizedSearchCV(clf, hyp_par_list[i], n_jobs=-1, cv=5)
            racv.fit(self.X_train_scaled, self.y_train)
            acc_hp = racv.best_score_
            opt_hp = str(racv.best_params_)
            new_row = pd.DataFrame({'Accuracy':acc_hp, 'Optimized hyperparameters':opt_hp},
                                index=[mdl])
            res_hyp_par = pd.concat([res_hyp_par, new_row], axis=0)

    def cross_val_clf(self):
        # OBS!!! köra Cross val predict, som bygger på den data som den inte sett
        # Cross validation through cross_val_predict
        temp_clf = {'Logistic Regression': LogisticRegression(),
        'Random Forest Classifier': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'Gradient Boosting Classifier': GradientBoostingClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()}

        res_hyp_par = pd.DataFrame(columns = ['Accuracy', 'Optimized hyperparameters'])

        for i, (mdl, clf) in enumerate(temp_clf.items()):
            y_pred_cv = cross_val_predict(clf, self.X_train_scaled, self.y_train, n_jobs=-1, cv=5)
            print(mdl, accuracy_score(self.y_train, y_pred_cv))

        
    def ensemble_method_vc(self):


        summa = pd.DataFrame(columns = ['Train Accuracy','Validate Accuracy']) 

        for rakna, clfs in enumerate(self.comb_list):
            mdls = self.comb_list2[rakna]
            ensemble = VotingClassifier(estimators=clfs, voting='hard', n_jobs=-1)
            ensemble.fit(self.X_train_scaled, self.y_train)
            acc_tr = ensemble.score(self.X_train_scaled, self.y_train)
            acc_te = ensemble.score(self.X_val_scaled, self.y_val)
            new_row = pd.DataFrame({'Train Accuracy':acc_tr, 'Validate Accuracy':acc_te}, index=[mdls])
            summa = pd.concat([summa, new_row], axis=0)

    def k_fold_cv(self):
        # Define the number of folds
        k = 5

        # Define the K-fold cross-validation object
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # Perform K-fold cross-validation on each model and compute accuracy scores
        for clf_name, clf in self.clf_dict.items():
            accuracies = []
            for train_index, test_index in kf.split(self.X_train):
                X_fold_train, X_fold_test = self.X_train_scaled[train_index], self.X_train_scaled[test_index]
                y_fold_train, y_fold_test = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
                clf.fit(X_fold_train, y_fold_train)
                y_pred = clf.predict(X_fold_test)
                accuracy = accuracy_score(y_fold_test, y_pred)
                accuracies.append(accuracy)
            mean_accuracy = np.sum(accuracies) / k
            print(clf_name, 'accuracy:', mean_accuracy)



    def ensemble_train_val(self):
        ensemble = VotingClassifier(estimators=self.estimators,
                                voting='hard',
                                n_jobs=-1)

        # Fit model to the training dataset
        ensemble.fit(self.X_train_scaled, self.y_train)

        # Test our model on the test dataset
        acc_tr = ensemble.score(self.X_train_scaled, self.y_train)
        acc_te = ensemble.score(self.X_val_scaled, self.y_val)

        summary = pd.DataFrame({'Train Accuracy':[acc_tr], 'Validate Accuracy':[acc_te]},
                        index=[f'Voting Classifier ({self.estimator_order})'])
        

    def test_ontouched_dataset(self):
        # Testing our model on the untouched test dataset
        # X_test_scaled = StandardScaler().fit_transform(self.X_test.astype(np.float64))

        from sklearn.ensemble import VotingClassifier

        #create our voting classifier
        ensemb_final = VotingClassifier(estimators=self.comb_list[1],
                                    voting='hard',
                                    n_jobs=-1)

        #fit model to training data
        ensemb_final.fit(self.X_train_scaled, self.y_train)

        #test our model on the validation data
        acc_test = ensemb_final.score(self.X_test_scaled, self.y_test)
        summary = pd.DataFrame({'Test Accuracy':[acc_test]},
                        index=[self.comb_list2[1]])



    def roc_curve_comp(self):
        # OBS NY KOD SOM SKA LÄGGAS IN!!!!!

        # ROC Curve comparison
        clf = {'Logistic Regression': LogisticRegression(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Support Vector Classifier': SVC(),
            'Gradient Boosting Classifier': GradientBoostingClassifier()}

        # lda has no ROC curve

        f, axes = plt.subplots(1, 4, figsize=(20, 5), sharey='row')

        for i, (mdl, clf) in enumerate(clf.items()):
            est = clf.fit(self.X_train_scaled, self.y_train)
            rc = RocCurveDisplay.from_estimator(est, self.X_test_scaled, self.y_test, ax=axes[i])
            rc.plot(ax=axes[i])
            rc.ax_.set_title(mdl)

        plt.show()


    def add_to_axis(self, ax, ax2, X_train, X_train_scaled, y_train, clf, labelsize, titlesize, limit):
        for i, value in enumerate(clf):
            if str(clf[value]) == str(SVC()):
                classifier = SVC(kernel="linear")
            else:
                classifier = clf[value]
                
            classifier.fit(X_train_scaled, y_train)
            try:
                imp = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.coef_[0]})).sort_values(by='im')
                imp2 = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.coef_[0]})).sort_values(by='im').tail(limit)
            except AttributeError:
                imp = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.feature_importances_})).sort_values(by='im')
                imp2 = (pd.DataFrame(data={'attr': X_train.columns,'im': classifier.feature_importances_})).sort_values(by='im').tail(limit)

            ax2[math.floor(i/2), i%2].barh(imp2['attr'], imp2['im'], align='center', color=(0.2, 0.6, 0.4, 0.4))
            ax2[math.floor(i/2), i%2].set_title(f'{value}, feature importance', size=titlesize)
            ax2[math.floor(i/2), i%2].tick_params(axis='both', which='major', labelsize=labelsize)

            ax[math.floor(i/2), i%2].barh(imp['attr'], imp['im'], align='center', color=(0.2, 0.6, 0.4, 0.4))
            ax[math.floor(i/2), i%2].set_title(f'{value}, feature importance', size=titlesize)
            ax[math.floor(i/2), i%2].tick_params(axis='both', which='major', labelsize=labelsize)


    def models_important_features(self, X_train, X_train_scaled, y_train, clf, width=10, height=8, labelsize=7, titlesize=9, limit=15):

        fig1, ax1 = plt.subplots(math.ceil(len(clf.keys())/2), 2, figsize=(width,math.ceil(len(clf.keys())/2)*height), tight_layout = True) 
        fig2, ax2 = plt.subplots(math.ceil(len(clf.keys())/2), 2, figsize=(width,math.ceil(len(clf.keys())/2)*height), tight_layout = True) 

        self.add_to_axis(ax1, ax2, X_train, X_train_scaled, y_train, clf, labelsize, titlesize, limit)


        fig1.suptitle("\n\n\n".join(["Important Features per classifier"]) + '\n\n\n', y=0.98, fontsize=titlesize*2 )
        fig2.suptitle("\n\n\n".join([f"Top {limit} important Features per classifier"]) + '\n\n\n', y=0.98, fontsize=titlesize*2)

        plt.show()

    def return_model_important_features(self):
        self.models_important_features(self.X_train, self.X_train_scaled, self.y_train, self.clf_dict, width=30, titlesize=14, labelsize=12, limit=10)



    def graph_features_enrolled(self):
        # WORK IN PROGRESS
        # Göra lite grafer över nedanstående features, jämföra skillnader i värden mellan res_train och res_enrolled

        yt = self.ensemb_final.predict(self.X_train_scaled)
        ye = self.ensemb_final.predict(self.X_enr_scaled)

        xt_2sem_app = self.X_train['Curricular units 2nd sem (approved)']
        xt_1sem_app = self.X_train['Curricular units 1st sem (approved)']
        xt_2sem_gra = self.X_train['Curricular units 2nd sem (grade)']
        xt_1sem_gra = self.X_train['Curricular units 1st sem (grade)']
        #xt_tui_fees = X_train['Tuition fees up to date']
        xt_2sem_enr = self.X_train['Curricular units 2nd sem (enrolled)']
        xt_1sem_enr = self.X_train['Curricular units 1st sem (enrolled)']

        xe_2sem_app = self.X_enr['Curricular units 2nd sem (approved)']
        xe_1sem_app = self.X_enr['Curricular units 1st sem (approved)']
        xe_2sem_gra = self.X_enr['Curricular units 2nd sem (grade)']
        xe_1sem_gra = self.X_enr['Curricular units 1st sem (grade)']
        #xe_tui_fees = X_enr['Tuition fees up to date']
        xe_2sem_enr = self.X_enr['Curricular units 2nd sem (enrolled)']
        xe_1sem_enr = self.X_enr['Curricular units 1st sem (enrolled)']

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

        plt.show()

        stat_sum_tra = xt_2sem_app.agg(['median', 'mean', 'min', 'max', 'std'])
        stat_sum_enr = xe_2sem_app.agg(['median', 'mean', 'min', 'max', 'std'])

        stat_sum = self.X_train.agg(['median', 'min', 'max', 'std'])
        stat_sum_sc = self.X_enr.agg(['median', 'min', 'max', 'std'])
        df1stat = stat_sum.transpose()
        df2stat = stat_sum_sc.transpose()

        df1styler = df1stat.style.set_table_attributes("style='display:inline'").set_caption('Statistics X_train')
        df2styler = df2stat.style.hide(axis='index').set_table_attributes("style='display:inline'").set_caption('Statistics X_enrolled')
        display_html(df1styler._repr_html_() + df2styler._repr_html_(), raw=True)