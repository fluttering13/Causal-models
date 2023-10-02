import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import optuna
'''
Content
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The data set includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents
Inspiration
To explore this type of models and learn more about the subject.
'''

path='./Customer-Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
f_path='./Customer-Churn/model_save/'
data_pd=pd.read_csv(path)

### list the columns
columns=data_pd.columns


##Find the missing part of the data
# for name in columns:
#     cond1=data_pd.isnull()[name]==True
#     print(data_pd[cond1])
###或者是這樣
# print(data_pd.isnull().sum())

###列表uniuqe的元素
# print(data_pd.nunique())
# for name in data_pd.columns:
#     print(pd.unique(data_pd[name]))

### categorical data
# for col in columns:
#     print(data_pd[col].value_counts())
    
###data type transfom and numericalize
# data_pd_numberical=data_pd.replace({'Yes':1,'No':0,'No internet service':2,'No phone service':2,'DSL':1,'Fiber optic':2,'Month-to-month':0,'One year':1,'Two year':2})
# data_pd_numberical=data_pd_numberical.replace({'Electronic check':0,'Mailed check':1,'Bank transfer (automatic)':2,'Credit card (automatic)':3})
# ###numberical feature
# data_pd_numberical=data_pd[['tenure','MonthlyCharges','TotalCharges']]


# ###轉float
# for name in data_pd_numberical:
#     print(name)
#     try:
#         data_pd_numberical[name]=data_pd_numberical[name].astype(float)
#     except:
#         data_pd_numberical[name] = pd.to_numeric(data_pd_numberical[name], errors='coerce')

### numberical feature correlation matrix
# print(data_pd_numberical.corr(method='pearson'))


### visualization: feature distribution
# features=data_pd_numberical.columns
# for i in range(0, len(features)):
#     plt.subplot(2, len(features)//2 + 1, i+1)
#     plt.hist(data_pd_numberical[features[i]], bins=50)
#     plt.xlabel(features[i])
#     plt.tight_layout()
# plt.show()

### Multivariate Analysis
# features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
#        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
#        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#        'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
# for feature in features:
#     sns.histplot(data = data_pd, x = feature, hue = 'Churn', multiple = 'dodge', bins=30)
#     plt.title(f'Histogram of {feature} with Churn')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.show()

###specfic selec
# cond=data_pd['MonthlyCharges']<=20
# new_df=data_pd[cond]
# features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
#        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
#        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#        'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
# for feature in features:
#     sns.histplot(data = new_df, x = feature, hue = 'Churn', multiple = 'dodge', bins=30)
#     plt.title(f'Histogram of {feature} with Churn')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.show()

## drop useless col
# data_pd.drop(columns=["customerID"], inplace=True)

le = LabelEncoder()
for c in data_pd.columns:
    data_pd[c] = le.fit_transform(data_pd[c])

cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', ]

X_train, X_test, y_train, y_test = train_test_split(data_pd[cols], data_pd["Churn"], train_size=0.8, random_state=42)

mms = MinMaxScaler()
X_train=mms.fit_transform(X_train)
X_test=mms.fit_transform(X_test)


def objective(trial):
    param_grid = {
        "C":trial.suggest_float("C",40,80),
        "degree":trial.suggest_int("degree",1,3),
        #'kernel': trial.suggest_categorical("kernel", ["poly", "rbf","sigmoid"])
        'kernel': trial.suggest_categorical("kernel", ["poly"])
    }
    svc = SVC(**param_grid)
    svc.fit(X_train, y_train)
    # y_pred = svc.predict(X_test)
    pickle.dump(svc, open(f_path+str(trial.number)+"_svc.m",'wb'))
    # CM=confusion_matrix(y_test, y_pred)
    # acc=(CM[0,0]+CM[1,1])/CM.sum()
    return svc.score(X_test, y_test)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials = 100)
print('Number of finished trials:', len(study.trials))
print('Best trial parameters:', study.best_trial.params)
print('Best score:', study.best_value)

fig=optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()
fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()