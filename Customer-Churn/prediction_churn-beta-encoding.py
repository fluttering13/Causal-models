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
import category_encoders as ce

class BetaEncoder(object):
        
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha = alpha_prior + n
        beta =  beta_prior + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
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
print(data_pd.columns)
###enocode the categorical meassage 
le = LabelEncoder()
for c in data_pd.columns:
    data_pd[c] = le.fit_transform(data_pd[c])
data_pd.drop(columns=["customerID"], inplace=True)

### list the columns
cols=data_pd.columns

#train, test, y_train, y_test = train_test_split(data_pd[cols], data_pd["Churn"], train_size=0.8, random_state=42)

###remember to kick off label before training
######beta encoding########
###input: train is train_data+label_data (put the label_y in the last one)
###output: transformed train_data+label_data
def beta_encoding_process(train,encoder_cols,N_min = 1000):
    feature_cols=[]
    learn_label_name=train.columns[-1]
    for c in encoder_cols:
        if c==learn_label_name:
            break
        be = BetaEncoder(c)
        be.fit(train, learn_label_name)
        # mean
        feature_name = f'{c}_mean'
        train[feature_name] = be.transform(train, 'mean', N_min)
        feature_cols.append(feature_name)

        # mode
        feature_name = f'{c}_mode'
        train[feature_name] = be.transform(train, 'mode', N_min)
        feature_cols.append(feature_name)

        # median
        feature_name = f'{c}_median'
        train[feature_name] = be.transform(train, 'median', N_min)
        feature_cols.append(feature_name)    

        # var
        feature_name = f'{c}_var'
        train[feature_name] = be.transform(train, 'var', N_min)
        feature_cols.append(feature_name)        

        # skewness
        feature_name = f'{c}_skewness'
        train[feature_name] = be.transform(train, 'skewness', N_min)
        feature_cols.append(feature_name)    

        # kurtosis
        feature_name = f'{c}_kurtosis'
        train[feature_name] = be.transform(train, 'kurtosis', N_min)
        feature_cols.append(feature_name)
        # train=train.dropna(how='any')
        # test=test.dropna(how='any')
    #train.drop(columns=learn_label_name, inplace=True)
    return train[feature_cols+[learn_label_name]]

encoder_cols=cols
new_data=beta_encoding_process(data_pd,encoder_cols, N_min = 1000)
cols=new_data.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(new_data[cols], data_pd["Churn"], train_size=0.8, random_state=42)

# mms = MinMaxScaler()
# X_train=mms.fit_transform(X_train)
# X_test=mms.fit_transform(X_test)


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