import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats.weightstats import ztest
from pandas import DataFrame as DF
import math
import random
import seaborn as sns

#####in average outcomes by treatment:Yi​=Ti​Yi​(1)+(1−Ti​)Yi​(0)
def treatment_estimator(col0,col1,col_label,N1,N2):
    return sum(col1*col_label)/(N1)+sum((1-col_label)*col0)/(N2)


data = pd.read_csv("./data/online_classroom.csv")

df=data.assign(class_format = np.select(
     [data["format_ol"].astype(bool), data["format_blended"].astype(bool)],
     ["online", "blended"],
     default="face_to_face"
 ))

mean_df=df.groupby(by='class_format').mean()
std_df=df.groupby(by='class_format').std()
count_df=df.groupby(by='class_format').count()
var_df=df.groupby(by='class_format').var()

group1=df[df['class_format']=='face_to_face']['falsexam'].values
group2=df[df['class_format']=='online']['falsexam'].values

group3=df[df['class_format']=='face_to_face']['falsexam'].values+10

def fisher_exp(group1,group2,N):
    causal_df=pd.DataFrame()
    label_array=np.concatenate([np.zeros(len(group1)),np.ones(len(group2))],0)

    label=pd.Series(label_array)

    Y_i0=np.concatenate([group1,np.zeros(len(group2))])
    # print(Y_i0)
    Y_i1=np.concatenate([np.zeros(len(group1)),group2])

    causal_df['Y_i0']=Y_i0
    causal_df['Y_i1']=Y_i1
    causal_df['treatment']=label
    #print(causal_df)
    causal_df.iloc[len(group1):,0]=causal_df.iloc[len(group1):,1]
    causal_df.iloc[:len(group2),1]=causal_df.iloc[:len(group2):,0]
    #print(causal_df)
    N1=len(group1)
    N2=len(group2)
    t0=treatment_estimator(causal_df['Y_i0'],causal_df['Y_i1'],causal_df['treatment'],N1,N2)
    ### check whether they are in normal distribution
    # statistic_1,pvalue_1=stats.shapiro(group1)
    # statistic_2,pvalue_2=stats.shapiro(group2)
    # print(pvalue_1,pvalue_2)

    # print(t0)
    t_list=[]
    ##generate new df
    #N=5000
    count=0
    count_prime=0
    for i in range(N):
        replaced_array=causal_df['Y_i0'].values
        np.random.shuffle(replaced_array)
        causal_df['Y_i0']=replaced_array

        t=treatment_estimator(causal_df['Y_i0'],causal_df['Y_i1'],causal_df['treatment'],N1,N2)
        t_list.append(t)
        if t>=t0:
            count=count+1
    std=np.std(t_list)
    print(sum(t_list)/len(t_list),t0)
    z_score=(t0-sum(t_list)/len(t_list))/(std/math.sqrt(len(t_list)))
    p_value=(1-stats.norm.cdf(z_score))*2
    return count/N,z_score,p_value


print(fisher_exp(group1,group2,5000))
print(fisher_exp(group1,group3,5000))
# print(count/N,t0)
# z_score=(t0-count/N)/np.std(t_list)
# print((1-stats.norm.cdf(z_score)))



###other way




