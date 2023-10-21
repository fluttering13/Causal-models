import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats.weightstats import ztest

data = pd.read_csv("./data/online_classroom.csv")

new_df=data.assign(class_format = np.select(
     [data["format_ol"].astype(bool), data["format_blended"].astype(bool)],
     ["online", "blended"],
     default="face_to_face"
 ))

mean_df=new_df.groupby(by='class_format').mean()
std_df=new_df.groupby(by='class_format').std()
count_df=new_df.groupby(by='class_format').count()
var_df=new_df.groupby(by='class_format').var()

group1=new_df[new_df['class_format']=='face_to_face']['falsexam'].values
group2=new_df[new_df['class_format']=='online']['falsexam'].values


print(group1,group2)
###以下接設定alpha=0.05，大於此則拒絕H0，小於則接受
###H0：此分布常態分佈
# print(stats.shapiro(group1))
# print(stats.shapiro(group2))
'''
拒絕為常態分佈
必須以無母數檢定應對
'''

###kruskalwallis test
# print(stats.mstats.kruskalwallis(group1, group2))
####差異不顯著

# group_1_mean=mean_df['falsexam','face_to_face']
#  .groupby(["class_format"])
#  .mean())
