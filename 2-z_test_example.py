import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import ztest
import math
data1 = [650,730,510,670,480,800,690,530,590,620,710,670,640,780,650,490,800,600,510,700]
data2 = [630,720,462,631,440,783,673,519,543,579,677,649,632,768,615,463,781,563,488,650]

sample_mean1 = np.mean(data1)
sample_mean2 = np.mean(data2)
sample_size1 = np.count_nonzero(data1)
sample_size2 = np.count_nonzero(data2)
population_mean_diff = 0
population_std1 = np.std(data1,ddof=1)
population_std2 = np.std(data2,ddof=1)
###以下設定alpha=0.05，大於此則接受H0，小於則拒絕
alpha = 0.05

###H0：此分布常態分佈
print(stats.shapiro(data1))
print(stats.shapiro(data2))


###Method 1: Using built in function of ztest

z,p = ztest(x1=data1,x2=data2,value=population_mean_diff,alternative='two-sided')
print('Z-score:',z,'\nP-value:',p)
 
    
###Method 2: Calculating Z-score  

zscore = ((sample_mean1-sample_mean2)-(population_mean_diff))/(math.sqrt((population_std1**2/sample_size1)+(population_std2**2/sample_size2)))
print('z_score',zscore)
print('p_value',(1-stats.norm.cdf(zscore))*2)


###拒絕虛無假設，兩個list有統計差別
