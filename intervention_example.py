import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import style

treatment_A_dead_number=[210,30]
treatment_B_dead_number=[5,100]
treatment_A_number=[1400,100]
treatment_B_number=[50,500]
data_dead_number={'treatment_A':treatment_A_dead_number,'treatment_B':treatment_B_dead_number}
data_dead_number=pd.DataFrame(data_dead_number,index=['mild','severe']).T
data_number={'treatment_A':treatment_A_number,'treatment_B':treatment_B_number}
data_number=pd.DataFrame(data_number,index=['mild','severe']).T

data_number=data_number.assign(total=data_number.sum(axis=1))
data_dead_number=data_dead_number.assign(total=data_dead_number.sum(axis=1))


print(data_dead_number/data_number)
def create_intervention(data_number,data_dead_number):
    indexes=data_number.index
    data_rates=data_dead_number/data_number
    data_rates=data_rates.rename(columns={'total':'correlation'})


    new_list=[]
    c_numbers=data_number.sum()
    for i in range(data_rates.shape[0]):
        tmp_sum=0
        for j in range(data_rates.shape[1]-1):
            E_y_given_t_c=data_rates.iloc[i,j]
            p_c=c_numbers.iloc[j]/c_numbers.iloc[-1]
            tmp_sum=tmp_sum+E_y_given_t_c*p_c
        new_list.append(tmp_sum)
    new_list=pd.DataFrame({'intervention':new_list},index=indexes)
    data_rates=pd.concat([data_rates,new_list],axis=1)
    return data_rates

data_rates=create_intervention(data_number,data_dead_number)
print(data_rates)

#data_rates.shape[0]



