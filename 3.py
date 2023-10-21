import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
'''
fewer students per teacher allow the teacher 
to give focused attention to each student
'''

df = pd.read_csv("./data/enem_scores.csv")
print(df.shape)

