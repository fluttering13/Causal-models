# Fisher-randomization
在之前的Introduction有介紹，什麼時候Causality會等價成correlation

也在裡面提及反事實的概念，所以很多因果模型就是在進行反事實項的推估

只要我們可以利用隨機化實驗消彌其他變項的影響，相關性就可以看成因果關係

回顧一下做隨機化實驗的重要性，這裡我們會介紹一個方式去評估未知反事實的方法Fisher randomization

在真實數據內，我們會利用盡量隨機化的變數來對反事實項進行填補，並且做統計假說來看顯不顯著

## 故事
故事是這樣的，今天在一個悠閒的午後，不如就來杯可口的奶茶吧

我們來看看流程是什麼：先倒牛奶，再倒紅茶；還是先倒紅茶，再倒牛奶

如我們想知道其中的差異，那我們就可以盲測，並且隨機化裡面的順序，並透過假設檢定來驗證

假設有總共有8次實驗，先倒牛奶占了4次，先倒紅茶佔了4次

那麼總共有 $C^8_4$ 排列方式，在亂猜的狀況下，只有1/70的機率才能夠猜中

當我們虛無假設為這兩者沒有任何的差異，然後真的做了實驗了，實驗結果是全部八次都有差異

那麼根據p值的定義就是把有差異極端狀況下的機率加在一起，這裡就是1/70

如果設定顯著水平為0.05，那我們就可以拒絕這個null hypothesis (在這邊是Sharp Null hypothesis)

## 穩定性假設(SUTVA) 與 個體因果效應
所有的干預效應，對這些個體之間並沒有交互作用

即在treatment T=1 或者 T=0 每個個體i的潛在結果寫成 $Y_{i}$

個體所看到的因果效應假設就還蠻自然定義成根據不同干預下潛在結果之間的差值 
$$\tau_i=Y_{i}(1)-Y_{i}(0)$$

## 干預分配機制

$$Y_i=T_iY_i(1)+(1-T_i)Y_i(0)$$

## i.i.d 超幾何分布

在這裡假設總共N次實驗，其中先倒牛奶佔了$N_1$次

隨機試驗的每種排列的機率都是相同的 P=$\frac{1}{C^{N}_{N_1}}$

## Null hypothesis
$$H_0:Y_i(0)-Y_i(1)=0$$
意思就是無論有沒有干預，結果應該都沒差

## Randomized experiments
那要怎麼把隨機事件考慮進去，就是 $T_1,T_2......$ 這些下標的relabeling做permutation

例如說有L個，那我們每次的得到一個 p-value $p_l$ ，之後取平均就是最後的p值了

## 例子
這邊舉一個給不給平板以判斷學習績效的一個例子
```
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
```
group1是面對面上課的分數，group2是線上上課的分數

接著我們把它轉成有反事實的因果表格
```

causal_df=pd.DataFrame()
label_array=np.concatenate([np.zeros(len(group1)),np.ones(len(group2))],0)

label=pd.Series(label_array)

Y_i0=np.concatenate([group1,np.zeros(len(group2))])
# print(Y_i0)
Y_i1=np.concatenate([np.zeros(len(group1)),group2])

causal_df['Y_i0']=Y_i0
causal_df['Y_i1']=Y_i1
causal_df['treatment']=label
```
反事實的部份我們先填零
```
         Y_i0      Y_i1  treatment
0    63.29997   0.00000        0.0
1    79.96000   0.00000        0.0
2    90.00000   0.00000        0.0
3    88.31000   0.00000        0.0
4    86.64000   0.00000        0.0
..        ...       ...        ...
209   0.00000  70.00000        1.0
210   0.00000   0.00000        1.0
211   0.00000  70.05000        1.0
212   0.00000  66.69000        1.0
213   0.00000  83.29997        1.0
```
之後我們引入虛無假設，$T_i(0)=T_i(1)$

```
causal_df.iloc[len(group1):,0]=causal_df.iloc[len(group1):,1]
causal_df.iloc[:len(group2),1]=causal_df.iloc[:len(group2):,0]
```
反事實表格就會長成這樣
```
         Y_i0      Y_i1  treatment
0    63.29997  63.29997        0.0
1    79.96000  79.96000        0.0
2    90.00000  90.00000        0.0
3    88.31000  88.31000        0.0
4    86.64000  86.64000        0.0
..        ...       ...        ...
209  70.00000  70.00000        1.0
210   0.00000   0.00000        1.0
211  70.05000  70.05000        1.0
212  66.69000  66.69000        1.0
213  83.29997  83.29997        1.0
```
這裡我們使用的estimator是無偏的，順便看一下初始的t0
```
N1=len(group1)
N2=len(group2)

#####in average outcomes by treatment:Yi​=Ti​Yi​(1)+(1−Ti​)Yi​(0)
def treatment_estimator(col0,col1,col_label,N1,N2):
    return sum(col1*col_label)/(N1)+sum((1-col_label)*col0)/(N2)
t0=treatment_estimator(causal_df['Y_i0'],causal_df['Y_i1'],causal_df['treatment'],N1,N2)
print(t0)
```
```
157.95434065780137
```
再來我們shuffle數據創造新的，做多次實驗
```
t_list=[]
##generate new df
N=5000
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
print(count/N)
```
這邊Exact p-value的定義為大於等於初始值
```
0.0016
```
小於0.025，我們拒絕虛無假設代表兩個set之間有因果關係存在

# Neyman's Repeated Sampling Approach
我們接下來來看另外一個評估的方式，Neyman框架

相較於Fisher，這裡的不同點在於Null hypothesis，是取期望值，也就是

這邊有N個個體中，有N_1個隨機進行處理，也就是在原本的基礎上考慮了抽樣這個步驟
$$H_0:E(Y_i(0)-Y_i(1))=0$$
## 總體平均因果效應
$$\tau=\sum^N_{i=1}(Y_i(1)-Y_i(0))$$
## 總體平均因果效應(無偏估計)
$$\hat\tau=\frac{1}{N_1}\sum^N_{i=1}T_i Y_i(1)-\frac{1}{N_2}\sum^N_{i=1}T_i Y_i(0)$$
## 方差
$$Var(\hat\tau)=S_1^2/N_1+S_0^2/N_0-{S_\tau}^2/N_0$$
where
$$S_\tau^2=\frac{1}{N-1}\sum_{i=1}^N(\tau_i-\tau_\mu)^2$$
但這邊會有一個問題就是$\tau_i$並無法直接從數據觀測，一般來說我們都是用其他的方式來對反事實項進行估計

$$S_\tau^2=\frac{1}{N-1}\sum_{i=1}^N(\tau_i-\tau_\mu)^2$$

所以這邊的方差在原本的是直接忽略$S_\tau^2/N_0$這一項，當然這會造成這個估計是有偏的

所以我們需要一個假設

## constant additive treatment effect

$$S_\tau=0$$

## 置信區間

此時虛無假設為
$$H_0:E(Y_i(0)-Y_i(1))=0$$

構造Z map到一個normal distribution

$$\frac{\hat\tau-\tau_0}{\sqrt{S_\tau}}$$

只要在這個置信區間內，我們就無法拒絕虛無假設，置信區間可以寫成

$$
\begin{aligned}
& C I=\left\{\tau_{\text {interval }}:\left|\frac{\tau-\tau_\mu}{\sqrt{S_\tau}}\right| \leq Z_{1-\alpha / 2}\right\} \\
& =\left\{\tau_{\text {interval }}: \tau-Z_{1-\alpha / 2} \sqrt{S_\tau} \leq \tau_{\text {interval }} \leq \tau+Z_{1-\alpha / 2} \sqrt{S_\tau}\right\}
\end{aligned}
$$

# 一般化與評論
回顧一下，在Fisher裡面我們有一個sharp null

$$H_0:Y_i(0)-Y_i(1)=0 \: \forall i=1,2,3......$$

這意味著我們假設所有的干預結果跟干預無關且，其隨機性如何做隨機化有關，所以我們做permutation test。我們稍微解釋一下

我們令分配機制的向量叫做T，這邊大寫都表示為向量的意思

$$T = \left( {{T_1},{T_2}......{T_n}} \right)$$

結果向量為

$$Y = \left( {{Y_1},{Y_2}......} \right)$$

那隨機化分配機制的條件機率可以如以下定義

$$P(T|Y)=C^N_{N_1}$$

選定的無偏統計量為

$$\hat\tau=\frac{1}{N_1}\sum^N_{i=1}T_i Y_i(1)-\frac{1}{N_2}\sum^N_{i=1}T_i Y_i(0)$$

我們想要檢驗虛無假設，但是因為反事實的關係，我們很難找這個統計量

可是我們知道這個統計量來自隨機化，於是我們可以用「捏」的方式去塑造一些pseudo data

也就是利用Monte Carlo 方法模擬統計量的分布，最簡單的方式就是做重複多次排列 再取平均

回顧Neyman的方法，加入了抽樣的想法

因為高估標準差的關係，Neyman的置性區間是比較保守的

## 小總結

對於Fisher方法，在我們知道某個母體很大的時候比較適用，例如說某個政策對某個國家的影響

因為我們很難知道比這個國家還大的母體是什麼

但是當這個母體不夠大的時候，這時比較適用於Neyman

## treatment effect heterogeneity
回顧Fish的方式，忽略異質性，這顯然不符合我們對干預造成效果的期待

所以有一種推廣的方式是相比sharp null 我們另虛無假設兩種干預的分布是一樣的

$$H_0:F_{Y(1)}=F_{Y(0)}$$

考慮以上的虛無假設，可以構造推斷框架如以下

$$H_0:F_{Y(1)}(y)=F_{Y(0)}(y+\Delta)$$

