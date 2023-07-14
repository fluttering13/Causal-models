# 自治模型 (Autonomy model)
自治的意思是源自JOHN ALDRICH 在1989 提到的概念

https://doi.org/10.1093/oxfordjournals.oep.a041889

主要就是兩個方程可以耦合在一起

以下我們會先從一個簡單的經濟學模型開始討論，可以包含需求數量 $q$ 與市場價格 $p$：

而家庭薪資收入 $i$ 會影響到需求 $q$；

市場工資率 $w$ 會影響到實際市場價格 $p$

需求 $q$ 與價格 $p$ 彼此互相影響
<div align=center><img src="https://github.com/fluttering13/Causal-models/blob/master/pic/autonomy_1.png" width="500px"/></div>

所以我們可以建立這樣的兩條耦合

$$q = {b_1}p + {d_1}i + {u_1}$$

$$p = {b_2}q + {d_2}w + {u_2}$$

在這裡，我們先化簡一下這個方程 $c_1={d_1}i + {u_1}$ 和 $c_2={d_2}w + {u_2}$
$$q = {b_1}p + {c_1}$$
$$p = {b_2}q + {c_2}$$
$${q_0} = {b_1}{p_0} + {c_1}$$
$${p_1} = {b_2}{q_0} + {c_2} = {b_1}{b_2}{p_0} + {b_2}{c_1} + {c_2}$$
$${q_1} = {b_1}{p_1} + {c_1} = {b_1}^2{b_2}{p_0} + \left( {{b_1}{b_2} + 1} \right){c_1} + {b_1}{c_2}$$
$${p_2} = {b_2}{q_1} + {c_2} = {b_1}^2{b_2}^2{p_0} + \left( {{b_1}{b_2}^2 + {b_2}} \right){c_1} + \left( {{b_1}{b_2} + 1} \right){c_2}$$
$${q_2} = {b_1}{p_2} + {c_1} = {b_1}^3{b_2}^2{p_0} + \left( {{b_1}^2{b_2}^2 + {b_1}{b_2} + 1} \right){c_1} + \left( {{b_1}^2{b_2} + {b_1}} \right){c_2}$$
$${p_3} = {b_2}{q_2} + {c_2} = {b_1}^3{b_2}^3{p_0} + \left( {{b_1}^2{b_2}^3 + {b_1}{b_2}^2 + {b_2}} \right){c_1} + \left( {{b_1}^2{b_2}^2 + {b_1}{b_2} + 1} \right){c_2}$$
根據規律
$${p_n} = {b_1}^n{b_2}^n{p_0} + \left( {{b_1}^{n - 1}{b_2}^n + {b_1}^{n - 2}{b_2}^{n - 1} + ... + {b_2}} \right){c_1} + \left( {{b_1}^{n - 1}{b_2}^{n - 1} + ... + {b_1}{b_2} + 1} \right){c_2}$$
經等比級數化簡
$${p_n} = {b_1}^n{b_2}^n{p_0} + {{{b_2}\left( {{b_1}^n{b_2}^n - 1} \right)} \over {\left( {{b_1}{b_2} - 1} \right)}}{c_1} + {{\left( {{b_1}^n{b_2}^n - 1} \right)} \over {\left( {{b_1}{b_2} - 1} \right)}}{c_2}$$
如果係數 $b_1$ 與 $b_2$ 介於0跟1之間，則n很大時收斂到
$${p_{n \to \infty }} = {{{b_2}{c_1} + {c_2}} \over {1 - {b_1}{b_2}}}$$
即，市場最終平衡價格與初始價格無關，主要被這些係數所決定

也就是在有突發事件的時候，所有價格都會用指數衰減至最終價格
<div align=center><img src="https://github.com/fluttering13/Causal-models/blob/master/pic/autonomy_2.png" width="500px"/></div>

簡單的code
```
import numpy as np
from matplotlib import pyplot as plt

b1=0.8
b2=0.8
d1=0.1
d2=0.1
i=100
w=100
u1=0
u2=0
q=[]
p=[]
p.append(100)
for k in range(0,10):
  q0=b1*p[-1]+d1*i+u1
  q.append(q0)
  p0=b2*q[-1]+d2*w+u2
  p.append(p0)

print(p[-1])
print('平衡價格',(b2*(d1*i+u1)+d2*w+u2)/(1-b1*b2))
plt.scatter(range(0,10+1),p)
```

當我們稍微調整一下參數，這裡代表遞增但是以自然指數衰減收斂至最終價格

```
b1=1.1
b2=0.76
```
<div align=center><img src="https://github.com/fluttering13/Causal-models/blob/master/pic/autonomy_3.png" width="500px"/></div>
即使是中途發生限價行為，還是會因應需求，會回歸到平衡價格

```
for k in range(0,point_number):
  q0=b1*p[-1]+d1*i+u1
  q.append(q0)
  p0=b2*q[-1]+d2*w+u2
  if k==10:
    p0=110
  p.append(p0)
```
<div align=center><img src="https://github.com/fluttering13/Causal-models/blob/master/pic/autonomy_5.png" width="500px"/></div>
我們可以在迴圈內加入讓這個係數跟現價有關的因子，例如價格過高，對於限價反應b1就會減少
例如我們可以這樣試想一個狀況，當一個新品出來，大家會開始瘋搶，但是最後一定會回到一個大家公認的市場價格

所以只要做簡單的擬合，就可以大致知道產品的峰值或是週期


```
for k in range(0,point_number):
  q0=b1*p[-1]+d1*i+u1
  q.append(q0)
  p0=b2*q[-1]+d2*w+u2
  p.append(p0)
  b1=b1-0.02*(p[-1]-100)/100
```
<div align=center><img src="https://github.com/fluttering13/Causal-models/blob/master/pic/autonomy_6.png" width="500px"/></div>
