# 前言
我原本是在做關於貝爾不等式的研究時，看到有些方法是基於因果模型。
以前看的相關研究就覺得雲裡霧裡的，
雖然有關注一下相關的研究，也僅僅只是之到了干預的概念，但並沒有很深刻。
在當兵的時侯拜讀了Judea Pearl的Causality Model, Reasoning and inference, Senond Edtion
對其中的東西理解了一些些，
在這個欄目我會深入淺出的講一下概念，並實作一些範本。

# 相關性v.s因果關係
日常常見很多相關性的定義方式，
往往相關性與因果關係有著密切的關係：
「這個人因為懶惰而導致考試不及格」
「有長期作息不穩定容易導致癌症的發生」
「這支股票因為美國進行降息政策而導致價格下跌」

但又往往不一樣
「富人往往戴得起奢侈品，但戴得起奢起品不一定是富人階級」
奢侈品同時也有社交價值在其中，於是人們會想要購買奢侈品的原因之一
「穿著鞋子睡覺的人跟沒穿著鞋睡覺的人比起來，隔天穿著鞋子睡覺的那群人頭痛的發作率比較高」
因為穿著鞋子的人通常是喝醉酒的人，但穿鞋子本身跟頭痛並無因果關係，但有高相關性。
以上諸多例子，都代表著我們往往看到線圖很像或是從數據上的相關性很強，就見到黑影就開槍，並不是一個合理的因果推斷，
像是像在常用的統計策略或是機器學習都是基於相關性，而導致後續延伸一系列的問題。

結論：要區分出因果關係需要建立起因果模型

# 本體論v.s認識論
在許多時候我們觀測各種數據，都是基於相關性，但往往我們想知道的是事情的因果關係
這其實是基於不同的哲學準則來觀測與預測事物，以下我們來以數據科學上面的例子來看
## Bayes詮釋 (Bayes interpretation)
認識論代表著我們是希望透過過往的經驗來更加深對於事物的理解
我們習慣觀察機率分布，時常有時候藉由過往經驗來進行預測
這是基於數據的方式來增強或是削減我們對於機率事件的信任程度
$$P\left( {H|e} \right) = \frac{{P(e|H)P\left( H \right)}}{{P(e)}}$$
其中 $P(H|e)$ 是後驗機率(posteriori probability)，根據過往的證據 $e$ (evidence)去看假設 $H$ (hypothesis)
而 $P(H)$ 被稱為先驗機率(Prior probability)，是根據過往的觀測，前人的經驗所得到的機率

舉例來說
假如今天火災警報器響起，

(1)在真的火災發生導致響起的機率為0.95 即 P(警報響|火災)=0.05

(2)在沒發生火災的時候導致響起的機率為0.01 即 P(警報響|沒發生火災)=0.01

(3)根據以往的經驗，某間樓發生火災的機率為P(火災)=0.0001

$$P\left( {H|e} \right) = \frac{{P(e|H)P\left( H \right)}}{{P(e)}}{\rm{ = }}\frac{{P(e|H)P\left( H \right)}}{{P(e|H) + P(e| \bot H)}} = \frac{{0.95 \times {{10}^{ - 4}}}}{{0.95 \times {{10}^{ - 4}} + 0.01 \times \left( {1 - {{10}^{ - 4}}} \right)}} \approx 0.00941$$
這意味著我們從過往的經驗，先驗機率 $0.0001$ 增強到後驗機率 $0.00941$
```
import numpy as np
import matplotlib.pyplot as plt
p_eh=0.95
p_enonh=0.01
p_0=0.0001
p_list=[]
def Baysian_loop(p_eh,p_0,p_enonh):
  p=p_eh/(p_eh+p_enonh*(1/p_0-1))
  return p

for i in range(0,10):
  p_list.append(p_0)
  p_i=Baysian_loop(p_eh,p_0,p_enonh)
  p_0=p_i
plt.scatter(range(0,10),p_list)
```
如果我們在經過四個迴圈下 條件機率p(e|H)會趨近於0.99987 越來越接近1
<div align=center><img src="https://github.com/fluttering13/Causal-models/blob/master/pic/b_loop.png" width="500px"/></div>
代表著如果我們得知過往的經驗，則我們對這個回顧性支持會越強烈。

## Bayes網路 (Bayes networks)
本體論代表著我們很自然地會去追朔本源，問說一個類別的實體是否存在於最基本的層次上？
去構造因果模型的方式就是使用有向無環圖DAG (Directed acyclic graph)，也被稱為是Bayes網路
$$P\left( {{x_1},......{x_n}} \right) = {\prod _i}P\left( {{x_i}|p{a_i}} \right)$$
整體分布 $P(x_1,......x_n)$ 可以被條件機率所連鎖，其中 $P(x_i|pa_i)$ 變數 $x_i$ 是基於父代變數 $pa_i$的。

舉例來說：

像用於描述所有古典事件著名的隱變量模型(local hidden variable model)可以被圖所表達
<div align=center><img src="https://github.com/fluttering13/Quantum-nonlocality/blob/master/Figure/Bell_sceanrio.png" width="300px"/></div>

$$P(a,b|x,y)=\sum_\lambda P(\lambda) P(a|x,\lambda) P(b|y,\lambda)$$

先提要一下結論，無論觀測環境如何改變，因果聯繫都應該是不變的，這符合追求本體論的精神。同時，透過機率分布去了解，去量化，又符合經驗論的精神，
那這跟以往數據學家透過機率分布來認識世界又有什麼程度上的差別呢？
這正是因果推斷要討論的事情，要如何去量化因果關係？以及在何種架構下的因果關係才是合理的？

# 因果推斷的幾個層次
1.關聯(Association)
2.干預(intervenion)
3.反事實(Counterfactual)
4.混雜(confounding)
5.分岐(ramification)

