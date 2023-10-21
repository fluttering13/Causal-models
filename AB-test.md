# A/B test
A/B test是一個廣泛運用的測試，通常是想要驗證在介入干涉之後是否在統計上有其影響

A被當作是控制組，就是什麼都不做

B為變化組，就是介入干涉後

以下我們介紹有關於A/B test實用的知識
## AB test sample數要取多少
來源自

https://datasciencegenie.com/how-to-calculate-the-sample-size-for-an-a-b-testing-including-calculator-derivation/

樣本數如果過少，統計檢定力不足，會只能看到片面的資訊

樣本數也不太可能到很多，因為需要耗費相當多的資源才能獲得數據



所以決定採樣的數字是多少是一件重要的事情

先說簡化過後的結論，所需要的sample數為，

$\sigma$為標準差，

$\delta$為MDE (Minimum Detectable Effect)，期望最小的指標效果差異，也就是區分介入前後的值

則所需要本數為
$$16 \frac{\sigma^2}{\delta^2}$$

例如說針對某些肥胖的人群，體重平均值為90kg，標準差為10kg，今天我們將減重5kg定義為有效

$$16*\frac{10^2}{5^2}=64$$

另外想要直接計算沒有簡化版的也可以直接利用scipy中的

```
import scipy
scipy.statsmodels.api.sm.stats.samplesize_proportions_2indep_onetail(
diff = 0.05, #MDE
prop2 = 0.2, #對照組資料所占的比例 p2=p1+MDE
power = 0.8, # 檢定力，當對立假說為真時拒絕虛無假說的機率，可以從先驗機率去抓
alpha=0.05 #Significance Level 顯著水準
null_var="prop2" #prop2 代表p2 ; pooled 代表(p1+p2)/2 使用合併變異數(pooled Variance)
)
```

### 推導

P1是對照組
P2是實驗組

先假定我們的虛無假說 (Null Hypothesis) 
$$H_0:p_1-p_2=0$$
則對立假說
$$H_1:p_1-p_2<0$$

根據I.I.D.假設，則抽樣分布$\hat{p}_1$與$\hat{p}_2$的平均跟標準差可以寫作
$$
\hat{p}_1-\hat{p}_2 \sim \mathcal{N}\left(p_1-p_2, \frac{p_1\left(1-p_1\right)}{n_1}+\frac{p_2\left(1-p_2\right)}{n_2}\right)
$$

標準化之後的$Z$可以寫成

Z=$\frac{(\hat{p}_1-\hat{p}_2)-(p_1-p_2)}{\sqrt{p_1(1-p1)/n_1}+p_2(1-p2)/n_2}$

引入T1 error $\alpha$
$$P[{H_0}\;is\;rejected|{H_0}\;is\;true] = \alpha $$
$$P[{H_0}\;is\;acepted|{H_0}\;is\;true] = 1-\alpha $$
帶入accepted的定義，$Z_{\alpha}$即信心水準
$$P[Z \le {Z_\alpha }|{H_0}\;is\;true]$$
令$P=(n_1 \hat{p_1}+n_2 \hat{p_2})/(n_1+n_2)$ pooled sample proportion

移項一下
$$P[\hat{p_1}-\hat{p_2} \le Z_{\alpha} \sqrt{P(1-P)(1/n_1+1/n_2)}| H_0\; is\; True]=1-\alpha $$
或是寫成
$$P[\hat{p_1}-\hat{p_2} \le Z_{\alpha} \sqrt{(\frac{\hat{p_1}+\hat{p_2}}{n_2})(1-\frac{(\hat{p_1}+\hat{p_1})}{2})}| H_0\; is\; True]=1-\alpha$$

再引入T2 error $\beta$
$$P[{H_0}\;is\;rejected|{H_1}\;is\;true] = \beta $$
在T1怎麼導這邊就怎麼導
$$P[\hat{p_1}-\hat{p_2} \gt Z_{\alpha} \sqrt{(\frac{\hat{p_1}+\hat{p_2}}{n_2})(1-\frac{(\hat{p_1}+\hat{p_1})}{2})}| H_1\; is\; True]=1-\beta$$
標準化一下
$$P[Z \gt \frac{Z_{\alpha} \sqrt{(\frac{\hat{p_1}+\hat{p_2}}{n_2})(1-\frac{(\hat{p_1}+\hat{p_1})}{2})}-\mu}{\sqrt{\hat{p_1}(1-\hat{p_1})/n_1+\hat{p_2}(1-\hat{p_2}/\hat{p_2})}}| H_1\; is\; True]=1-\beta$$
又 根據 $z_{\beta}$ 的定義
$$-z_\beta \ge Z$$
則
$$-z_\beta \gt \frac{Z_{\alpha} \sqrt{(\frac{\hat{p_1}+\hat{p_2}}{n_2})(1-\frac{(\hat{p_1}+\hat{p_1})}{2})}-\mu}{\sqrt{\hat{p_1}(1-\hat{p_1})/n_1+\hat{p_2}(1-\hat{p_2}/\hat{p_2})}}$$
把裡面不等式 $n_2$ 拿出來
$$n_2 \ge (z_\alpha \sqrt{(\hat{p_1}+\hat{p_2})(1-(\hat{p_1}+\hat{p_2}/2))+z_\beta\sqrt{\hat{p_1}(1-\hat{p_1})+\hat{p_2}(1-\hat{p_2})}})^2/\mu^2$$

根據$p_2=(1+\delta)p_1$，$\delta$即MDE，又因為i.i.d
所以我們抽樣也可以有$\hat{p_2}=(1+\delta)\hat{p_1}$，帶入化簡後

$$n_2=\frac{z_\alpha \sqrt{(2+\delta)(2-2\hat{p_1}-p_1\delta)}/2+z_\beta \sqrt{1-p_1+(1+\delta)(1-p_1)\delta}}{\delta^2\hat{p_1}}$$

通常會設定$\alpha=0.025$與$\beta=0.05$，則
$$Z_{\alpha}=-1.96,Z_{\beta}=-0.84$$
帶入上面就可以大概可以得到
$$(3.96)^2 \frac{\sigma^2}{\delta^2} \approx 16 \frac{\sigma^2}{\delta^2}$$

## AB test 什麼時候要停止實驗
引源自 https://www.evanmiller.org/how-not-to-run-an-ab-test.html

通常我們可能會想重複坐著一系列的實驗，已證明在數據量大的時候

例如做了十次，然後有八次統計顯著 兩次不顯著，這個時候如果我們就取顯著的平均，那就落入了陷阱，因為這邊沒有把每一次有多顯著的資訊放進去

因為以顯著的次數來統計就會喪失了對顯著的衡量 

比較好的方式應該是用連續的measure去衡量

例如說：

$$\delta=(t_{\alpha/2}+t_\beta)\sigma\sqrt{2/n}$$

其中$t_{\alpha/2}$是t統計的顯著水平

或其他的提到的下列的方式

### Bayesian A/B Testing
源自
https://www.evanmiller.org/bayesian-ab-testing.html

以下我們引入Bayesian公式，並以beta distribution作為先驗，然後依據你手中有的數據去更新後驗，這裡我們直接給公式，詳細推導可直接看上方網址

以下這個measure是由beta function $B$所構成，當實驗組比控制組有成效的機率
$$P(P_B>P_A)=\sum_{i=0}^{\alpha_B-1}\frac{B(\alpha_A+i,\beta_B+\beta_A)}{(\beta_B+i)(B(1+i,\beta_B)B(\alpha_A,\beta_A))}$$   
$\alpha$代表成功數，$\beta$代表失敗數
再來是推廣到三個ABC
$$P(P_C>max(p_A,p_B))=1-P(p_A>p_C)-p(p_B>p_c)+\sum_{i=0}^{\alpha_A-1}\sum_{i=0}^{\alpha_B-1}\frac{B(\alpha_C+i+j,\beta_C+\beta_B+\beta_A)}{(\beta_B+i)(B(1+i,\beta_A)(\beta_B+j)B(1+j,\beta_B)B(\alpha_C,\beta_C))}$$   

那如果是追蹤連續數據的實驗呢？例如說當周每日的銷售數目$\alpha_1$跟上周的每日銷售數目$\alpha_2$相比如何
那我們就使用
$$P(\lambda_1 \gt \lambda_2)=\sum_{k=0}^{\alpha_1-1}\frac{(\beta_1+\beta_2)^{-k-\alpha_2}\beta^k_1\beta^{\alpha_2}_2}{(k+\alpha_2)B(k+1,\alpha_2)}$$
同樣也可以推廣到三個比較對象
$$P(\lambda_1 \gt max(\lambda_2,\lambda_3))=1-P(\lambda_2>\lambda_1)-P(\lambda_3>\lambda_1)+\sum_{k=0}^{\alpha_2-1}\sum_{k=0}^{\alpha_3-1}\frac{\beta_1^{\alpha_1}\beta_2^{k}\beta_3^{l}}{(\beta_1+\beta_2+\beta_3)}\frac{\Gamma(k+l+\alpha_1)}{\gamma(k+1)\gamma(l+1)\gamma(\alpha_1)}$$