# Phillips曲線(Phillips Curve)
Phillips曲線是指薪資增長率與失業率的曲線，由於很難衡量通貨膨脹，通常是以薪資增長率作為通絡膨脹的指標。
以下來介紹一下相關的金融模型：

# Lucas islands model
在理想封閉狀態裡面，市場流通的總金額數應該是不變的，但假如總流通的金額變多
是否會影響到市場的行為？
在介紹這個模型前，我們先提到兩個專有名詞
## 需求衝擊(Demand shocks)
需求衝擊是指當人們對商品或服務的需求改變時，造成價格行為改變的事件
如珍珠奶茶經由新聞傳播與廣告宣傳，造成人們在某段時期特別想喝
## 總量衝擊(Aggregate shocks)
當貨幣流通數量改變而造成的價格行為改變的事件
如日本央行實施大量印鈔量化寬鬆政策，讓日圓持續下跌

事實上我們只能觀察到價格與產品供應的變化，事實上是很難區分這兩者造成的影響。
## 傳播機制(propagation mechanism)
### 競爭市場(competitive markets)
購買者會根據價格聰明的選擇他們的策略去購買
並且這個市場沒有額外的勞工來進行額外的生產提供

### 隨機期望值(Rational Expectations)
現價/平均價可以當作是一個真實價格比率，認為是我們對於價格行為的衡量
這些的機率分布我們假定它們都是正態分布
### 不完美的資訊(imperfect information)
生產者只能觀察到現價，並試圖猜測總量衝擊的影響來進行價格制定；
購買者則是因為需求衝擊造成的影響來制定購買現性的策略，然而在不完美的資訊下，我們很難得知真實比率 $r_i$

## The model
註：為了方便符號使用，以下小寫已經取了log。
假設存在著N個島嶼，在其中一個島嶼z，則當前的供給 $q_i$ 可由 係數$\gamma$ 
$${q_i} = \left( {{p_i} - p} \right)/\left( {\gamma  - 1} \right) = {r_i}/\left( {\gamma  - 1} \right)$$
與現價 $p_i$ 和均價 $p$ 所決定 (即真實比率 $r_i$)
接著我們來看看他是怎麼來的
### 效用函數(utility function)
我們考慮一個很常在經濟學使用的效用函數
$$\max \left( {{C_i} - \frac{1}{\gamma }{L_i}^\gamma } \right)\;s.t.\;{C_i} = \frac{{{P_i}{Q_i}}}{P},{Q_i} = {L_i}$$
$C_i$為根據平衡點所預估當前消費數量，而在理想上勞力投入 $L$ 可以完全轉換為供給數量 $Q$
我們整理一下再取個log可以得到
$${q_i} = \left( {{p_i} - p} \right)/\left( {\gamma  - 1} \right)$$
這代表著供應商會隨著價格升高而提供出貨的意願。

然而在不完美的資訊下，我們因為很難得知真實比率$r_i$，所以我們引入了隨機的概念，同時考慮了正態分布
$${y_i}\left( z \right) = \gamma E\left( {{r_i}|{p_i}} \right)$$
其中由正態分布公式給出
$$E\left( {{r_i}|{p_i}} \right) = E\left( {{r_i}} \right) + \frac{{{\sigma _{{r_i},{p_i}}}}}{{{\sigma _{{p_i}}}^2}}\left( {{p_i} - E\left( {{p_i}} \right)} \right)$$
$$\frac{{{v_r}}}{{{v_p} + {v_r}}}\left( {{p_i} - E\left( p \right)} \right)$$
在真實的情況，我們還考慮了信號化參數$Y$，因為真實情況供給會取決於需求端而進行修正，如GDP
$$\left( {{p_i} - p} \right)/\left( {\gamma  - 1} \right) = y - \eta \left( {{p_i} - p} \right) + {z_i}$$
以上我們考慮了冪係數 $\eta$ 與係數 $z_i$，前者與觀測行為相關(如用GDP來測量)，後者與當前價格行為所決定
