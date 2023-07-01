# Phillips曲線(Phillips Curve)
Phillips曲線是指薪資增長率與失業率的曲線，由於很難衡量通貨膨脹，通常是以薪資增長率作為通絡膨脹的指標。
以下來介紹一下相關的金融模型：

# Lucas islands model
在理想封閉狀態裡面，市場流通的總金額數應該是不變的，但假如總流通的金額變多
是否會影響到市場的行為？
在介紹這個模型前，我們先提到兩個專有名詞
## 需求衝擊(Demand shocks)
需求衝擊是指當人們對商品或服務的需求改變時，造成價格行為改變的事件
如珍珠奶茶經由新聞傳播與廣告宣傳，造成人們在某段時期特別想喝它
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
假設存在著N個島嶼，在其中一個島嶼z，則當前的供給 $q_i$ 可由係數 $\gamma$ 
$${q_i} = \left( {{p_i} - p} \right)/\left( {\gamma  - 1} \right) = {r_i}/\left( {\gamma  - 1} \right)$$
與現價 $p_i$ 和均價 $p$ 所決定 (即真實比率 $r_i$)
同時也是意味著，當現價大於均價時，供給者會比較樂意提高他們的供應量。同時，這個現在也考慮了因為總流通量而變高的因素。
接著我們來看看他是怎麼來的
註：不同的島嶼意味著對於一個環境不同的測量，為了方便說明我們在此寫的是$q_i$，尚未進行訊號化
實際上我們考慮的是
$${y_i}\left( z \right) = \gamma \left( {{p_i}\left( z \right) - p} \right) + {z_i} = \gamma r + {z_i}$$
這個 $z_i$ 將訊號化所導致的修正都考慮進去
### 效用函數(utility function)
我們考慮一個很常在經濟學使用的效用函數
$$\max \left( {{C_i} - \frac{1}{\gamma }{L_i}^\gamma } \right)\;s.t.\;{C_i} = \frac{{{P_i}{Q_i}}}{P},{Q_i} = {L_i}$$
$C_i$為根據平衡點所預估當前消費數量，而在理想上勞力投入 $L$ 可以完全轉換為供給數量 $Q$
我們整理一下再取個log可以得到
$${q_i} = \left( {{p_i} - p} \right)/\left( {\gamma  - 1} \right)$$
這代表著供應商會隨著價格升高而提供出貨的意願。
$E\left( {{r_i}|{p_i}} \right) = \frac{{{v_r}}}{{{v_p} + {v_r}}}\left( {{p_i} - E\left( p \right)} \right)$
然而在不完美的資訊下，我們因為很難得知真實比率 $r_i$，所以我們引入了隨機的概念，同時考慮了正態分布
$${q_i}\left( z \right) = \gamma E\left( {{r_i}|{p_i}} \right)$$
其中由正態分布公式給出
$$E\left( {{r_i}|{p_i}} \right) = E\left( {{r_i}} \right) + \frac{{{\sigma _{{r_i},{p_i}}}}}{{{\sigma _{{p_i}}}^2}}\left( {{p_i} - E\left( {{p_i}} \right)} \right)$$
其中 $E$代表期望值， $\sigma$ 代表標準差
$$\frac{{{v_r}}}{{{v_p} + {v_r}}}\left( {{p_i} - E\left( p \right)} \right)$$
其中 $E\left( {{r_i}} \right)$ 為0(已經取了log)， $v$為變異數, 其中我們做了一個假設
$${\mathop{\rm cov}} \left( {{r_i},p} \right) = 0$$
真實比率 $r_i$應該與價格變化 $p_i$無關
$${q_i} = \gamma \frac{{{v_r}}}{{{v_p} + {v_r}}}\left( {{p_i} - E\left( p \right)} \right)$$
$\frac{{{v_r}}}{{{v_p} + {v_r}}}$ 稱為信號強度(strength of signal)
假如 ${v_r} >  > {v_p}$ 代表 真實價格的變化越大 真實比率的條件期望值越接近 $p_i-E(p_i)$
假如 ${v_p} >  > {v_r}$ 代表 價格浮動的變化遠大於真實價格的變化 真實比率的條件期望值越低於 $p_i-E(p_i)$

##信號化參數 $Y$
在真實的情況，我們還考慮了信號化參數 $Y$，因為真實情況供給會取決於需求端而進行修正，如GDP
$$\left( {{p_i} - p} \right)/\left( {\gamma  - 1} \right) = y - \eta \left( {{p_i} - p} \right) + {z_i}$$
以上我們考慮了冪係數 $\eta$ 與係數 $z_i$，前者與觀測行為相關(如用GDP來測量)，後者由當前價格行為所決定

## Lucas供給曲線(Lucas supply curve)
### 總量供給曲線(Aggregate Supply Curve)
我們把信號化參數的條目的結果結合前面
$$y = (\frac{1}{{\gamma  - 1}} + \eta )\left( {{p_i} - p} \right) - {z_i} = b\left( {{p_i} - p} \right) - {z_i}$$
### 總量需求曲線(Aggregate Demand Curve)
此處我們使用最簡單的線性關係
$$y=m-p$$
平衡點即兩式相等
$${p_e} = \frac{1}{{1 + b}}m + \frac{b}{{1 + b}}E\left( m \right)$$
$${y_e} = \frac{b}{{1 + b}}m - \frac{b}{{1 + b}}E\left( m \right)$$
其中使用了 $E(m)=E(p)$ ，我們也可以解釋為什麼Phillips曲線是成負相關的
