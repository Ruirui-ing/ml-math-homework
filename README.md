# 线性代数与机器学习基础作业

**主题：奇异值分解、贝叶斯推断与交叉熵**

姓名：`填写姓名`　学号：`填写学号`　班级：`填写班级`　提交日期：`填写日期`

> **案例说明**：本文的矩阵例子为自行构造的教学例子。贝叶斯部分采用“工业设备故障预测与告警复检”作为示例项目，所有样本数量、先验和成本均为明确设定的模拟数据，不是实际部署或实测结果。正式提交时，应将其对应到自己真实参与的项目，并说明数据来源与假设；不得把模拟结果表述为真实项目成果。

本文中的“贝叶斯过程”指“建立概率模型 → 指定先验 → 结合数据得到后验 → 预测 → 决策 → 用新数据继续更新”的贝叶斯推断流程。所有交叉熵数值默认使用自然对数，单位为 nat。

---

## 1. 奇异值分解：原理、过程与自设计例子

### 1.1 定义与矩阵维度

对任意实矩阵 $A\in\mathbb{R}^{m\times n}$，存在奇异值分解（Singular Value Decomposition，SVD）：

$$
A=U\Sigma V^T.
$$

其中 $U\in\mathbb{R}^{m\times m}$、$V\in\mathbb{R}^{n\times n}$ 都是正交矩阵，满足

$$
U^TU=I_m,\qquad V^TV=I_n.
$$

$\Sigma\in\mathbb{R}^{m\times n}$ 是矩形对角矩阵。令 $p=\min(m,n)$，其对角元素满足

$$
\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_p\geq 0.
$$

这些非负数称为奇异值。$U$ 的列向量是左奇异向量，$V$ 的列向量是右奇异向量。若 $\operatorname{rank}(A)=r$，则恰好有 $r$ 个正奇异值。SVD 不要求 $A$ 为方阵或可逆矩阵。[1]

从几何上看，$V^T$ 先对输入作正交坐标变换，$\Sigma$ 在各主方向上拉伸、压缩或消去分量，$U$ 再把结果转换到输出坐标系。正交变换可能包含旋转或反射，不应一律称为纯旋转。

### 1.2 从特征值分解得到 SVD 的过程

#### 第一步：计算 $A^TA$

$A^TA$ 是实对称半正定矩阵，因为对任意向量 $x$：

$$
x^TA^TAx=\|Ax\|_2^2\geq 0.
$$

因此，可以为 $A^TA$ 选择一组标准正交特征向量。

#### 第二步：求特征值和右奇异向量

求解

$$
A^TAv_i=\lambda_i v_i,\qquad i=1,\ldots,n,
$$

并将特征值从大到小排列。选取单位化且两两正交的 $v_i$，令

$$
V=[v_1,v_2,\ldots,v_n].
$$

重特征值对应的特征空间内，也应选择标准正交基，而不是任意一组未正交化的特征向量。

#### 第三步：计算奇异值

取前 $p=\min(m,n)$ 个特征值的非负平方根：

$$
\sigma_i=\sqrt{\lambda_i},\qquad i=1,\ldots,p.
$$

若 $n>m$，其余 $n-m$ 个特征值必为零。特别需要注意：**奇异值是 $A^TA$ 特征值的平方根，不是直接取这些特征值。**[1]

#### 第四步：求左奇异向量

对于正奇异值 $\sigma_i>0$，定义

$$
u_i=\frac{Av_i}{\sigma_i}.
$$

这样便有

$$
Av_i=\sigma_i u_i.
$$

这些向量两两正交且长度为 1，因为

$$
\begin{aligned}
u_i^Tu_j
&=\frac{v_i^TA^TAv_j}{\sigma_i\sigma_j}\\
&=\frac{\lambda_jv_i^Tv_j}{\sigma_i\sigma_j}\\
&=\delta_{ij}.
\end{aligned}
$$

当 $\sigma_i=0$ 时，不能使用除以 $\sigma_i$ 的公式。此时，零特征值对应的右奇异向量属于 $\ker(A)$；为了构造完整的 $U$，可在 $\ker(A^T)$ 中选择标准正交基，补足剩余 $m-r$ 列。

#### 第五步：组装并验证

组成 $U$、$\Sigma$ 和 $V$，检查

$$
U^TU=I_m,\qquad V^TV=I_n,\qquad U\Sigma V^T=A.
$$

只保留非零奇异值及其对应向量，还可写成紧致形式：

$$
A=U_r\Sigma_rV_r^T
=\sum_{i=1}^r\sigma_i u_i v_i^T,
$$

其中 $U_r$ 为 $m\times r$，$\Sigma_r$ 为 $r\times r$，$V_r$ 为 $n\times r$。

### 1.3 自设计例子：一个 $3\times 2$ 非方阵

设

$$
A=
\begin{bmatrix}
3&1\\
1&3\\
1&-1
\end{bmatrix}.
$$

选择非方阵，是为了展示完整 SVD 中 $U$、$\Sigma$ 和 $V$ 的维度差异。

#### （1）计算 $A^TA$

$$
\begin{aligned}
A^TA
&=
\begin{bmatrix}
3&1&1\\
1&3&-1
\end{bmatrix}
\begin{bmatrix}
3&1\\
1&3\\
1&-1
\end{bmatrix}\\
&=
\begin{bmatrix}
11&5\\
5&11
\end{bmatrix}.
\end{aligned}
$$

#### （2）求特征值

$$
\begin{aligned}
\det(A^TA-\lambda I)
&=(11-\lambda)^2-25\\
&=(\lambda-16)(\lambda-6)=0.
\end{aligned}
$$

因此

$$
\lambda_1=16,\qquad\lambda_2=6,
$$

从而

$$
\boxed{\sigma_1=4,\qquad\sigma_2=\sqrt6.}
$$

#### （3）求右奇异向量

当 $\lambda_1=16$ 时，解 $(A^TA-16I)v_1=0$，得到 $x=y$。归一化后

$$
v_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}.
$$

当 $\lambda_2=6$ 时，解 $(A^TA-6I)v_2=0$，得到 $x=-y$。归一化后

$$
v_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

因此

$$
V=\frac1{\sqrt2}
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}.
$$

#### （4）求左奇异向量并补全正交基

$$
Av_1=\frac1{\sqrt2}\begin{bmatrix}4\\4\\0\end{bmatrix},
\qquad
u_1=\frac{Av_1}{4}
=\frac1{\sqrt2}\begin{bmatrix}1\\1\\0\end{bmatrix}.
$$

$$
Av_2=\frac1{\sqrt2}\begin{bmatrix}2\\-2\\2\end{bmatrix},
\qquad
u_2=\frac{Av_2}{\sqrt6}
=\frac1{\sqrt3}\begin{bmatrix}1\\-1\\1\end{bmatrix}.
$$

完整的 $U$ 需要三列。取

$$
u_3=\frac1{\sqrt6}\begin{bmatrix}1\\-1\\-2\end{bmatrix}.
$$

直接计算可得

$$
u_1^Tu_3=0,\qquad u_2^Tu_3=0,\qquad\|u_3\|_2=1,
$$

同时 $A^Tu_3=0$，符合从 $\ker(A^T)$ 补全正交基的要求。

#### （5）写出完整分解

$$
U=
\begin{bmatrix}
\frac1{\sqrt2}&\frac1{\sqrt3}&\frac1{\sqrt6}\\
\frac1{\sqrt2}&-\frac1{\sqrt3}&-\frac1{\sqrt6}\\
0&\frac1{\sqrt3}&-\frac2{\sqrt6}
\end{bmatrix},
$$

$$
\Sigma=
\begin{bmatrix}
4&0\\
0&\sqrt6\\
0&0
\end{bmatrix},
\qquad
V^T=\frac1{\sqrt2}
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}.
$$

于是

$$
\boxed{A=U\Sigma V^T.}
$$

#### （6）验证结果

用秩一矩阵之和验证：

$$
4u_1v_1^T=
\begin{bmatrix}
2&2\\
2&2\\
0&0
\end{bmatrix},
$$

$$
\sqrt6u_2v_2^T=
\begin{bmatrix}
1&-1\\
-1&1\\
1&-1
\end{bmatrix}.
$$

两者相加恰好为原矩阵 $A$，分解成立。

### 1.4 从分解理解低秩近似

只保留最大的一个奇异值，可得到秩一近似

$$
A_1=4u_1v_1^T=
\begin{bmatrix}
2&2\\
2&2\\
0&0
\end{bmatrix}.
$$

它舍去了第二个奇异方向。本例可直接计算误差：

$$
\|A-A_1\|_F=\sqrt6\approx2.449490.
$$

保留的平方 Frobenius 范数比例为

$$
\frac{\sigma_1^2}{\sigma_1^2+\sigma_2^2}
=\frac{16}{22}\approx72.73\%.
$$

这个比例衡量矩阵数值能量的保留程度，**不等于预测准确率，也不能直接解释为保留了同样比例的业务信息**。在更大的矩阵上，截断 SVD 可用于低秩表示与近似重构；NumPy 官方教程展示了其在图像近似中的用法。[2]

### 1.5 手算与工程实现的区别

上述通过 $A^TA$ 求特征值的方法便于学习，但不应把它机械地当作所有数值实现的首选。对满列秩矩阵，二范数条件数满足

$$
\kappa_2(A^TA)=\kappa_2(A)^2,
$$

构造 $A^TA$ 可能放大数值精度问题。实际计算通常直接调用成熟的 SVD 实现。[3]

本例可使用：

```python
import numpy as np

A = np.array([[3., 1.], [1., 3.], [1., -1.]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
A_reconstructed = U @ np.diag(s) @ Vt

print(s)  # [4.         2.44948974]
print(np.allclose(A, A_reconstructed))  # True
```

NumPy 返回的是 `U, s, Vt`；对实矩阵，第三个返回值是 $V^T$，不是 $V$。`full_matrices=False` 保留 $\min(m,n)$ 个方向，秩亏时不一定等于只保留正奇异值的紧致 SVD。[1]

库函数给出的奇异向量可能与手算结果整体相差一个负号；同一个奇异值对应的左右向量同时变号，不影响分解。重奇异值对应子空间内也存在基的选择自由。因此，应检查正交性、奇异值与重构结果，而不是逐元素要求奇异向量与某一种手算结果相同。

---

## 2. 贝叶斯推断：设备故障预测与告警复检案例

### 2.1 从条件概率得到贝叶斯公式

对事件 $H$ 和证据 $E$，当 $P(E)>0$ 时，联合概率可以从两个方向展开：

$$
P(H,E)=P(E\mid H)P(H)=P(H\mid E)P(E).
$$

所以

$$
\boxed{P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}.}
$$

这里 $P(H)$ 是先验，$P(E\mid H)$ 是给定假设时观察到证据的概率，$P(E)$ 是归一化所需的证据概率，$P(H\mid E)$ 是后验。公式的核心是根据观测更新判断，而不是把两个方向的条件概率混为一谈。[4]

对连续未知参数 $\theta$ 和数据 $\mathcal D$，相应形式为

$$
p(\theta\mid\mathcal D)
=\frac{p(\mathcal D\mid\theta)p(\theta)}
{\int p(\mathcal D\mid t)p(t)\,dt}.
$$

后验预测还需考虑参数不确定性：

$$
p(y_*\mid\mathcal D)
=\int p(y_*\mid\theta)p(\theta\mid\mathcal D)\,d\theta.
$$

这里的小写 $p(\theta\mid\mathcal D)$ 是密度；某个连续参数精确等于一个数值的概率，不能直接由密度值代替。

### 2.2 项目问题、样本与建模假设

本例设计一个工业设备故障预测项目：在观察窗口开始时，传感器系统可能发出告警；希望判断设备在随后 24 小时内发生故障的概率，并据此安排复检。

每条样本是“某设备的一段 24 小时观察窗口”。定义

$$
Y_i=
\begin{cases}
1,&\text{窗口内经核实发生故障},\\
0,&\text{窗口内未发生故障},
\end{cases}
$$

$$
E_i=
\begin{cases}
1,&\text{窗口开始时出现告警},\\
0,&\text{窗口开始时未出现告警}.
\end{cases}
$$

未知参数 $\theta$ 表示目标设备群体在一个观察窗口内的基准故障概率。

为得到可手算的模型，假设样本来自同型号、相近工况的设备；在给定 $\theta$ 后，故障标签近似独立同分布；所分析阶段的故障机制基本稳定。实际采样应避免重复或重叠窗口被当成独立证据，并检查同一设备记录的相关性。

设定如下教学数据：

| 项目 | 设定 | 作用 |
|---|---:|---|
| 先验平均故障率 | 2% | 表达观测前的判断 |
| 先验强度 | 100 | 决定先验相对新数据的权重 |
| 新观测窗口数 $n$ | 100 | 用于更新基准故障率 |
| 核实故障数 $s$ | 8 | 真正的故障标签，不是告警数量 |
| 检出率 $t=P(E=1\mid Y=1)$ | 0.90 | 故障窗口中出现告警的概率 |
| 误报率 $f=P(E=1\mid Y=0)$ | 0.05 | 无故障窗口中出现告警的概率 |

$t=0.90$ 和 $f=0.05$ 可以对应一组独立模拟验证数据：100 个故障窗口中有 90 个告警，1000 个无故障窗口中有 50 个告警。这种按类别构造的验证集只用于说明条件概率，不能用其中的类别比例直接估计部署群体的故障率。

本例将 $t$ 和 $f$ 暂按已知常数处理，只对 $\theta$ 的不确定性进行贝叶斯建模。所有数值均为教学设定，未声称来自真实设备。

### 2.3 建立观测模型与似然

在给定 $\theta$ 后，故障标签服从伯努利分布：

$$
Y_i\mid\theta\sim\operatorname{Bernoulli}(\theta),
$$

$$
P(Y_i=y_i\mid\theta)=\theta^{y_i}(1-\theta)^{1-y_i}.
$$

对于 $\mathcal D=(y_1,\ldots,y_n)$，记 $s=\sum_{i=1}^n y_i$，条件独立假设给出

$$
\begin{aligned}
p(\mathcal D\mid\theta)
&=\prod_{i=1}^n\theta^{y_i}(1-\theta)^{1-y_i}\\
&=\theta^s(1-\theta)^{n-s}.
\end{aligned}
$$

代入本例数据：

$$
p(\mathcal D\mid\theta)=\theta^8(1-\theta)^{92}.
$$

若只记录“100 次中有 8 次故障”，则计数似然为 $\binom{100}{8}\theta^8(1-\theta)^{92}$。组合数不依赖 $\theta$，因此不会改变下面的后验分布。

### 2.4 选择 Beta 先验并解释参数

选择

$$
\theta\sim\operatorname{Beta}(\alpha,\beta),
$$

其密度为

$$
p(\theta)=\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},
\quad0<\theta<1,
$$

其中

$$
B(\alpha,\beta)=\int_0^1 t^{\alpha-1}(1-t)^{\beta-1}\,dt.
$$

Beta 分布的定义域与概率参数一致，并且与伯努利／二项模型共轭，便于把故障与正常样本分别加入参数。[5][6]

利用先验均值和强度确定参数：

$$
\frac{\alpha}{\alpha+\beta}=0.02,
\qquad\alpha+\beta=100.
$$

得到

$$
\boxed{\theta\sim\operatorname{Beta}(2,98).}
$$

这里的强度 100 是用于表达先验权重的设定，不是本次实际采集的 100 条数据；不能用同一批数据既构造先验又计算似然，造成重复计数。真实项目应说明先验来自哪些更早的数据或专家判断，并做先验敏感性分析。

### 2.5 推导后验分布

由贝叶斯公式：

$$
\begin{aligned}
p(\theta\mid\mathcal D)
&\propto p(\mathcal D\mid\theta)p(\theta)\\
&\propto\theta^s(1-\theta)^{n-s}
\theta^{\alpha-1}(1-\theta)^{\beta-1}\\
&=\theta^{\alpha+s-1}(1-\theta)^{\beta+n-s-1}.
\end{aligned}
$$

归一化后可识别为

$$
\boxed{\theta\mid\mathcal D\sim
\operatorname{Beta}(\alpha+s,\beta+n-s).}
$$

所以本例为

$$
\boxed{\theta\mid\mathcal D\sim\operatorname{Beta}(10,190).}
$$

其完整密度为

$$
p(\theta\mid\mathcal D)
=\frac{\theta^9(1-\theta)^{189}}{B(10,190)}.
$$

后验不仅提供一个估计数值，还保留了对 $\theta$ 的不确定性。[6]

### 2.6 后验均值、可信区间与预测

后验均值为

$$
\mathbb E[\theta\mid\mathcal D]
=\frac{10}{10+190}=0.05.
$$

也可以写成先验均值与样本故障率的加权平均：

$$
\begin{aligned}
\frac{\alpha+s}{\alpha+\beta+n}
&=\frac{\alpha+\beta}{\alpha+\beta+n}
\frac{\alpha}{\alpha+\beta}
+\frac{n}{\alpha+\beta+n}\frac{s}{n}\\
&=\frac{100}{200}\times0.02
+\frac{100}{200}\times0.08\\
&=0.05.
\end{aligned}
$$

因此，新数据将平均故障率从 2% 修正为 5%，但没有直接等同于样本故障率 8%。

令 $F$ 为 $\operatorname{Beta}(10,190)$ 的累积分布函数，95% 等尾可信区间为

$$
[F^{-1}(0.025),F^{-1}(0.975)]
\approx[0.02435727,0.08411464].
$$

这些分位数由 `scipy.stats.beta.ppf` 数值计算。[7] 在所选先验、模型与数据条件下，$\theta$ 落在该区间内的后验概率为 95%；它不是单台设备“是否故障”的区间，也不应与频率学派置信区间的重复抽样解释混用。

对一个尚未观察告警信息的新窗口，预测故障概率为

$$
\begin{aligned}
P(Y_*=1\mid\mathcal D)
&=\int_0^1\theta p(\theta\mid\mathcal D)\,d\theta\\
&=\mathbb E[\theta\mid\mathcal D]\\
&=0.05.
\end{aligned}
$$

本例积分恰好等于后验均值，是伯努利模型的结构所致；不能把“将参数替换为后验均值”等同于所有模型中的完整后验预测。

### 2.7 收到告警后再次更新判断

设新窗口出现告警，即 $E_*=1$。将参数不确定性积分掉：

$$
P(Y_*=1,E_*=1\mid\mathcal D)
=\int_0^1t\theta p(\theta\mid\mathcal D)\,d\theta
=t\mu,
$$

其中 $\mu=\mathbb E[\theta\mid\mathcal D]=0.05$。

同理，告警的总概率为

$$
P(E_*=1\mid\mathcal D)
=t\mu+f(1-\mu).
$$

因此

$$
\begin{aligned}
r&=P(Y_*=1\mid E_*=1,\mathcal D)\\
&=\frac{t\mu}{t\mu+f(1-\mu)}\\
&=\frac{0.90\times0.05}
{0.90\times0.05+0.05\times0.95}\\
&=\frac{18}{37}\\
&\approx0.486486.
\end{aligned}
$$

**结论：未考虑告警时的预测故障率为 5%；出现告警后变为约 48.65%，而不是 90%。**

90% 是 $P(E=1\mid Y=1)$，而需要计算的是 $P(Y=1\mid E=1,\mathcal D)$。两者方向不同，后者还取决于基准故障率与误报率。

用赔率形式检查同一结果：

$$
\underbrace{\frac{r}{1-r}}_{\text{告警后赔率}}
=
\underbrace{\frac{0.05}{0.95}}_{\text{告警前赔率}}
\times
\underbrace{\frac{0.90}{0.05}}_{\text{似然比}}
=\frac{18}{19}.
$$

注意，这里通过“联合概率的积分 ÷ 告警概率的积分”进行条件化；并不是直接在 $p(\theta\mid\mathcal D)$ 下平均各个 $P(Y_*=1\mid E_*=1,\theta)$。

### 2.8 从概率预测到复检决策

概率模型不直接决定行动，还需要损失函数。设 $a_1$ 表示复检，$a_0$ 表示不复检，使用如下**教学相对损失**：

| 决策 | 实际无故障 $Y=0$ | 实际有故障 $Y=1$ |
|---|---:|---:|
| 不复检 $a_0$ | 0 | 1000 |
| 复检 $a_1$ | 100 | 0 |

这里将正确决策的损失归一化为 0，只比较不必要复检与漏检的相对损失；不是声称真实复检没有成本或能完全消除故障。真实部署应替换为完整的业务成本表。

给定后验故障概率 $r$，两种行动的期望损失为

$$
R(a_1)=100(1-r),\qquad R(a_0)=1000r.
$$

当 $R(a_1)<R(a_0)$ 时选择复检：

$$
100(1-r)<1000r
\quad\Longleftrightarrow\quad
r>\frac{100}{1100}\approx0.090909.
$$

本例中

$$
R(a_1)\approx51.35,
\qquad R(a_0)\approx486.49.
$$

因此，在该假设损失模型下选择复检。即使故障概率未超过 50%，也可能需要复检：**行动阈值由误报与漏报的相对损失决定，不必固定为 0.5。**

### 2.9 持续更新与项目验证

假设之后获得另外 50 个已核实的窗口，其中 2 个发生故障，则上一轮后验可作为下一轮先验：

$$
\operatorname{Beta}(10,190)
\longrightarrow
\operatorname{Beta}(10+2,190+48)
=\operatorname{Beta}(12,238).
$$

新的后验均值为

$$
\frac{12}{250}=0.048.
$$

在同一固定参数模型下，分批更新与一次使用全部数据相同。这里只能加入新得到的真实标签，不能把“模型预测为故障”当作已经核实的故障。

项目验证应使用未参与建模的后续时间段，并检查同一设备的相关记录是否跨越训练和测试边界。需要评估概率预测的交叉熵、告警的误报与漏报，以及设定成本下的总损失；当前模拟推导并不提供任何真实项目准确率。

本模型的主要局限是统一基准故障率、近似独立假设以及将 $t,f$ 当作常数。不同设备型号、工况变化或设备老化可能需要分组／分层模型或随时间变化的模型；告警器参数也可进一步设置 Beta 先验。还应检查先验强度变化对结论的影响，而不是默认某个先验一定正确。

---

## 3. 交叉熵：定义、原理、应用与例子推导

### 3.1 从信息量到交叉熵

设某个离散事件的概率为 $q_i$，其自信息为

$$
I(i)=-\log q_i.
$$

事件越不可能发生，真正发生时的信息量越大。若真实分布为 $P=(p_1,\ldots,p_K)$，熵定义为

$$
H(P)=-\sum_{i=1}^Kp_i\log p_i.
$$

若事件仍由 $P$ 产生，但使用预测分布 $Q=(q_1,\ldots,q_K)$ 评价其信息量，则平均信息量为

$$
\boxed{H(P,Q)=-\sum_{i=1}^Kp_i\log q_i.}
$$

这就是交叉熵；它也可理解为真实分布下，模型所给负对数概率的期望。[8]

$P,Q$ 均需为概率分布，即各分量非负、总和为 1。对离散交叉熵，采用以下约定：$p_i=0$ 的项贡献为 0；若存在 $p_i>0$ 而 $q_i=0$，交叉熵为正无穷。使用自然对数时单位为 nat，使用以 2 为底的对数时单位为 bit。

### 3.2 交叉熵与 KL 散度的关系

KL 散度定义为

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_i p_i\log\frac{p_i}{q_i}.
$$

拆开对数可得

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\sum_i p_i\log p_i-\sum_i p_i\log q_i\\
&=-H(P)+H(P,Q).
\end{aligned}
$$

因此

$$
\boxed{H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q).}
$$

KL 散度非负，并在 $P=Q$ 时为零。[9] 还可以用 $\log t\leq t-1$ 验证非负性：当 $Q$ 在 $P$ 的支持集上为正时，

$$
\begin{aligned}
-D_{\mathrm{KL}}(P\|Q)
&=\sum_{i:p_i>0}p_i\log\frac{q_i}{p_i}\\
&\leq\sum_{i:p_i>0}(q_i-p_i)\\
&\leq0.
\end{aligned}
$$

于是，在真实分布 $P$ 固定时：

$$
H(P,Q)\geq H(P),
$$

最小交叉熵等于 $H(P)$，而**不一定等于 0**。只有目标分布的熵为零，例如独热标签，理论最小值才是 0。交叉熵通常也不对称，所以不能把它当成满足全部距离公理的严格距离。

#### 分布层面的数值例子

令

$$
P=(0.7,0.2,0.1),\qquad Q=(0.6,0.3,0.1).
$$

则

$$
\begin{aligned}
H(P,Q)
&=-0.7\ln0.6-0.2\ln0.3-0.1\ln0.1\\
&\approx0.828631,
\end{aligned}
$$

而

$$
H(P)\approx0.801819,
\qquad D_{\mathrm{KL}}(P\|Q)\approx0.026812.
$$

两者相加为 $0.828631$，验证了上述恒等式。当 $Q=P$ 时，交叉熵下降到 $0.801819$，仍不是 0。

### 3.3 从最大似然推导二分类交叉熵

设样本为 $(x_i,y_i)$，其中 $y_i\in\{0,1\}$。模型输出

$$
q_i=P(y_i=1\mid x_i;w).
$$

单个标签在模型下的概率为

$$
p(y_i\mid x_i;w)=q_i^{y_i}(1-q_i)^{1-y_i}.
$$

若样本在给定参数后条件独立，则似然为

$$
L(w)=\prod_{i=1}^Nq_i^{y_i}(1-q_i)^{1-y_i}.
$$

最大化似然等价于最小化负对数似然。取平均后得到

$$
\begin{aligned}
J(w)
&=-\frac1N\log L(w)\\
&=-\frac1N\sum_{i=1}^N
\left[y_i\log q_i+(1-y_i)\log(1-q_i)\right].
\end{aligned}
$$

这就是平均二分类交叉熵（Binary Cross-Entropy，BCE）。因此，BCE 并非任意拼出的误差函数，而是伯努利概率模型的平均负对数似然。[10]

#### 四个样本的计算例子

仍以设备故障标签为例，设

$$
y=(1,0,1,0),\qquad q=(0.8,0.3,0.6,0.1).
$$

分别计算：

| 样本 | 真实标签 $y_i$ | 预测故障概率 $q_i$ | 单样本损失 |
|---|---:|---:|---:|
| 1 | 1 | 0.8 | $-\ln0.8\approx0.223144$ |
| 2 | 0 | 0.3 | $-\ln0.7\approx0.356675$ |
| 3 | 1 | 0.6 | $-\ln0.6\approx0.510826$ |
| 4 | 0 | 0.1 | $-\ln0.9\approx0.105361$ |

平均交叉熵为

$$
\boxed{J\approx0.299001.}
$$

对于真实故障样本 $y=1$，模型给出 $q=0.9$ 时，损失为 $-\ln0.9\approx0.105361$；错误地给出 $q=0.01$ 时，损失为 $-\ln0.01\approx4.605170$。因此，交叉熵会强烈惩罚“非常自信但错误”的概率预测。

### 3.4 Sigmoid 与 BCE 的梯度

令 $z=w^Tx+b$，并令

$$
q=\operatorname{sigmoid}(z)=\frac1{1+e^{-z}}.
$$

对单个样本，

$$
\ell=-y\log q-(1-y)\log(1-q).
$$

先对 $q$ 求导：

$$
\frac{\partial\ell}{\partial q}
=-\frac yq+\frac{1-y}{1-q}
=\frac{q-y}{q(1-q)}.
$$

利用

$$
\frac{\partial q}{\partial z}=q(1-q),
$$

得到

$$
\boxed{\frac{\partial\ell}{\partial z}=q-y.}
$$

继续使用链式法则：

$$
\nabla_w\ell=(q-y)x,
\qquad\frac{\partial\ell}{\partial b}=q-y.
$$

该结果表明，预测概率高于标签时，梯度下降会倾向于降低相应得分；预测概率低于标签时，则倾向于提高得分。

### 3.5 多分类：Softmax、交叉熵与完整推导例子

对互斥的 $K$ 类问题，模型输出未归一化得分 $z_1,\ldots,z_K$，通过 Softmax 转换为概率：

$$
q_j=\frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}.
$$

若标签为独热向量 $y$，单样本多分类交叉熵为

$$
\ell=-\sum_{j=1}^K y_j\log q_j.
$$

设真实类别为 $c$，则 $y_c=1$、其余为 0，故

$$
\ell=-\log q_c
=-z_c+\log\sum_{j=1}^Ke^{z_j}.
$$

对任意 $z_k$ 求导：

$$
\begin{aligned}
\frac{\partial\ell}{\partial z_k}
&=-\mathbf1(k=c)+\frac{e^{z_k}}{\sum_j e^{z_j}}\\
&=q_k-y_k.
\end{aligned}
$$

即

$$
\boxed{\nabla_z\ell=q-y.}
$$

Softmax 与交叉熵结合后的导数，恰好是预测概率与目标概率之差。[8]

#### 数值计算与一次梯度更新

假设三个互斥类别是“正常”“轴承故障”“过热”，模型输出

$$
z=(2,1,0.1),\qquad y=(1,0,0).
$$

Softmax 的分母为

$$
e^2+e^1+e^{0.1}\approx11.212509.
$$

因此

$$
q\approx(0.659001,0.242433,0.098566).
$$

交叉熵为

$$
\ell=-\ln0.659001\approx0.417030.
$$

对得分的梯度为

$$
\nabla_z\ell=q-y
\approx(-0.340999,0.242433,0.098566).
$$

为了直接观察梯度作用，暂把 $z$ 当作待优化变量，使用学习率 $\eta=0.1$ 更新一次：

$$
\begin{aligned}
z'&=z-0.1(q-y)\\
&\approx(2.034100,0.975757,0.090143).
\end{aligned}
$$

重新计算得到

$$
\ell'\approx0.398888<0.417030.
$$

这次更新提高了真实类别的得分，并降低其他类别的得分，损失随之下降。实际神经网络训练会通过链式法则把该梯度传到网络参数，而不是一般地把各样本的得分当作彼此独立的参数。

### 3.6 应用与实现注意事项

二分类任务，例如设备是否故障，可以采用一个 Sigmoid 输出与 BCE；互斥多分类任务，例如从多种状态中选一种，可以采用 Softmax 与多分类交叉熵。对下一词预测，也可以把词表中的每个候选词当作类别，用真实词的负对数预测概率训练模型。若一个样本可以同时具有多个标签，不能机械套用只表达互斥类别的单个 Softmax，通常应分别建模各标签的二分类概率。

计算时应避免直接计算巨大指数。令 $m=\max_j z_j$，则

$$
\log q_j
=(z_j-m)-\log\sum_k e^{z_k-m}.
$$

直接由稳定的对数概率计算交叉熵，比先得到可能下溢为零的概率再取对数更合适。BCE 也可直接从得分稳定计算：

$$
\ell(z,y)=\max(z,0)-yz+\log(1+e^{-|z|}).
$$

交叉熵适合比较**同一目标任务与同一评估数据**上的概率预测，并不意味着损失较低就一定在所有阈值下拥有更高的分类准确率。类别不平衡、概率校准与业务成本仍需要单独检查。对当前设备案例，评估时必须将预测概率与后续核实的故障标签比较，而不是与模型自己的告警结果比较。

### 3.7 与贝叶斯部分的联系：MLE、MAP 和完整后验

在第 2 节中，只利用 100 个窗口的标签时，负对数似然为

$$
-\log p(\mathcal D\mid\theta)
=-8\log\theta-92\log(1-\theta).
$$

最小化它，等价于最小化这些样本的 BCE，得到最大似然估计

$$
\hat\theta_{\mathrm{MLE}}=\frac8{100}=0.08.
$$

若再加入 $\operatorname{Beta}(2,98)$ 先验，最大后验估计（MAP）对应的目标为

$$
\begin{aligned}
J_{\mathrm{MAP}}(\theta)
&=-\log p(\mathcal D\mid\theta)-\log p(\theta)\\
&=-9\log\theta-189\log(1-\theta)+C.
\end{aligned}
$$

求导并令其为零：

$$
-\frac9\theta+\frac{189}{1-\theta}=0
\quad\Longrightarrow\quad
\hat\theta_{\mathrm{MAP}}=\frac9{198}\approx0.045455.
$$

因此，本例应区分三个对象：

$$
\hat\theta_{\mathrm{MLE}}=8\%,
\qquad
\hat\theta_{\mathrm{MAP}}\approx4.5455\%,
\qquad
\mathbb E[\theta\mid\mathcal D]=5\%.
$$

MAP 是后验密度的众数，后验均值是另一种汇总方式；完整贝叶斯推断保留的是 $\operatorname{Beta}(10,190)$ 这一分布。**最大似然等价于最小化相应交叉熵，但单纯最小化交叉熵不自动等于完整的贝叶斯推断。**

---

## 4. Markdown 仓库组织、复现与提交

### 4.1 文件结构

```text
ml_math_homework/
├── README.md          # 作业正文：三部分完整推导
├── SUBMISSION.md      # 提交学习委员的信息模板
├── results.md         # 已运行的数值验证结果
├── requirements.txt   # 数值验证依赖
├── .gitignore
└── scripts/
    └── verify.py      # SVD、贝叶斯与交叉熵的核验程序
```

全部作业正文与结果报告均使用 Markdown；Python 文件是补充的可复现实验，不代替数学推导。

### 4.2 运行数值验证

本次核验环境为 Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0。安装依赖后，在项目目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/verify.py
```

需要重新生成结果报告时执行：

```bash
python scripts/verify.py > results.md
```

脚本在输出报告前会检查正交性、矩阵重构、后验更新、可信区间分位数、告警概率积分、交叉熵恒等式，以及 Softmax 交叉熵的有限差分梯度。其结果不依赖随机采样。

### 4.3 推送到自己的远程仓库

**当前交付的是本地仓库文件包，尚未推送到任何远程仓库。**

先在自己的 Git 托管平台创建空仓库，并准备对应的登录或 SSH 身份验证。下面的命令用于“本地新目录 + 远程空仓库”的场景；运行前必须替换用户名、邮箱与远程地址。GitHub 官方文档给出了初始化本地仓库、添加远程地址与推送的流程。[11]

```bash
cd ml_math_homework

git init
git config user.name "替换为你的姓名或Git用户名"
git config user.email "替换为你的Git提交邮箱"

git add README.md SUBMISSION.md results.md requirements.txt .gitignore scripts/verify.py
git commit -m "完成SVD、贝叶斯推断与交叉熵作业"
git branch -M main

git remote add origin "替换为你的远程仓库地址"
git remote -v
git push -u origin main
```

若已有带提交历史的远程仓库，应先克隆该仓库，再把作业文件复制进去提交，避免无意覆盖已有历史。不要为解决普通推送冲突而直接使用强制推送。不要把密码、访问令牌或 `.env` 文件提交到仓库。

### 4.4 发给学习委员的信息

将 `SUBMISSION.md` 中的姓名、学号、班级、仓库主页地址与访问方式填写完整，再把仓库地址交给学习委员。提交前应在网页端确认最新提交和公式显示正常，并确保学习委员及老师具有访问权限；不要只提供本地路径或只有自己能够打开的私有链接。

正式提交前，还需要把示例项目明确对应到自己的实际项目，检查所有模拟数据与真实数据的标记，并完成本文顶部的个人信息。

---

## 参考资料

以下资料用于支持概念、公式和软件接口；本文矩阵、模拟业务数据、成本设置及相应数值计算为本文构造。

1. NumPy. [numpy.linalg.svd 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).
2. NumPy Tutorials. [Linear algebra on n-dimensional arrays](https://numpy.org/numpy-tutorials/tutorial-svd/).
3. Driscoll 与 Braun. *Fundamentals of Numerical Computation*. [The normal equations](https://tobydriscoll.net/fnc-julia/leastsq/normaleqns.html).
4. Chris Piech，Stanford University. *Probability for Computer Science*. [Bayes' Theorem](https://chrispiech.github.io/probabilityForComputerScientists/en/part1/bayes_theorem/).
5. Chris Piech，Stanford University. *Probability for Computer Science*. [Beta Distribution](https://chrispiech.github.io/probabilityForComputerScientists/en/part4/beta/).
6. Johnson、Ott 与 Dogucu. *Bayes Rules!*. [Chapter 3: The Beta-Binomial Bayesian Model](https://www.bayesrulesbook.com/chapter-3).
7. SciPy. [scipy.stats.beta 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html).
8. Zhang 等. *Dive into Deep Learning*. [Softmax Regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html).
9. Zhang 等. *Dive into Deep Learning*. [Information Theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html).
10. scikit-learn. [log_loss 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
11. GitHub Docs. [Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
