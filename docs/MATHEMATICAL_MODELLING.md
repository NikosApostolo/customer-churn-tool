# Mathematical modelling

This document explains the calculations behind the app. The dashboard keeps the practical summaries; equations and technical details belong here.

For input rules and implementation details, see [Methodology](METHODOLOGY.md). For interpreting the charts, see [Reading the dashboard](READING_THE_DASHBOARD.md).

## 1. What is being predicted?

Each historical customer has an outcome $y$: 1 means they left, and 0 means they stayed. The model uses customer information $x$ to estimate

$$
p(x)=P(y=1\mid x).
$$

In plain language, this is a model estimate of how likely a customer with those recorded characteristics is to belong to the churn group. It is not a guarantee or an estimate of what an offer will change.

The supplied dataset does not define a future prediction window. This app evaluates historical classification, rather than demonstrating that it can predict departures in the next month.

| Symbol | Meaning |
|---|---|
| $x$ | Original customer fields |
| $z$ | Customer fields after preprocessing |
| $y$ | Recorded outcome: 0 stayed, 1 left |
| $p$ | Estimated churn probability |
| $F$ | Model score before conversion to probability |
| $n$ | Number of observations in the calculation being described |

## 2. Preparing the inputs

Implementation: [src/model.py](../src/model.py).

Customer IDs identify rows but are not predictors. The churn outcome is also excluded from predictors. Any missing-value replacements, scaling values, and category mappings are learned from training rows only.

### Numeric values

Missing numeric values are replaced by the training median: the middle value among the available training observations. If a numeric column is entirely missing during fitting, the configured imputer fills it with zero.

For each numeric field, the scaler computes the training mean and standard deviation:

$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i,
\qquad
s=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2},
\qquad
z_i=\frac{x_i-\mu}{s}.
$$

These calculations use the imputed training values. The same $\mu$ and $s$ are then used on validation and test rows. A field with zero variance uses a scale factor of 1.

Scaling puts numerical fields on comparable scales. This matters for logistic regression because its coefficient penalty otherwise depends strongly on the units used. Trees do not require scaling, but both pipelines use the same preprocessing design.

### Categories

A missing category becomes the text `missing`. One-hot encoding then creates a separate indicator for each training category:

$$
z_{ic}=\begin{cases}1 & \text{if customer }i\text{ has category }c,\\0 & \text{otherwise.}\end{cases}
$$

For example, contract type becomes separate columns for month-to-month, one-year, and two-year contracts. All categories are retained. An unseen category at prediction time produces zeros in that field's encoded columns instead of causing an error.

## 3. Logistic regression

Despite its name, logistic regression is used here for classification. It combines the transformed inputs into a score:

$$
F(z)=b+\sum_{j=1}^{d}\beta_jz_j.
$$

Here $b$ is the intercept, $d$ is the number of transformed inputs, and $\beta_j$ is the coefficient learned for input $j$.

The sigmoid function converts this unrestricted score into a number between 0 and 1:

$$
p(z)=\sigma(F(z))=\frac{1}{1+e^{-F(z)}}.
$$

Equivalently,

$$
\log\left(\frac{p}{1-p}\right)=F(z).
$$

The quantity $p/(1-p)$ is the odds, and its natural logarithm is called log-odds. A score of 0 corresponds to probability 0.5. An illustrative score of 1.1 corresponds to approximately 0.7503; this is a numerical example, not a fitted customer result.

### What does training minimize?

For a customer with observed outcome $y$ and predicted probability $p$, the loss is

$$
\ell(y,p)=-y\log(p)-(1-y)\log(1-p).
$$

This is negative log-likelihood, also called binary log loss. It penalizes confident mistakes heavily: predicting a probability close to zero for someone who actually left produces a large loss.

With the app's scikit-learn configuration, an equivalent training objective is

$$
J(b,\beta)=\sum_{i=1}^{n}\ell(y_i,p_i)
+\frac{1}{2C}\sum_{j=1}^{d}\beta_j^2,
\qquad C=1.
$$

The second term is L2 regularization: it discourages unnecessarily large coefficients. The intercept is not penalized by the configured `lbfgs` solver. Dividing the entire objective by $n$ gives the equivalent mean-loss form, with penalty coefficient $1/(2Cn)$.

The derivatives show how predictions guide training:

$$
\frac{\partial J}{\partial\beta_j}
=\sum_{i=1}^{n}(p_i-y_i)z_{ij}+\frac{\beta_j}{C},
\qquad
\frac{\partial J}{\partial b}
=\sum_{i=1}^{n}(p_i-y_i).
$$

The optimizer repeatedly adjusts the coefficients to reduce this objective. The app allows up to 2,000 optimization iterations; this is a limit, not a requirement to take exactly 2,000 steps.

For a scaled numeric input, increasing it by one training standard deviation multiplies the predicted odds by $e^{\beta_j}$, holding the other inputs fixed. Because all category indicators are retained, comparing categories $a$ and $b$ uses $e^{\beta_a-\beta_b}$ rather than interpreting either coefficient alone.

Logistic regression is the baseline: it provides a comparatively simple model against which to judge the more flexible tree model.

## 4. XGBoost

XGBoost builds many small decision trees. Each new tree adjusts the score produced by the previous trees:

$$
F_t(z)=F_{t-1}(z)+\eta f_t(z),
\qquad
p_t(z)=\sigma(F_t(z)).
$$

A tree routes a customer through questions about their inputs and assigns a score at the final leaf. If $q(z)$ identifies that leaf and $w_k$ is its score, then $f_t(z)=w_{q(z)}$.

The learning rate $\eta$ makes each addition smaller. The app uses 400 trees, a learning rate of 0.03, and maximum depth 2.

### Why these tree updates?

The model uses the same binary log loss, with penalties on tree complexity. For a tree with $T$ leaves, the relevant penalty is

$$
\Omega(f)=\gamma T+\frac{\lambda}{2}\sum_{k=1}^{T}w_k^2.
$$

Here $\lambda$ discourages large leaf scores, and $\gamma$ is the minimum gain penalty for adding a split. The app explicitly sets $\lambda=5$ and uses the library default $\gamma=0$ and no L1 leaf penalty.

At each boosting step, XGBoost approximates the change in loss using first and second derivatives with respect to the current raw score:

$$
g_i=p_i-y_i,
\qquad
h_i=p_i(1-p_i).
$$

The resulting approximate objective for a new tree is

$$
\sum_i\left[g_i f_t(z_i)+\frac{1}{2}h_i f_t(z_i)^2\right]+\Omega(f_t).
$$

For customers assigned to a particular leaf, define $G=\sum_i g_i$ and $H=\sum_i h_i$. The leaf score minimizing this quadratic approximation is

$$
w^*=-\frac{G}{H+\lambda}.
$$

A proposed split into left and right leaves is assessed using

$$
\text{Gain}=\frac{1}{2}\left[
\frac{G_L^2}{H_L+\lambda}
+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma.
$$

This compares the approximate objective before and after splitting. The learning rate subsequently shrinks the tree's contribution.

The configured histogram method groups values into bins to search for splits efficiently. Each tree uses a random 90% sample of training rows and 90% of columns. A fixed seed helps make runs reproducible.

The app does not run a hyperparameter search or early stopping. Both models train with fixed settings. The XGBoost settings were revised using the [model improvement study](MODEL_IMPROVEMENT.md), which also explains the evaluation limitations. Minimum child weight is 10, requiring sufficient summed Hessian in each child. XGBoost optimizes log loss; the separate model-selection criterion is validation average precision.

## 5. Keeping evaluation separate from training

The app makes a stratified random split: approximately 60% training, 20% validation, and 20% test. Stratified means each part retains approximately the original churn proportion. Rounding can slightly change the percentages.

1. Fit each preprocessing pipeline and model on training customers.
2. Compare average precision on validation customers and select the model with the higher value.
3. Report both models' results on untouched test customers.

The selected model is not refitted on the validation or test rows. Neither model uses class weights or resampling in this implementation.

This separation prevents data leakage: information from customers used for evaluation must not influence fitting or model selection. A random split still does not establish performance on future customers; that would require a suitable time-based study.

## 6. Evaluation metrics

A probability becomes a yes/no prediction using

$$
\hat y_i=\mathbb{1}(p_i\geq0.50).
$$

The symbol $\mathbb{1}$ means 1 when the condition is true and 0 otherwise. Ranking customers uses their continuous probabilities, so a customer can rank highly even when their probability is below 0.50.

### Confusion matrix

| Actual outcome | Predicted stayed | Predicted left |
|---|---|---|
| Stayed | True negative (TN) | False positive (FP) |
| Left | False negative (FN) | True positive (TP) |

A false positive is a customer incorrectly flagged as leaving. A false negative is a customer who left but was missed.

$$
\text{Precision}=\frac{TP}{TP+FP},
\qquad
\text{Recall}=\frac{TP}{TP+FN}.
$$

Precision asks: of those flagged, how many actually left? Recall asks: of those who left, how many did we flag? The implementation returns 0 when either metric's denominator is zero.

Accuracy alone can hide poor detection when most customers stay. Precision and recall expose the different types of mistakes.

### Average precision

Changing the decision threshold changes precision and recall. Average precision summarizes this ranking performance:

$$
\mathrm{AP}=\sum_k(R_k-R_{k-1})P_k.
$$

Here $P_k$ and $R_k$ are precision and recall as the threshold is lowered through distinct scores. Tied scores are processed together. This is a recall-weighted sum, not trapezoidal integration of the plotted curve.

Higher is better. The churn prevalence gives a useful reference for an uninformative ranking, although a finite random ranking need not equal that prevalence exactly. Validation AP selects the model before test results are examined.

### Brier score

$$
\mathrm{Brier}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2.
$$

This measures squared probability error; lower is better. A good Brier score does not by itself establish that every predicted probability is well calibrated. Calibration means, for example, that roughly 70% of customers receiving a 70% prediction actually churn.

### Churn-capture curve

Sort the test customers by decreasing risk. For a selected fraction $q$ between 0 and 1, take the first $m=\lceil qn\rceil$ customers:

$$
\text{Capture}(q)=
\frac{\sum_{i=1}^{m}y_{(i)}}{\sum_{i=1}^{n}y_i}.
$$

The parentheses indicate ranked order, and the ceiling rounds upward to a whole customer. The curve answers how many historical churners appear near the top of the ranking. The diagonal represents random selection in expectation. Capturing churners is not the same as preventing their departures.

## 7. SHAP: explaining one prediction

Implementation: [src/explain.py](../src/explain.py).

SHAP assigns contributions to the inputs for one customer's model score. For $d$ transformed inputs, the Shapley contribution of input $j$ is

$$
\phi_j=\sum_{S\subseteq\{1,\ldots,d\}\setminus\{j\}}
\frac{|S|!(d-|S|-1)!}{d!}
\left[v_x(S\cup\{j\})-v_x(S)\right].
$$

$S$ is a subset of inputs, and $v_x(S)$ is the explainer's reference score when those inputs are known. The weights average the extra contribution of input $j$ across possible orders of revealing the inputs. Efficient explainers avoid explicitly enumerating every subset.

The explanation reconstructs the raw model score:

$$
F(x)=\phi_0+\sum_{j=1}^{d}\phi_j,
\qquad
p(x)=\sigma\left(\phi_0+\sum_j\phi_j\right).
$$

$\phi_0$ is the reference score. Positive contributions raise the score; negative contributions lower it.

### References used by this app

- **Logistic regression:** `LinearExplainer` uses the first 100 training rows as its background. With its independent-feature explanation, $\phi_j=\beta_j(z_j-\bar z_j)$ and $\phi_0=b+\sum_j\beta_j\bar z_j$, where $\bar z_j$ is the background mean.
- **XGBoost:** `TreeExplainer` receives no explicit background and explains raw scores. With the pinned SHAP version, this uses tree-path-dependent reference information from the fitted trees. Tree coverage statistics are not necessarily simple customer counts.

These are different reference conventions, so their explanation baselines need not match. Neither method establishes what would happen if a business intervened on a customer feature. Correlated inputs can affect how contributions are allocated.

The code sums contributions from one-hot columns back into their original customer field. This preserves the total score, but it is not necessarily equivalent to recomputing Shapley values with each original field treated as a single player. The chart shows the five largest absolute contributions; the remaining contributions still belong to the complete explanation.

### Why these bars are not percentage points

Suppose an illustrative reference score is $-1$ and three contributions are $+0.8$, $+0.5$, and $-0.2$. Then

$$
F=-1+0.8+0.5-0.2=0.1,
\qquad
p=\sigma(0.1)\approx0.525.
$$

The reference probability is approximately 0.269, but a contribution of $+0.8$ does not mean an 80-percentage-point increase. Conversion through the sigmoid is nonlinear. Also, converting an average raw score to probability does not generally give the average predicted probability.

## 8. Retention scenarios

Implementation: [src/business.py](../src/business.py).

The calculator considers the selected model's ranked test customers. Define:

| Symbol | Meaning |
|---|---|
| $N$ | Number of ranked customers |
| $q$ | Percentage to contact, between 0 and 100 |
| $m=\lceil Nq/100\rceil$ | Number contacted |
| $P=\sum_{i=1}^{m}p_{(i)}$ | Expected churn count among those contacted, using model estimates |
| $s$ | Assumed fraction of otherwise-churning contacted customers whose churn is prevented |
| $C$ | Offer cost per contacted customer |
| $V$ | Value preserved for each prevented departure |

The calculations are

$$
\begin{aligned}
\text{Expected customers retained}&=Ps,\\
\text{Offer spending}&=mC,\\
\text{Avoided loss}&=PsV,\\
\text{Net value}&=PsV-mC.
\end{aligned}
$$

Every contacted customer incurs the offer cost, including customers who would have stayed. The success assumption is not an offer acceptance rate. Customer value should reflect the value the business actually preserves, ideally contribution or profit over a consistent period. Costs and values must use the same currency.

For example, contact 100 customers whose risks sum to 60. At an assumed 20% prevention rate, a customer value of 500, and an offer cost of 20:

- Expected customers retained: $60\times0.20=12$.
- Avoided loss: $12\times500=6{,}000$.
- Offer spending: $100\times20=2{,}000$.
- Estimated net value: $4{,}000$.

These are illustrative assumptions, not measured campaign results.

### When is a plan worthwhile?

For $PV>0$, the campaign's break-even prevention rate is

$$
s_{\mathrm{break-even}}=\frac{mC}{PV}.
$$

A scenario has positive estimated net value only when its assumed $s$ exceeds that rate. A break-even rate above 100% cannot be achieved within these assumptions. The implementation returns an infinite break-even value when $PV=0$; inspect the net-value formula directly in that case, since zero spending can produce zero net value.

For one additional customer, the expected contribution is

$$
psV-C.
$$

When $sV>0$, contacting that customer adds positive expected value only if

$$
p>\frac{C}{sV}.
$$

With $C=20$, $V=500$, and $s=0.15$, the threshold is approximately 0.267. A customer with risk 0.8 contributes an estimated 40, while one with risk 0.1 contributes an estimated loss of 12.5. At $s=0.03$, even a customer with risk 1 contributes a loss of 5. Retention plans are therefore not automatically worthwhile.

### Prediction is not treatment effect

The model learns associations between recorded inputs and churn. It does not learn how much an offer changes churn.

If $Y(0)$ is the outcome without an offer and $Y(1)$ the outcome with an offer, the relevant prevention effect is

$$
\tau(x)=E[Y(0)-Y(1)\mid x].
$$

The calculator effectively substitutes the assumption $\tau(x)\approx p(x)s$. The dataset does not establish that assumption, an untreated baseline risk, or an individual offer effect. A randomized retention experiment would be needed to measure campaign impact. Probability errors and changes in customer behaviour also affect the scenario.

## References

The equations above describe the implemented models and configured versions; worked examples are illustrative.

- [scikit-learn 1.5: logistic regression and its objective](https://scikit-learn.org/1.5/modules/linear_model.html#logistic-regression).
- [scikit-learn 1.5: LogisticRegression configuration](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [XGBoost: introduction to boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
- [XGBoost: model parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
- [SHAP: LinearExplainer](https://shap.readthedocs.io/en/stable/generated/shap.LinearExplainer.html).
- [SHAP: TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).
- [SHAP 0.46.0 implementation](https://github.com/shap/shap/tree/v0.46.0/shap/explainers).
