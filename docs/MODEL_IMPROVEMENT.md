# Model improvement study

## Browser profiles

The browser app uses separate built-in profiles for the IBM Telco and Churn Prediction 2020 schemas. The Churn Prediction 2020 profile now uses a deeper XGBoost model. The IBM profile uses the shallower configuration from the earlier study below.

The comparison used 7,043 labeled IBM Telco rows and 4,250 labeled Churn Prediction 2020 rows. For each dataset, the published test rows were excluded from model selection. On the remaining rows, four outer stratified folds evaluated the selection process. Three inner folds chose among the current depth-2 model and variants with depths 3–6. Imputation and encoding were fitted within each training fold. Selection used average precision (AP), which measures how well the model ranks customers who actually left. Brier score measures probability error. Lower is better.

| Built-in profile | Mean outer-fold AP, old → selected | Mean outer-fold Brier, old → selected | Decision |
| --- | ---: | ---: | --- |
| IBM Telco | 0.6714 → 0.6714 | 0.1332 → 0.1332 | Keep depth 2 |
| Churn Prediction 2020 | 0.7788 → 0.8793 | 0.0590 → 0.0362 | Use depth 6 |

The depth-6 variant won on all four Churn Prediction 2020 outer folds. After the choice was fixed, it was fitted on the same training rows as the previous browser profile and checked on the existing 850-row test split:

| Churn Prediction 2020 test metric | Previous | Updated |
| --- | ---: | ---: |
| Average precision | 0.7746 | 0.8554 |
| Brier score | 0.0618 | 0.0376 |
| Churners found among the highest-risk 20% of customers | 79.2% | 83.3% |
| Actual churners flagged at 50% risk | 55.0% | 73.3% |

An additional sigmoid calibration step failed to improve mean Brier score. The exported profile uses the model's raw probability estimates. The [profile exporter](../web/tools/export_profiles.py) records the exact configuration. These comparisons reuse labeled data from the same datasets and do not establish performance on later customers. The separate 750-row Churn Prediction 2020 scoring file was used for inference checks because it has no labels. The app's browser-trained models for custom historical uploads use a separate training path.

## Earlier Streamlit study

### Conclusion

A smaller, more strongly constrained XGBoost model produced a modest and reasonably consistent improvement on the IBM Telco sample. This configuration is used by the original Streamlit app and the built-in IBM browser profile.

Average precision (AP) measures ranking quality across decision thresholds. An AP of 0.66 does not mean 66% accuracy or that every flagged customer has a 66% chance of leaving. The sample churn rate is approximately 26.5%, a useful reference for an uninformative ranking.

### What changed

| Setting | Previous | Updated |
|---|---:|---:|
| Maximum tree depth | 3 | 2 |
| Number of trees | 180 | 400 |
| Learning rate | 0.05 | 0.03 |
| L2 leaf penalty | 2 | 5 |
| Minimum child weight | 1 (default) | 10 |

Shallower trees make fewer chained decisions. Stronger leaf penalties and a larger minimum child weight discourage small, unstable partitions. More trees with smaller updates allow gradual learning. Minimum child weight is a threshold on summed second derivatives, not a literal minimum number of customers.

The inputs, cleaning, 60/20/20 app split, logistic baseline, classification threshold, SHAP implementation, business formula, and interface are unchanged. No new dependencies are required.

### Evaluation protocol

The 1,409 customers in the already-published test split were excluded from model development. The remaining 5,634 customers were used for the experiment.

Five outer stratified folds evaluated model performance. Within each outer training group, three inner folds selected among five fixed candidates:

- Existing XGBoost.
- Shallower XGBoost with the updated settings above.
- XGBoost with depth 3, 400 trees, learning rate 0.03, L2 penalty 10, and minimum child weight 10.
- Logistic regression with C=1.
- Logistic regression with C=0.1.

All other XGBoost settings match the app. Every fit learns its imputation, scaling, and category encoding from that fit's training rows only. AP is the selection metric. Brier score is reported as a complementary probability-error measure, not a second tuning target.

The outer split seed is 2026; inner split seed is 101. The original published-test exclusion uses seed 42, as does model fitting. The search was kept deliberately small. There was no search over random seeds to find a flattering result.

Nested validation separates choosing settings from evaluating the procedure. See [scikit-learn's explanation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

### Results on development customers

These are mean outer-fold results, with identical training and evaluation customers for each candidate.

| Candidate | Mean AP | Mean Brier, lower is better |
|---|---:|---:|
| Existing XGBoost | 0.668085 | 0.134184 |
| Shallower XGBoost | 0.672815 | 0.133120 |
| More regularized depth-3 XGBoost | 0.670345 | 0.133701 |
| Logistic regression, C=1 | 0.659246 | 0.135020 |
| Logistic regression, C=0.1 | 0.658581 | 0.135221 |

The inner procedure selected shallower XGBoost in all five outer folds. Therefore its mean AP is also the nested selection procedure's AP. It beat the existing configuration in all five outer folds, by an average of 0.00473 AP, or 0.47 percentage points.

Fold-to-fold AP standard deviations were approximately 0.0184 for the existing model and 0.0226 for the shallower model. These describe variation across folds, not confidence intervals. The improvement is small relative to variation in absolute scores.

#### Additional stability check

After the nested experiment, both fixed XGBoost configurations were compared over five folds using seed 2027 and five using seed 2028. The published test customers remained excluded.

| Configuration | Mean AP | Mean Brier |
|---|---:|---:|
| Existing XGBoost | 0.668574 | 0.133736 |
| Shallower XGBoost | 0.671364 | 0.132989 |

The shallower model had higher AP in 8 of 10 comparisons. These folds overlap and reuse the development dataset; they are a follow-up stability check, not ten independent confirmations or fresh external tests. No further configuration changes were made based on these results.

### Check on the existing app split

After fixing the new configuration, the normal app training entry point was run once. It still trains on 60%, selects the model on 20%, and reports the remaining 20%.

| XGBoost measure | Previous | Updated |
|---|---:|---:|
| Validation AP | 0.6459 | 0.6480 |
| Test AP | 0.6631 | 0.6674 |
| Test precision at 0.50 | 0.6644 | 0.6643 |
| Test recall at 0.50 | 0.5134 | 0.5027 |
| Test Brier | 0.1350 | 0.1345 |

The updated model was still selected over logistic regression on validation AP. Ranking improved slightly, but recall fell at the fixed threshold. This is not an improvement on every measure. Changing the threshold could change recall, but would not by itself improve AP.

The existing test results had already been published and discussed before this study. This check is therefore a secondary comparison on a previously exposed test set, not a pristine new final test. The nested results use more training data per outer fit than the app's 60% fit, so their absolute scores should not be compared directly to the app test AP.

### Reproduce the experiment

Use the pinned project requirements and a copy of the original IBM CSV. The script refuses to substitute fictional fallback data: pass the actual CSV path.

From the repository root:

```bash
python experiments/compare_models.py /path/to/Telco-Customer-Churn.csv
python experiments/compare_models.py /path/to/Telco-Customer-Churn.csv --stability
python -m pytest -q
```

Replace the CSV path with its actual location. The scripts write results beside the experiment script.

Saved records:

- [Nested validation results](../experiments/results.json).
- [Stability comparisons](../experiments/stability.json).
- [Updated app-split results](../experiments/published_split.json).
- [Experiment code](../experiments/compare_models.py).

The dataset SHA-256 is `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. It has 7,043 rows, no duplicate customer IDs, and 40 repeated predictor profiles before cleaning. Matching profiles do not necessarily identify the same customer; the supplied data provides no additional entity linkage or time grouping.

The experiment and all 21 existing tests ran successfully on Python 3.12 with scikit-learn 1.5.2 and XGBoost 2.1.4. Tests cover preprocessing isolation, split isolation, predictions, both SHAP explainers, business calculations, charts, and Streamlit sample/upload flows. Dependency deprecation warnings remain.

### What this does and does not establish

This study supports a small ranking improvement on held-out portions of this dataset. It cannot prove that overfitting is absent, and it does not establish transfer to another dataset. Additional tuning against these same evaluations would weaken their value.

A stronger next step requires new information: customer histories measured before departure, a defined future prediction window, and a later time period or independent company dataset. Then the same frozen training procedure can be evaluated without changing it in response to the new test results.

The app still retrains and chooses between logistic regression and XGBoost for each uploaded dataset. The updated default is not guaranteed to win for every upload. Probability calibration and the effect of retention offers have not been established by this experiment.
