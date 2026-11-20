# Methodology

For equations, derivations, and worked examples, see [Mathematical modelling](MATHEMATICAL_MODELLING.md).

For chart-by-chart explanations and examples, see [Reading the dashboard](READING_THE_DASHBOARD.md).

## Input data

The app accepts a UTF-8, comma-separated CSV containing historical customer records. Files must be at most 10 MB, with 100–15,000 rows and at least 20 examples of each outcome.

`Churn` is required and must contain Yes/No or 1/0. Missing outcomes are rejected rather than guessed. `customerID` is optional; when supplied, values must be nonempty and unique. Otherwise, row identifiers are generated for display.

At least one supported feature is required:

| Type | Columns |
|---|---|
| Numeric | `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` |
| Categorical | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

Other columns are ignored and reported in the cleaning notes. Restricting inputs to named fields avoids silently treating arbitrary identifiers or outcome summaries as features. It does not establish when the inputs were measured: all supplied features must have been available before the churn outcome.

The sample is downloaded from IBM's public CSV. If downloading or validation fails, the app creates 800 fictional records with a fixed seed and labels the source as synthetic. Results from that fallback are not Telco results.

## Cleaning and preparation

Column names lose surrounding spaces. Category values are stripped and lowercased, and common missing-value markers become missing entries. Numeric fields are converted to numbers; invalid, negative, and nonfinite values become missing. `SeniorCitizen` accepts only 0 or 1. Categorical fields with more than 50 distinct values are rejected to catch likely free text or identifiers.

These format checks do not estimate values from the dataset. The learned preparation happens inside each model's training pipeline:

- Missing numeric values are filled with the training median. An entirely missing numeric column is retained and filled with zero by the imputer.
- Numeric inputs are standardized using training means and standard deviations.
- Missing categories are filled with `missing`.
- One-hot encoding turns category values into numeric indicator columns. Previously unseen categories are accepted and encoded as zeros for that field.

Customer IDs remain available for display, but are not used as features. The churn outcome is also excluded from the inputs.

## Training and selection

The split uses random seed 42 and preserves approximately the same class proportions in each group:

| Group | Share | Use |
|---|---:|---|
| Training | 60% | Fit data preparation and both models |
| Validation | 20% | Choose the model using average precision |
| Test | 20% | Report final metrics and rank unseen customers |

Logistic regression uses a maximum of 2,000 optimization iterations. XGBoost uses 400 trees, maximum depth 2, learning rate 0.03, row and column sampling of 0.9, L2 regularization of 5, and minimum child weight 10. These settings were chosen in a small nested-validation study excluding the published test customers; see [Model improvement study](MODEL_IMPROVEMENT.md). They remain fixed when the app trains on an upload.

The model with the highest validation average precision is selected. Both candidates' test results are displayed, but those results do not change the selection. The models are not subsequently fitted on the test customers. All displayed rankings therefore come from customers held out of both training and selection.

The app uses a fixed probability threshold of 0.50 for precision, recall, and the confusion matrix. Ranking uses the continuous risk score, so customers can still be ordered even when their scores are below the threshold.

## Evaluation

Precision measures how often a churn flag is correct. Recall measures how many actual churners are found. A confusion matrix shows true and false predictions for both outcomes. These are more informative than accuracy alone when the classes are imbalanced.

Average precision evaluates ranking quality across thresholds. Its reference level depends on the churn proportion, so results from datasets with different class proportions should not be compared without context.

Brier score is the mean squared difference between predicted probabilities and observed 0/1 outcomes. It measures probability error but does not, by itself, prove that the probabilities are calibrated. Calibration would mean that customers receiving similar estimated risks churn at approximately those rates over many observations.

## Individual explanations

Logistic regression is explained with SHAP LinearExplainer, using up to 100 training customers as its background. XGBoost is explained with TreeExplainer in raw-output mode, using its tree-path reference.

Both explainers return contributions in log-odds units. Contributions from one-hot indicators are summed into their original feature, so users see Contract rather than several encoded contract columns. The chart displays the five largest absolute contributions, and the table includes all features.

The explanations describe why a model assigns its score relative to its reference. They do not identify proven causes of churn. Different models or references can distribute contributions differently, especially when inputs are correlated.

## Retention scenario

Customers are sorted by decreasing risk. The contact percentage selects the highest-risk share of the test group, rounding the count upward.

The calculator uses four inputs:

| Input | Meaning |
|---|---|
| Contact percentage | Share of ranked test customers to contact |
| Offer cost | Cost for each contacted customer, whether retained or not |
| Customer value | Lost contribution or profit over a chosen time horizon |
| Offer success rate | Assumed share of otherwise-churning contacted customers whose churn is prevented |

Let p be the contacted customers' risk scores, s the assumed success rate, v the customer value, n the number contacted, and c the offer cost:

```text
Expected churners without an offer = sum(p)
Expected retained customers = sum(p) × s
Avoided loss = sum(p) × s × v
Offer spend = n × c
Net savings = avoided loss − offer spend
```

For example, scores totaling 60, a success rate of 20%, customer value of 500, and 100 contacts costing 20 each give 12 expected retained customers and net savings of 4,000.

This calculation assumes the risk estimates are useful probability estimates and the chosen success rate applies to contacted would-be churners. The data does not contain an offer experiment, so the success rate is a scenario input. No claim of measured financial impact follows from the calculation.

## Caching and error handling

The sample is cached for one hour. Training bundles are cached for 30 minutes, with at most three entries, and individual explanations are also cached. Changing campaign controls or the selected customer reuses the trained model. A different input file triggers a new calculation unless its result is already cached.

The app converts validation failures into readable messages. Unexpected errors are logged on the server and shown as a generic recovery message. An explanation failure leaves the rankings and business section available. These measures handle expected failures but cannot guarantee availability during a hosting outage or exhausted server resources.

## Scope

This is an application of established machine learning methods to a retention workflow. Its practical contribution is connecting model comparison, customer ranking, individual explanations, and campaign assumptions in one interface.

Separate scoring of unlabeled future customers, time-based validation, probability calibration, fairness assessment, campaign experiments, and production access controls remain outside the current implementation.


## Visual summaries

Interactive charts are implemented in `src/charts.py` with Altair. Customer exploration summarizes the full historical input, while model metrics, risk rankings, capture curves, and campaign scenarios use held-out test customers. Chart data is derived from the same trained bundle used for the displayed results.

The capture curve cumulatively counts actual test churners in descending risk order. Its diagonal reference represents random selection in expectation. The campaign curve applies the exact calculator formula at each integer contact percentage, with the same rounding rule for contact counts. Automated checks compare these curve values against the calculator.

The precision–recall chart varies the decision threshold for display only. Neither this curve nor the test capture curve changes model selection or the fixed 0.50 classification threshold. Large precision–recall displays are limited to 2,000 points per model, retaining endpoints; scores and metrics always use all observations.

The four-section layout separates exploration, evaluation, individual explanations, and business assumptions. Expandable sections retain exact scores and technical details without making raw tables the main interface. The visual redesign does not change model settings, the data split, or the measured Telco results above.
