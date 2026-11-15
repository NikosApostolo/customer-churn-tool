# Reading the dashboard

The dashboard separates four questions: what the historical data contains, how well the models predict, which customers receive high scores, and what a retention campaign could cost. Use the **Explore the dashboard** selector to visit those sections in that order on a first visit.

The sample comes from IBM's Telco Customer Churn dataset. An upload must contain historical churn outcomes and supported Telco-style inputs. This version does not score a separate unlabeled future-customer file.

## Customers

### Customers who stayed or left

The two bars count recorded outcomes across the loaded dataset. A churn rate of 26.5% means about 27 of every 100 customers left. This chart describes the historical sample; it is not a prediction.

The imbalance matters. If most customers stay, a model could predict “stay” for everyone and achieve a high accuracy while finding no customers who leave. That is why the Models tab focuses on precision, recall, and ranking quality.

### Missing values

This chart counts empty or invalid feature entries after format cleaning and before filling. Missing churn labels are not filled: they make an upload invalid because the outcome is needed for training and evaluation.

The app uses training medians for missing numbers and a `missing` category for missing text. It does not calculate replacement values from test customers. An entirely missing numeric column is kept and filled with zero by the imputer.

### Churn within customer groups

Select an input, such as Contract. Each bar shows the fraction of customers in that group who left. For numerical inputs with more than eight distinct values, the app creates six equal-width ranges and orders them numerically. Missing values form a separate group.

Tap or hover over a bar to check its customer count, or open **Group rates and customer counts** below it. A high rate in a group of five people is less reliable than the same rate in a group of hundreds. The chart does not show confidence intervals or establish that differences are statistically significant.

These charts explore the full historical dataset. They do not alter the fixed model settings, select features, or select the winning model. Differences between groups are associations, not evidence that changing an input causes someone to stay.

## Models

### The selected model

Both models learn the same training customers. Logistic regression combines weighted inputs into a probability estimate. XGBoost combines small decision trees to capture more complex patterns and interactions.

The data is separated into approximately 60% training, 20% validation, and 20% testing. Training teaches the models. Validation average precision chooses between them. Testing reports final performance. Test outcomes do not choose the winner, tune settings, or change the classification cutoff.

### Precision and recall bars

The bars compare both models at the same fixed cutoff: a risk score of 50% or more is flagged as churn.

- **Precision:** of the flagged customers, how many actually left? Precision of 66% means about 66 in every 100 flagged customers left, and the other 34 stayed.
- **Recall:** of the customers who actually left, how many were flagged? Recall of 51% means about 51 in every 100 churners were found, and 49 were missed.

These percentages use different denominators, so they answer different questions. More correct flags and fewer missed churners are both useful, but improving one can reduce the other.

### Precision–recall curves

The horizontal axis is recall: how many actual churners are found. The vertical axis is precision: how often a churn flag is correct. Each line shows what happens as the score cutoff changes. A line closer to the top-right generally represents a better tradeoff.

The dashed horizontal line is the churn rate in the test group, the expected precision of a ranking with no predictive information. Average precision, or AP, summarizes precision across recall increments. AP is not accuracy and should be interpreted against the churn proportion.

The display does not tune the cutoff. For large uploads, curves are downsampled for display while retaining endpoints; AP is calculated from all scores.

### The confusion matrix

Rows show actual outcomes and columns show model predictions. The four cells count:

| Actual outcome | Predicted stay | Predicted leave |
|---|---|---|
| Stayed | Correct stay prediction | False alarm |
| Left | Missed churner | Churner found |

The model selector defaults to the selected model. Switch it to inspect the other candidate. The text beside the chart spells out the important counts.

### Exact scores

The expandable table includes validation AP, test precision, test recall, test AP, and test Brier score. Brier score is the average squared error of probability predictions; lower is better. A low Brier score alone does not establish that the probabilities match real-world frequencies.

## Individual risk

### Risk distribution

The histogram counts held-out test customers in 10-point score ranges. For example, a bar spanning 60–70% contains customers whose estimated churn probabilities lie in that range.

The scores are estimates from the model. Calibration has not been established: a score of 70% does not mean we have verified that exactly 70 of every 100 such customers will leave.

### Churners found by ranking

Starting with the highest scores, the curve shows the share of actual test churners contained in the selected share of test customers. At 20% on the horizontal axis, the vertical value answers: “If we inspected the highest-risk fifth of customers, what fraction of all test churners would be included?”

The dashed diagonal is random selection in expectation. A curve above it indicates a useful historical ranking. It does not show how many people a retention offer would save. Actual outcomes are used only for this retrospective evaluation.

### Customer list

The table is sorted from highest to lowest estimated churn risk. Scores are displayed with one decimal place and a visual bar. Downloading the table preserves this ordering. The list contains test customers only, so their outcomes were not used to train or select the model.

### Individual explanation

Select a customer to see the score and the seven largest SHAP contributions. The chart combines encoded category columns back into their original input fields. Axis labels show readable field names; exact values appear in the chart details, the upward/downward summaries, and the expandable customer-input table.

- Coral bars push the model score up relative to the explanation's reference.
- Teal bars push the score down.
- Longer bars indicate stronger contributions on the model's score scale.

SHAP explains the model's behavior. It does not prove why the person left, or what would persuade them to stay. Log-odds are the internal score scale used for these contributions; a bar of 0.5 does not mean a 50-percentage-point probability increase. Contributions from different models or references should not be treated as directly interchangeable.

The two summaries below the chart identify the strongest upward and downward influences. A feature can increase the prediction without making the final risk high overall, because other inputs can offset it. The expandable section includes every contribution and the customer's cleaned input record.

## Retention plan

### Inputs

Choose the highest-risk percentage of test customers to contact. Contact counts are rounded up to a whole customer. Offer cost applies to every contacted customer. Customer value should represent lost contribution or profit over a defined period. Use one currency for both monetary inputs.

The assumed success rate is the share of otherwise-churning contacted customers whose churn the offer prevents. It is not the percentage of all contacts who accept the offer, and it is not estimated from the dataset.

### Money breakdown

The chart shows avoided loss, offer spend, and net savings. Spend is negative because it reduces the benefit. Net savings is the difference between avoided loss and spend; it is not a third amount to add to the first two.

Expected retained customers = sum of contacted risk scores × assumed success rate.

Net savings = expected retained customers × customer value − contacted customers × offer cost.

A fractional customer count is an expected average across similar campaigns, not a fraction of a person.

### Savings across campaign sizes

The line applies the same cost, value, and success assumptions to contact shares from 0% through 100%. The highlighted dot marks the current choice. Above the zero line means positive estimated net savings; below it means estimated costs exceed benefits.

The curve can turn downward when lower-risk contacts cost more than their expected benefit. Its highest point is an implication of the chosen assumptions, not a proven optimal policy. The app does not estimate how treatment effectiveness varies between customers.

### What would establish real value?

The dataset contains no retention experiment. A real campaign would need to randomly assign comparable customers to receive or not receive the offer, then compare their outcomes and costs. That would measure whether the intervention changes churn, rather than merely identifying people likely to leave.

## Keeping the display responsive

Models and explanations are cached: unchanged calculations are reused instead of repeated. Moving a campaign slider does not retrain the model. A different input file gets a new calculation unless its result is already cached. The data controls sit on the page rather than inside a sidebar. Only the selected section renders, reducing chart work on small devices. Section changes preserve the campaign assumptions and selected customer.

Detailed implementation: [Methodology](METHODOLOGY.md).


## Small-screen use

Open **Choose or change customer data** near the top to load a sample or select a CSV from your device. After loading, use the full-width **Explore the dashboard** selector to switch sections. You do not need to find a hidden sidebar or swipe through a wide tab row.

Charts and form columns stack below 900 CSS pixels. Summary cards form a two-column grid, with wrapping labels and smaller headings on phones. Input text uses a minimum 16px size and controls have generous touch targets. Chart axis titles use shorter or multiple lines, while dense tables retain their own horizontal scrolling.

The group-rate table, model counts, customer summaries, and expandable SHAP values provide alternatives to hover-only details. Download buttons span the available width. CSV selection and download prompts are handled by the device's browser.

Campaign settings and the selected customer persist while moving between sections in the current session. Upload data is temporary; a server restart can clear it.

The live layout was checked in Chrome at 340, 454, 681, and 1,363 CSS-pixel viewport widths, including section changes, input sizes, chart bounds, and campaign recalculation. These responsive browser-width checks do not replace testing every physical phone, operating system, and browser. Rendering and file-picker behavior can differ, particularly between Safari and Chrome.
