# Model card

## Purpose

Retain is an educational decision-support prototype for exploring telecom churn and
customer segmentation. It connects the original coursework question—whether customer
value adds useful retention context—to an inspectable training pipeline and a working demo.

The demo model is **value-aware gradient boosting**. Its designation is fixed in the
training code, rather than chosen from holdout results. Binary logistic regression,
binary gradient boosting, and a prior-only dummy classifier provide comparisons.

## Data and inputs

The selected file is the IBM Telco Customer Churn sample at
`Datasets/Final/WA_Fn-UseC_-Telco-Customer-Churn.csv`: 7,043 accounts, 1,869 observed
churners (26.5%), and 11 blank `TotalCharges` values. IBM describes its Telco sample
as fictional company data. See the [dataset notes](../Datasets/README.md) and
[IBM's sample description](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

The model uses 19 features:

- Account history: tenure, monthly charges, and total charges.
- Contract and billing: contract term, payment method, and paperless billing.
- Services: phone, multiple lines, internet, security, backup, device protection,
  technical support, and streaming TV/movies.
- Demographics: gender, senior-citizen indicator, partner, and dependents.

Customer identifiers and `Churn` are excluded from model inputs. Categories are
one-hot encoded. Numeric inputs use median imputation followed by standard scaling;
categorical inputs use most-frequent imputation. These transformations are fitted
inside each training fold. Unknown inference categories are handled by the encoder,
while the demo constrains users to known choices.

## Target and observed value

The four labels are `High_Churn`, `High_NoChurn`, `Low_Churn`, and `Low_NoChurn`.
Higher observed value means `TotalCharges` is at least the median of the fitting
partition. The default training partition produces a cutoff of **$1,398.125**.
Every cross-validation fold learns its own cutoff. Blank historical charges are
assigned to the lower observed-value tier; the numeric feature is still imputed
from the training median.

The account churn score is:

```text
P(churn) = P(High_Churn) + P(Low_Churn)
```

The demo combines that score with the account's known historical-charge tier.
Its displayed value tier is read from the input charges, rather than inferred
from the model's most likely four-class label. A value-tier mistake therefore
does not become a missed churn prediction.

Historical charges describe past billing. They do not measure prospective customer
lifetime value, margin, or the amount an intervention would save. Because the
value tier is directly determined by an input feature, learning that tier is largely
redundant. Binary churn prediction combined with known value is a simpler alternative.

## Training and evaluation protocol

1. Split accounts 80/20 with churn-stratified sampling and seed 42: 5,634 training
   accounts and 1,409 holdout accounts, including 374 holdout churners.
2. Generate five-fold out-of-fold churn scores on the training partition. All
   candidates use the same folds and holdout accounts.
3. Choose each model's review threshold by maximizing F2 over 0.10–0.80 in 0.01
   increments. Ties favor the highest threshold. F2 places more weight on recall
   than precision; it is a modeling policy rather than a measured business optimum.
4. Fit each model on the full training partition and evaluate the untouched holdout.
   Save the demo pipeline, its review threshold, and report together.

Both boosting models use 100 estimators, learning rate 0.05, and depth 2.
Logistic regression uses its default regularization with a 2,000-iteration limit.
The refreshed benchmark uses no resampling. These are fixed comparison settings,
not the output of an exhaustive hyperparameter search.

## Default-run results

| Model | Average precision | ROC AUC | Churn recall | Precision | Accounts flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy (prior) | 0.265 | 0.500 | 100.0% | 26.5% | 1,409 |
| Logistic regression | 0.634 | 0.842 | 92.2% | 43.7% | 790 |
| Binary gradient boosting | 0.661 | 0.844 | 89.8% | 43.3% | 776 |
| Value-aware gradient boosting | 0.652 | 0.842 | 89.0% | 43.1% | 773 |

At its 14% review threshold, the demo model catches 333 of 374 churners and
misses 41. Its queue also includes 440 non-churners. Four-class argmax accuracy
is 80.1%, a separate metric with a different decision rule. Neither number is
an 89% accuracy claim for high-value customers.

Binary gradient boosting has slightly higher average precision in this run;
logistic regression has higher recall and precision at its selected threshold.
The four-class approach adds a segmentation experiment, but this comparison
does not establish that it predicts churn better. The
[generated benchmark](../reports/benchmark.md) and
[JSON report](../reports/benchmark.json) contain the detailed results.

## Charge diagnostic

A missed churner is an actual churner whose summed churn score falls below the
review threshold. The diagnostic sums `TotalCharges` for those customers and
divides by recorded charges for all holdout churners. Missing charges contribute
zero to this diagnostic; an empty denominator returns zero.

The default demo run misses customers with **$108,706.55** of recorded historical
charges, 20.8% of the holdout churners' recorded charges. This is an exposure
proxy. Future revenue, intervention cost, retention uplift, and realized savings
cannot be estimated from this figure. A model that flags every account has zero
missed-charge exposure but creates the largest review queue.

## Limits and next experiments

- The dataset is a static sample. There is no validated prediction horizon,
  forward-looking time split, or evidence of performance on a live telecom business.
- The reported probabilities are uncalibrated model scores. Brier scores are
  recorded, but reliability curves and calibration are future experiments.
- Results come from one holdout split with no confidence intervals. Small
  differences should not be presented as proof of superiority.
- Demographic features remain for comparability with the coursework. Subgroup
  performance and fairness have not been evaluated.
- The demo's suggested outreach is illustrative. No causal analysis or experiment
  establishes that the suggested actions reduce churn.
- The median-based value definition, F2 policy, and demo input bounds are design
  choices. A deployed system would need business-specific value, budget, and costs.

Useful next work includes calibration on separate validation data, repeated or
time-based evaluation, subgroup analysis, and a retention-cost model with an
explicit review budget.

## Reproduction and artifacts

Run `python model_train.py` in the pinned environment. It creates
`artifacts/churn.joblib`, `reports/benchmark.json`, and `reports/benchmark.md`.
The report records the data SHA-256, split fingerprints, seed, dependency versions,
and confusion matrices. The saved demo model is fitted on the training partition;
the holdout remains excluded.

Only load artifacts produced by this trusted training workflow. Joblib uses
pickle-based serialization, and scikit-learn does not support loading models
across different dependency versions. See
[scikit-learn's persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

The [coursework archive](../archive/coursework/README.md) retains the original
notebooks, scripts, report, and outputs. Its figures use an older methodology
and are historical records; the current portfolio results come from the
refreshed workflow above.
