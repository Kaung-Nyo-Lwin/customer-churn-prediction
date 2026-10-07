# Dataset inventory

The current training workflow uses exactly one file:

```text
Final/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

It contains 7,043 accounts, 21 columns, and a `Churn` label with values `Yes`/`No`.
The workflow uses 19 predictors after excluding `customerID` and `Churn`.
Eleven `TotalCharges` entries are blank; their treatment is documented in the
[model card](../docs/model-card.md).

The dataset belongs to the IBM Telco Customer Churn sample family. The original
coursework collected candidate datasets through Kaggle. Primary references are
[IBM's sample description](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
and [IBM's example repository](https://github.com/IBM/telco-customer-churn-on-icp4d),
which includes the classic Telco customer-churn CSV.

The other CSVs, spreadsheets, and compressed archives in this directory are
historical candidate datasets from the class project. They are retained for
traceability and are excluded from the current benchmark. They should not be
combined with the selected file or treated as extra training accounts.

The committed benchmark records the selected file's SHA-256:

```text
88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a
```

Dataset attribution and upstream terms remain separate from the project code.
