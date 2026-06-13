# Customer Churn Prediction using Logistic Regression

## Project Review

This project focuses on predicting customer churn for a banking dataset using logistic regression. The main goal was to understand which customer behaviors are linked with churn and then build a simple, interpretable machine learning model that can identify customers who are more likely to leave.

I chose logistic regression because it is a strong starting point for a binary classification problem. It is easy to interpret, works well with properly scaled numerical data, and gives probability scores that can be adjusted based on the business need.

In this dataset, churn means that the customer's average balance falls below the minimum balance requirement in the next quarter.

## Dataset

The dataset contains 28,382 customer records and 21 columns. The features cover three main areas:

| Area | Example columns |
| --- | --- |
| Customer profile | `age`, `gender`, `dependents`, `occupation`, `city` |
| Banking relationship | `vintage`, `customer_nw_category`, `branch_code`, `days_since_last_transaction` |
| Transaction activity | `current_balance`, monthly credits, monthly debits, previous and current balances |

The target column is `churn`, where:

- `0` means the customer did not churn
- `1` means the customer churned

One important thing I noticed early was that the data is imbalanced:

| Class | Count | Percentage |
| --- | ---: | ---: |
| Not churned | 23,122 | 81.47% |
| Churned | 5,260 | 18.53% |

Because only about 18.5% of customers churned, accuracy alone would not be a very useful metric. A model could predict most customers as "not churned" and still look accurate, while failing at the actual business problem.

## What I Did

### 1. Understood the problem as binary classification

The project is a binary classification task. The model has to separate customers into two groups: likely to churn and not likely to churn.

For a bank, this kind of prediction can be useful because it gives the business a chance to take action before a customer leaves. For example, customers with high churn probability could be targeted with retention offers, service calls, or personalized engagement.

### 2. Handled missing values

Before training the model, I checked for missing values and handled them column by column.

| Column | Missing values | Treatment |
| --- | ---: | --- |
| `days_since_last_transaction` | 3,223 | Filled with `999` to represent no recent transaction |
| `dependents` | 2,463 | Filled with `0` |
| `city` | 803 | Filled with the most common city code, `1020` |
| `gender` | 525 | Encoded Male/Female as `1/0` and missing as `-1` |
| `occupation` | 80 | Filled with the most common category, `self_employed` |

This step was necessary because logistic regression cannot work directly with missing values.

### 3. Converted categorical data into numeric features

The `occupation` column was converted using one-hot encoding. This created separate columns such as:

- `occupation_company`
- `occupation_retired`
- `occupation_salaried`
- `occupation_self_employed`
- `occupation_student`

This was better than assigning random numbers to occupations because occupation is not an ordered variable.

### 4. Transformed and scaled numerical columns

The balance and transaction columns had large ranges and skewed values. To make them more suitable for logistic regression, I applied:

- Log transformation
- Standard scaling

This helped bring features such as balances, credits, and debits onto a more comparable scale.

### 5. Built a baseline model

I first trained a logistic regression model using a selected set of baseline features, including:

- `current_month_debit`
- `previous_month_debit`
- `current_balance`
- `previous_month_end_balance`
- `vintage`
- occupation-related features

This gave me a simple starting point before trying more features and feature selection.

### 6. Evaluated using the right metrics

Since the dataset is imbalanced, I focused more on:

- ROC-AUC
- Recall
- Precision
- Confusion matrix

Recall was especially important because, in a churn problem, missing an actual churner can be costly. If the model fails to identify a customer who is about to churn, the bank loses the chance to retain that customer.

### 7. Used cross-validation

I used 5-fold stratified cross-validation to test the model more reliably. This helped check whether the model was performing consistently across different splits of the data instead of depending on only one train-test split.

### 8. Applied Recursive Feature Elimination

After trying the baseline model and the model with all features, I used Recursive Feature Elimination (RFE) to rank the features.

The top 10 features selected by RFE were:

| Rank | Feature |
| ---: | --- |
| 1 | `current_balance` |
| 2 | `average_monthly_balance_prevQ` |
| 3 | `occupation_company` |
| 4 | `average_monthly_balance_prevQ2` |
| 5 | `current_month_balance` |
| 6 | `previous_month_balance` |
| 7 | `current_month_debit` |
| 8 | `occupation_retired` |
| 9 | `previous_month_debit` |
| 10 | `occupation_student` |

This result made sense because churn in this dataset is closely related to balance behavior. Customers with weaker or declining balance patterns are more likely to become risky from a churn point of view.

## Results

I compared three versions of the logistic regression model.

| Model | Mean ROC-AUC across 5 folds |
| --- | ---: |
| Baseline feature model | 0.7642 |
| Model with all features | 0.7462 |
| RFE top 10 feature model | 0.7980 |

The best result came from the model using the top 10 RFE-selected features. This was an interesting finding because adding all features did not improve the model. A smaller and more relevant feature set performed better.

At the default threshold of `0.5`, the RFE model had high precision but low recall:

| Metric | Mean value |
| --- | ---: |
| Recall | 0.2150 |
| Precision | 0.7255 |

This means the model was careful when predicting churn, but it missed many actual churners.

I then lowered the classification threshold to `0.14`, which improved recall significantly:

| Metric | Mean value |
| --- | ---: |
| Recall | 0.8257 |
| Precision | 0.2880 |

This showed the tradeoff clearly. A lower threshold catches more churners, but also increases false positives. In a real business setting, the final threshold would depend on how expensive it is to contact customers compared to how expensive it is to lose them.

## What I Learned

This project helped me understand that machine learning is not only about fitting a model. A lot of the real work happens before and after model training.

Some of my main learnings were:

- Missing value treatment should be based on the meaning of each column, not done blindly.
- Accuracy is not always the right metric, especially for imbalanced classification problems.
- Logistic regression can be a very useful model when the goal is to build something interpretable.
- Feature scaling matters a lot for linear models.
- More features do not always lead to better performance.
- Feature selection can improve both performance and explainability.
- The classification threshold should be selected based on the business objective.

The threshold tuning part was one of the most useful parts of the project for me. It showed that model outputs are not just technical numbers. They can directly affect business decisions.

## What Could Be Improved

Possible next steps:

- Move scaling and feature selection inside cross-validation to avoid possible data leakage.
- Try other models such as Random Forest, XGBoost, or LightGBM.

## Final Thoughts

This project was a good exercise in applying machine learning to a realistic business problem. I worked through the main steps of a supervised learning workflow: preprocessing, feature engineering, model training, cross-validation, feature selection, and evaluation.

The final model using RFE-selected features achieved the best ROC-AUC score of around 0.798. More importantly, the project showed how recall and precision can be adjusted through threshold tuning depending on the business goal.

For me, the biggest takeaway was that a good machine learning solution is not just the model with the highest score. It should also be understandable, connected to the business problem, and evaluated using metrics that match the real-world decision being made.
