# COVID-19 New Cases Prediction ML Project (OWID Dataset)

**Author:** Shlomi Jakubowicz
**Contact:** jshlomi@yahoo.com or jshlomi81@gmail.com

## Project Overview

This project focuses on developing a machine learning model to predict "new_cases_smoothed_per_million," a 7-day smoothed average of new confirmed COVID-19 cases per 1,000,000 people. We utilize the comprehensive COVID-19 dataset provided by Our World in Data (OWID), which includes a wide array of relevant indicators. The smoothed target variable helps in reducing daily fluctuations and allows for better comparisons across regions with different population sizes.

## Motivation

Accurate prediction of new COVID-19 cases is crucial for:

* **Public Health Planning:** Efficient allocation of healthcare resources.
* **Policy Making:** Informed decisions on public health measures.
* **Economic Stability:** Anticipating disruptions and planning for continuity.
* **Public Awareness:** Enhancing understanding and compliance with health guidelines.
* **Research and Development:** Guiding research on virus dynamics and intervention effectiveness.

## Dataset

* **Source:** Our World in Data (OWID) Covid-19 Dataset ([https://docs.owid.io/projects/covid/en/latest/dataset.html](https://docs.owid.io/projects/covid/en/latest/dataset.html))
* **Time Span:** January 2020 to August 2024
* **Size:** 429,435 records, 67 features
* **Key Categories:** Cases, Deaths, Testing, Hospitalizations, Vaccinations, Population Metrics, Policy Responses, and Date.
* **Target Feature:** `new_cases_smoothed_per_million` (New confirmed cases of COVID-19 (7-day smoothed) per 1,000,000 people).

## Data Preparation and EDA

* The 'date' column was converted to datetime format.
* Rows with missing 'continent' values were removed to focus on specific countries.
* The target feature has approximately 30% zero values and around 3% missing values.
* Exploratory Data Analysis (EDA) included visualizations of the target variable over time, its distribution, and average values by location.
* Analysis of estimated key factors (Stringency Index, Population Density, GDP per Capita, Vaccination Rate) from 2020-2023 suggested their impact on COVID-19 spread.

## Data Characteristics

* Most features, including the target variable, exhibited skewness, indicating a non-normal distribution.
* Outlier detection using IQR was performed.
* Spearman correlation was used to assess feature relationships due to non-normal data.

## Outlier and Missing Value Handling

* Log transformation was applied to skewed numerical features before IQR-based outlier detection.
* Detected outliers were set to NaN and, along with other missing values, were imputed using the MICE imputer.

## Feature Engineering, Feature Selection, and Data Leakage Prevention

* **Feature Engineering:** Created `total_cases_change_7` (change in total cases over the last 7 days) and extracted time-based features (year, month, week, day, day of the week).
* **Data Leakage Prevention:** Removed features like `total_cases` and `total_cases_per_million` as they contain future information. All new cases features except the target were dropped.
* **Feature Selection:** Four techniques were employed:
    1.  Lasso (L1 Regularization)
    2.  Ridge (L2 Regularization)
    3.  Gradient Boosting Regressor (Feature Importances)
    4.  Random Forest Regressor (Feature Importances)
    The results of these methods were combined to identify the most consistently important features.

## Model Training and Evaluation

* **Cross-Validation:** 5-fold cross-validation was performed using geographical splits based on `iso_code` to ensure generalization to new regions. The `iso_code` was *not* used as a training feature to prevent leakage.
* **Feature Scaling:** StandardScaler was applied to numerical features within each fold.
* **Models Compared:** Dummy Regressor, Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, SVR, K-Nearest Neighbors Regressor, XGBoost Regressor.
* **Evaluation Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R2), Root Mean Squared Error (RMSE), and Root Mean Squared Logarithmic Error (RMSLE).
* Detailed performance metrics for each model across all folds were recorded.

## Conclusion

The **Random Forest Regressor** consistently outperformed other models across most evaluation metrics, demonstrating strong predictive capability for smoothed new COVID-19 cases per million. Gradient Boosting Regressor and XGBoost Regressor also showed competitive results. Ensemble methods proved to be more effective than linear models or nearest neighbors for this forecasting task.
