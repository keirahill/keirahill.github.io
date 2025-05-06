---
layout: post
title: Obesity Project
subtitle: Which combinations of lifestyle habits have an impact on obesity, and can obesity levels be predicted using these features?
cover-img: /assets/img/pexels-n-voitkevich-6942035.jpg
thumbnail-img: /assets/img/obesitythumb.jpg
tags: [clustering, classification]
author: Keira Hill
---

# Executive Summary
This project aims to determine if BMI can be predicted using lifestyle factors, based on survey data from Latin America. Obesity is a global concern and risk factor for many chronic diseases. Understanding how lifestyle impacts BMI can help management of health.
Data clustering revealed lifestyle differences between genders, leading to separate prediction models. The XGBoost regressor model performed well on the combined dataset (R-squared: 80%, MAPE: 8.6%). The female model outperformed this (R-squared: 91%, MAPE: 6.4%), whilst the male model was less accurate (R-squared: 62%), indicating lifestyle factors are more predictive for females.
Family history of overweight was the most significant predictor, including genetic data could improve accuracy. 
An expanded source of data collection method could provide a more diverse, less biased dataset.


# Project Background
This project examines obesity levels and the impact of lifestyle factors such as eating habits, physical activity, age, and family history. 
Over recent decades, obesity has become a global concern (Figure 1) (Tiwari and Balasundaram, 2023; Tzenios, 2023). Obesity is a risk factor for chronic diseases like cancer, diabetes, and heart disease, leading to high mortality rates and healthcare costs (Abbott, Lemacks and Greer, 2022; Tiwari and Balasundaram, 2023). Lifestyle factors, including nutrition and physical activity, significantly influence the development and management of obesity (Tanaka and Nakanishi, 1996; Araromi et al., 2024; CDC, 2024). 
Understanding lifestyle factors most significantly impacting obesity, can inform personal health improvements via aspects such as diet, exercise and alcohol consumption (Tanaka and Nakanishi, 1996). 

{:.image-caption}
| ![BMI](/assets/img/project_obesity/1.png) |
|:-----:|
| *Figure 1. (TIWARI AND BALASUNDARAM, 2023)* |


# Data Engineering
## Dataset
The dataset for this project, [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) was sourced from The UCI Machine Learning Repository. 
It contains 17 attributes and 2,111 observations, collected via an anonymous survey from individuals in Peru, Mexico and Columbia. 485 respondents answered questions on nutritional and physical behaviours (Appendix One). Nutritional status classifications (Figure 2) were applied based on BMI calculated from weight and height, with additional categories for 'Pre-obesity.' 


{:.image-caption}
| ![BMI](/assets/img/project_obesity/2.png) |
|:-----:|
| *Figure 2. (WORLD HEALTH ORGANISATION, 2022)* |

Due to an imbalance towards ‘normal’ weight from the respondents, additional data was synthetically generated using SMOTE, balancing the dataset and increasing the observations to 2,111 (Palechor and Manotas, 2019).

## ETL

Tools used for processing and analytics in this project include Python, accessed through Azure Databricks, offering a user-friendly interface. Downloaded source data was uploaded to GitHub as a Panda’s DataFrame.

```javascript
obesity_data= pd.read_csv(
    "https://raw.githubusercontent.com/keirahill/keirahill.github.io/master/Datasets/ObesityDataSet_raw_and_data_sinthetic.csv"
)
```

Initial exploration examined data types and checked for missing values (Figure 3).


{:.image-caption}
| ![BMI](/assets/img/project_obesity/4.png) |
|:-----:|
| *Figure 3. Initial Exploration of dataset* |

No missing data was identified; however, some features are numeric when categorical data was expected. This discrepancy will be explored. Columns were renamed for clarity and consistency (Appendix One).
24 duplicate rows were identified. Due to limited data, it was unclear whether these are true duplicates, therefore, the records were retained.
Object features of the dataset were examined (Appendix Two), revealing some ordinal features. These columns were set to ordered categorical data types.
 
```javascript
// reference: (The Py4DS Community, 2023)
// sets objects to categorical for ordering 
obesity_data["food_between_meals_c"] = obesity_data["food_between_meals_c"].astype(
    CategoricalDtype(
        categories=["no", "Sometimes", "Frequently", "Always"], ordered=True
    )
)
obesity_data["alcohol_frequency_c"] = obesity_data["alcohol_frequency_c"].astype(
    CategoricalDtype(
        categories=["no", "Sometimes", "Frequently", "Always"], ordered=True
    )
)
obesity_data["weight_class_c"] = obesity_data["weight_class_c"].astype(
    CategoricalDtype(
        categories=["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"], ordered=True
    )
)
```
Some categorical fields have been encoded as numbers, also, several columns are floats rather than integers, as expected. Distributions of these features was explored through histograms (Figure 4). 

{:.image-caption}
| ![BMI](/assets/img/project_obesity/6.png) |
|:-----:|
| *Figure 4. Histograms to show distributions on columns that should be categorical* |

The histograms display three or four main responses, suggesting float generation through SMOTE. These features were rounded to the correct number of responses, allowing for interpretation of the original answer. Text responses were added to the dataset for use in the methodology stage. 
Many machine learning models require categorical data to be encoded as numerical values. Remaining ordinal features were encoded using Scikit-learn’s OrdinalEncoder and Panda’s get_dummies  was used for one-hot encoding of nominal features (Potdar, Pardawala, Taher, and Pai, Chimney, 2017).

```javascript
// reference: (Stack Overflow, 2022)
// applies ordinal encoding to ordinal features
from sklearn import preprocessing
OrdinalEncoder = preprocessing.OrdinalEncoder(categories=[["no", "Sometimes", "Frequently", "Always"]])
obesity_data['food_between_meals_e'] = OrdinalEncoder.fit_transform(obesity_data[['food_between_meals_c']])
obesity_data['alcohol_frequency_e'] = OrdinalEncoder.fit_transform(obesity_data[['alcohol_frequency_c']])


OrdinalEncoder2 = preprocessing.OrdinalEncoder(categories=[["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]])
obesity_data['weight_class_e'] = OrdinalEncoder2.fit_transform(obesity_data[['weight_class_c']])

// function for one hot encoding nominal categorical features
def applydummies(df, col, drop):
    dummies = pd.get_dummies(df[col], prefix=col, dtype='int',drop_first=drop)
    df = pd.concat([df, dummies], axis=1)
    return df

obesity_data = applydummies(obesity_data, 'gender_c', True) // drop one column for features with only 2 responses 
obesity_data = applydummies(obesity_data, 'family_history_with_overweight_c', True)
obesity_data = applydummies(obesity_data, 'frequent_high_cal_food_c', True)
obesity_data = applydummies(obesity_data, 'smoker_c', True)
obesity_data = applydummies(obesity_data, 'calorie_monitoring_c', True)
obesity_data = applydummies(obesity_data, 'transport_type_c', False) // dont drop first column as easier ton interpret with all responses
```
Two additional attributes were added to support the exploratory data analysis (EDA):  
- BMI, using weight / height²
- Combined weight class with four categories (underweight, healthy, overweight and obese)

