---
layout: post
title: Obesity Project
subtitle: Which combinations of lifestyle habits have an impact on obesity, and can obesity levels be predicted using these features?
cover-img: /assets/img/obesityheader2.jpg
thumbnail-img: /assets/img/obesitythumb.jpg
tags: [clustering, classification]
author: Keira Hill
---

# Introduction
This project examines obesity levels and the influence of lifestyle factors. Over the past several decades, obesity has been increasing (Tzenios, 2023), posing a risk for chronic diseases such as cancer, diabetes, and heart disease (Abbott, Lemacks, and Greer, 2022; Tiwari and Balasundaram, 2023). These conditions contribute to high mortality rates and healthcare costs. Lifestyle factors play a crucial role in managing obesity (Tanaka and Nakanishi, 1996; Araromi et al., 2024; CDC, 2024). Understanding which lifestyle factors most significantly impact obesity can help individuals manage aspects within their control.

# Dataset
The dataset for this project, [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), is sourced from the UCI Machine Learning Repository. Data was collected via an anonymous survey from individuals in Peru, Mexico, and Colombia. A total of 485 respondents answered questions about their nutritional and physical behaviour. Classifications of nutritional status, as defined by WHO and provided in figure 1, were applied based on BMI calculated by weight and height, with an additional split on ‘Pre-obesity’.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/BMITable.png) |
|:-----:|
| *Figure 1. (WORLD HEALTH ORGANISATION, 2022)* |

Due to an imbalance towards 'normal' weight among respondents, additional data was synthetically generated using SMOTE, increasing the observations to 2,111 (Palechor and Manotas, 2019).
  
The data was loaded into the notebook to be explored using Python 

```javascript
obesity_data = pd.read_csv(
    "https://github.com/mwaskom/seaborn-data/raw/master/diamonds.csv"
)
obesity_data = obesity_data.toPandas()
```

Initial exploration reveals no missing values. However, some columns expected to contain object data were numeric, requiring further investigation (figure 2). 

{:.image-caption}
| ![BMI](/assets/img/project_obesity/InitalExploration.png) |
|:-----:|
| *Figure 2. Initial Exploration of dataset* |

Most column names were not self-explanatory, so they were renamed for clarity and consistency. Ordinal features were categorised as 'categorical' with specified orders.   
```javascript
// reference: (The Py4DS Community, 2023)
// sets objects to categorical for ordering 
obesity_data["food_between_meals"] = obesity_data["food_between_meals"].astype(
    CategoricalDtype(
        categories=["no", "Sometimes", "Frequently", "Always"], ordered=True
    )
)
obesity_data["alcohol_frequency"] = obesity_data["alcohol_frequency"].astype(
    CategoricalDtype(
        categories=["no", "Sometimes", "Frequently", "Always"], ordered=True
    )
)
obesity_data["weight_class"] = obesity_data["weight_class"].astype(
    CategoricalDtype(
        categories=["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"], ordered=True
    )
)
```
Categorical columns encoded as floats were analysed through histograms (figure 3). 

{:.image-caption}
| ![BMI](/assets/img/project_obesity/CategorigaclHistograms.png) |
|:-----:|
| *Figure 3. Histograms to show distributions on columns that should be categorical* |

The float values may have been generated during the SMOTE process used for data balancing. These columns were therefore rounded to match the correct number of responses for each feature, allowing them to be interpreted according to the original response options. Additionally, a set of dictionaries has been created to map the encoded responses back to the text responses for clarity within the methodology stage.
```javascript
// Mapping dictionaries
usually_veg_in_food_mapping = {1: 'Never', 2: 'Sometimes', 3: 'Always'}
daily_main_meals = {1: '1-2', 3: '3', 4: 'More than 3'}
daily_water_consumption_mapping = {1: 'Less than 1L', 2: '1 to 2L', 3: 'More than 2L'}
physical_activity_frequency_mapping = {0: 'None', 1: '1-2 days', 2: '2-4 days', 3: '4-5 days'}
tech_time_mapping = {0: '0-2 hours', 1: '3-5 hours', 2: 'more than 5 hours'}
```
An additional attribute for BMI, calculated using the formula weight/height², was added to support the exploratory data analysis (EDA). 

The final transformation involved splitting the data into training and testing datasets, a crucial step to avoid overfitting and to evaluate model performance. Studies suggest that allocating 20-30% of the data for testing yields optimal results (Gholamy, Kreinovich, and Kosheleva, 2018).
```javascript
// train test split
from sklearn.model_selection import train_test_split

// We have retained 20% of the data for the test set, random state set for reproducability
df_train, df_test = train_test_split(obesity_data, test_size=0.2, random_state=1234)
```
