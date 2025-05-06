---
layout: post
title: Obesity Project
subtitle: Which combinations of lifestyle habits have an impact on obesity, and can obesity levels be predicted using these features?
cover-img: /assets/img/pexels-n-voitkevich-6942035.jpg
thumbnail-img: /assets/img/obesitythumb.jpg
tags: [clustering, classification, regression]
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

# Data Analysis
## Hypotheses
Hypothesis 1: Higher water intake associated with lower BMI, as water acts as an appetite suppressant.
Hypothesis 2: Frequency of exercise directly affects BMI, with no exercise increasing the likelihood of being overweight.

## EDA
Clustering will be part of the EDA to identify data patterns. EDA will be performed on the full dataset, with train and test splits used to evaluate clusters. Python packages, including Plotly for its interactivity, will be used. 
Figure 5 provides initial insights on numerical features, showing feature adjustments made.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/8.png) |
|:-----:|
| *Figure 5. Initial insights on numeric features* |

## Continuous Data
A pairplot (Figure 6) was conducted on the continuous data, to explore distributions and relationships. The distribution confirms a right-skew in age, which appears accurate, so all data was retained. There is a strong correlation between weight and BMI, confirmed by a correlation plot (Figure 7). For modelling purposes, weight will be dropped when BMI is used.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/9.png) |
|:-----:|
| *Figure 6. Pairplot* |

  
```javascript
sns.heatmap(obesity_data_numeric.corr(), annot = True, cmap='Blues').set_title('Correlation of Variables')
```

{:.image-caption}
| ![BMI](/assets/img/project_obesity/10.png) |
|:-----:|
| *Figure 7. Correlation matrix* |


## BMI vs Weight Class

A boxplot (Figure 8) was created to evaluate the BMI and weight class relationship, confirming BMI was used to determine weight class, although bandings are not exact, due to synthetic data. BMI will be used going forwards.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/11.png) |
|:-----:|
| *Figure 8. Boxplot of BMI vs Weight Class* |

## Categorical Features
Distributions of categorical features were explored (Figure 9), revealing some for consideration.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/12.png) |
|:-----:|
| *Figure 9. Distributions* |

Box plots were used to visualise BMI distributions across each category (Figure 10). Many align with expectations, i.e. physical activity supporting Hypothesis 2. However, some results were unexpected, with water consumption contradicting Hypothesis 1.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/13.png) |
|:-----:|
| *Figure 10. Boxplots of BMI distributions* |
  
  
## Clustering
### K-Protoypes
The dataset was clustered using K-Prototypes, a method that handles mixed datatypes, applying k-means for numeric data and matching categories for nominal data (Ruberts, 2020). Ordinal features were treated as numeric. All numeric features were normalised using Scikit-learn’s MinMax scaler, ensuring equal contribution.
BMI and weight class were excluded from the modelling to identify patterns within the other features.

```javascript
// list categorical and numeric features to be used in clustering, ordinal catergorical features have been set as numeric
featurescat = ['gender_c_Male',  'family_history_with_overweight_c_yes', 'frequent_high_cal_food_c_yes', 'smoker_c_yes', 'calorie_monitoring_c_yes', 'transport_type_c_Automobile', 'transport_type_c_Bike', 'transport_type_c_Motorbike', 'transport_type_c_Public_Transportation', 'transport_type_c_Walking']
featuresnumeric= ['age__y', 'height__m', 'weight__kg', 'usually_veg_in_food_e', 'daily_main_meals_e', 'daily_water_consumption_e', 'physical_activity_frequency_e', 'tech_time_e', 'food_between_meals_e', 'alcohol_frequency_e']

// normalise numeric data through minmax scaling
scaler = pp.MinMaxScaler()
obesity_data_clusters = obesity_data
obesity_data_clusterscat = obesity_data_clusters[featurescat]   // categorical features
obesity_data_clustersnum = obesity_data_clusters[featuresnumeric]  // numeric features
obesity_data_clustersnum_scaled = scaler.fit_transform(obesity_data_clustersnum) // normalise the numeric features between 0 and 1

obesity_data_clustersnum_scaled = pd.DataFrame(obesity_data_clustersnum_scaled, columns=featuresnumeric) // create a pandas dataframe of normalised numeric features
obesity_data_combined = pd.concat([obesity_data_clusterscat, obesity_data_clustersnum_scaled], axis=1) // combine the categorical and normalised numeric features into one dataframe

// Elbow method to find the optimal number of clusters to use
// References: (Dmitriy, 2019; Ruberts, 2020) 

from kmodes.kprototypes import KPrototypes
// Convert column names to indices which are required for k-protoypes
categorical_indices = [obesity_data_combined.columns.get_loc(col) for col in featurescat]

random_seed = 42 // set ramdom seed for reproducability

ks = range(1, 10) // range of clusters to test
inertias = [] // create an empty df to append values to
// loop through each value of k
for k in ks:
    // Create a Kprotoypes instance with k clusters
    model = KPrototypes(n_clusters=k, random_state=random_seed)
    
    // Fit model to samples
    model.fit(obesity_data_combined,  categorical = categorical_indices )
    
    // Append the inertia to the list of inertias
    inertias.append(model.cost_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

```

An elbow plot (Figure 11) suggests the optimal number of clusters is 4, with observations fairly evenly distributed (Figure 12).

{:.image-caption}
| ![BMI](/assets/img/project_obesity/14.png) |
|:-----:|
| *Figure 11. Elbow plot* |

    
{:.image-caption}
| ![BMI](/assets/img/project_obesity/15.png) |
|:-----:|
| *Figure 12. Distrbutions of clusters* |
  
  
### Factor Analysis
Visualising clusters with 16 features is challenging, requiring a feature reduction. As it is a mixed dataset, FAMD from the Prince package was chosen (Mahmood, Md Sohel, 2021). Although only 33% of the variance is explained by the first three features (Figure 13), plotting these clusters against these three reveals distinct patterns, indicating the data can be grouped (Figure 14).

{:.image-caption}
| ![BMI](/assets/img/project_obesity/16.png) |
|:-----:|
| *Figure 13. Explained varaince by features* |

  
{:.image-caption}
| ![BMI](/assets/img/project_obesity/17.png) |
|:-----:|
| *Figure 14. 3D scatter plot of clusters* |

  
### Cluster Analysis

{:.image-caption}
| ![BMI](/assets/img/project_obesity/18.png) |
|:-----:|
| *Figure 15. BMI and Weight Class by cluster* |
  
  
{:.image-caption}
| ![BMI](/assets/img/project_obesity/19.png) |
|:-----:|
| *Figure 16. Distubutions of each cluster by all features* |

To explore using SHAP, the data was split into training and testing datasets to evaluate model performance. Studies suggest that allocating 20-30% of the data for testing yields optimal results (Gholamy, Kreinovich and Kosheleva, 2018). 

```javascript
// train test split
from sklearn.model_selection import train_test_split

// retained 20% of the data for the test set
df_train, df_test = train_test_split(obesity_data_clusters, test_size=0.2, random_state=1234)

```

An XGBoost classifier was applied to the training dataset to predict clusters in the test dataset. The model achieved an accuracy, average recall and precision of 97%, ensuring confidence in SHAP outcomes (Figure 17). 

{:.image-caption}
| ![BMI](/assets/img/project_obesity/21.png) |
|:-----:|
| *Figure 17. Accuracy stats of XGBoost classifier model* |

SHAP beeswarm plots (Figure 18) display the importance of each feature for predicting clusters, with density and colour providing additional insights (Lundberg, 2021). Figure 19 provides a summary of the clusters.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/22.png) |
|:-----:|
| *Figure 18. SHAP Beeswarm for each cluster* |

    
{:.image-caption}
| ![BMI](/assets/img/project_obesity/23.png) |
|:-----:|
| *Figure 19. Summary of most important features in predicting each cluster* |

Analysis identified four clusters: two male, two female. The most influential metrics are similar for each gender, though often with opposing values. Therefore, a regression model will be used to predict BMI, with separate gender models to evaluate any improvement. 

## XGBoost

XGBoost, a popular gradient boosting algorithm for classification and regression (Brownlee, 2016; Tuychiev, 2023) was chosen for predicting BMI. Weight was excluded to focus other feature impacts. Evaluation results for combined, female and male models are in Figure 20 and scatter plots of predicted vs. actual BMI in Figure 21.

```javascript
// reference: (Fullegar, 2024)
data = {'Combined': [r2_score, rmse, mape],
        'Male': [r2_score_male, rmse_male, mape_male], 
        'Female':[r2_score_female, rmse_female, mape_female]} 
  
df_comp = pd.DataFrame(data) 
df_comp = df_comp.rename(index={0 : 'R squared score', 
                                1 : 'Mean Squared Error', 
                                2 : 'Mean Absolute Percentage Error' })
df_comp
acc_df = df_comp.sort_values(by = 'R squared score', axis = 1, ascending = False)
acc_df.T
```

{:.image-caption}
| ![BMI](/assets/img/project_obesity/24.png) |
|:-----:|
| *Figure 20. Accuracy stats for each model* |

{:.image-caption}
| ![BMI](/assets/img/project_obesity/25.png) |
|:-----:|
| *Figure 21. Actual vs Predicted BMI* |

The combined model outperformed the male model with a R-squared of 80% vs. 62%. The female model performed the best, achieving an R-squared of 91% and a MAPE of 6.4%

# Results
## SHAP
Figure 22 compares the SHAP beeswarm plots for the combined, male, and female models. The majority of the top 10 features are consistent. Interestingly, features identified in the hypotheses are not among the top 10 impacts. The EDA confirmed increased physical activity is associated with lower BMI, however increased water consumption is associated with higher BMI, contrary to hypothesis. 

{:.image-caption}
| ![BMI](/assets/img/project_obesity/26.png) |
|:-----:|
| *Figure 22. SHAP beeswarm for each model and feature importance summary* |

  
## Power BI
Due to its ease in filtering data and creating interactive visuals, a power BI report was created.

{:.image-caption}
| ![BMI](/assets/img/project_obesity/27.png) |
|:-----:|
| *Figure 23. Sankey visual of feature importance on predicting BMI* |

  
{:.image-caption}
| ![BMI](/assets/img/project_obesity/28.png) |
|:-----:|
| *Figure 24. BMI Analysis Dashboard - Combined * |

  
{:.image-caption}
| ![BMI](/assets/img/project_obesity/29.png) |
|:-----:|
| *Figure 25. BMI Analysis Dashboard - Example hover to view distributions* |

  
{:.image-caption}
| ![BMI](/assets/img/project_obesity/30.png) |
|:-----:|
| *Figure 26. BMI Analysis Dashboard - Male and Female * |

  
# Improvements
## BMI
BMI is a traditional measure for obesity. At an individual level, it can be misleading, depending on gender, age, and physicality (De Lorenzo et al., 2019). An alternative is body fat percentage, with modelling techniques being explored to estimate using body variables (Muñoz et al., 2025).

## Data Collection
Survey data collection can lead to low or biased responses, and geography may influence outcomes. Exploring alternative data collection methods could improve insights:   
- Social Media: Facebook marketing adverts to track lifestyle diseases (Araujo et al., 2017) or Twitter data to understand lifestyle choices (Islam and Goldwasser, 2021)
- Wearable Technology: Using smartphones and watches to collect data (Papapanagiotou et al., 2025) or smartwatches to monitor calorie intake (Levi et al., 2025)

## Non-Lifestyle Features
Genetic factors contribute significantly to obesity (Herrera and Lindgren, 2010; Mahmoud, Kimonis and Butler, 2022). This project identified family history was the most important feature in predicting BMI. Machine learning models have been used with genetic profiles to predict obesity levels (Montañez et al., 2017).

# Appendix

{:.image-caption}
| ![BMI](/assets/img/project_obesity/Picture1.png) |
|:-----:|
| *Appendix One. Survey Questions, Possible Answers and Variable Names * |

{:.image-caption}
| ![BMI](/assets/img/project_obesity/Picture2.png) |
|:-----:|
| *Appendix Two. Check of categorical responses within dataset * |

# References

Abbott, L., Lemacks, J. and Greer, T. (2022) ‘Development and Evaluation of a Measure for Social Support Provided by Friends during Lifestyle Management Programs’, Healthcare, 10(5), p. 901. Available at: https://doi.org/10.3390/healthcare10050901.  

Araromi, N. et al. (2024) ‘Managing obesity with lifestyle modification, outcomes, and recommendations’, Medical Research Archives, 12(7). Available at: https://doi.org/10.18103/mra.v12i7.5425.  

Araujo, M. et al. (2017) ‘Using Facebook Ads Audiences for Global Lifestyle Disease Surveillance: Promises and Limitations’, in Proceedings of the 2017 ACM on Web Science Conference. New York, NY, USA: Association for Computing Machinery (WebSci ’17), pp. 253–257. Available at: https://doi.org/10.1145/3091478.3091513.  

Brownlee, J. (2016) XGBoost With Python: Gradient Boosted Trees with XGBoost and scikit-learn. Machine Learning Mastery. Available at: https://repositories.nust.edu.pk/xmlui/bitstream/handle/123456789/16368/xgboost_with_python.pdf?sequence=1&isAllowed=y (Accessed: 27 April 2025).  

CDC (2024) About Chronic Diseases, Chronic Disease. Available at: https://www.cdc.gov/chronic-disease/about/index.html (Accessed: 6 March 2025).  

De Lorenzo, A. et al. (2019) ‘Why primary obesity is a disease?’, Journal of Translational Medicine, 17(1), p. 169. Available at: https://doi.org/10.1186/s12967-019-1919-y.  

Gholamy, A., Kreinovich, V. and Kosheleva, O. (2018) Why 70/30 or 80/20 Relation Between Training and Testing Sets: A Pedagogical Explanation. Available at: https://www.cs.utep.edu/vladik/2018/tr18-09.pdf (Accessed: 8 March 2025).  

Herrera, B.M. and Lindgren, C.M. (2010) ‘The Genetics of Obesity’, Current Diabetes Reports, 10(6), pp. 498–505. Available at: https://doi.org/10.1007/s11892-010-0153-z.  

Islam, T. and Goldwasser, D. (2021) ‘Analysis of Twitter Users’ Lifestyle Choices using Joint Embedding Model’, Proceedings of the International AAAI Conference on Web and Social Media, 15, pp. 242–253. Available at: https://doi.org/10.1609/icwsm.v15i1.18057.  

Levi, I. et al. (2025) ‘Estimation of Food Intake Quantity Using Inertial Signals from Smartwatches’. arXiv. Available at: https://doi.org/10.48550/arXiv.2502.06649.  

Lundberg, S. (2021) Beeswarm plot, SHAP. Available at: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html (Accessed: 29 April 2025).  
 
Mahmood, Md Sohel (2021) Factor Analysis of Mixed Data, Towrds Data Science. Available at: https://towardsdatascience.com/factor-analysis-of-mixed-data-5ad5ce98663c/ (Accessed: 25 April 2025).  
 
Mahmoud, R., Kimonis, V. and Butler, M.G. (2022) ‘Genetics of Obesity in Humans: A Clinical Review’, International Journal of Molecular Sciences, 23(19), p. 11005. Available at: https://doi.org/10.3390/ijms231911005.  
 
Montañez, C.A.C. et al. (2017) ‘Machine learning approaches for the prediction of obesity using publicly available genetic profiles’, in 2017 International Joint Conference on Neural Networks (IJCNN). 2017 International Joint Conference on Neural Networks (IJCNN), pp. 2743–2750. Available at: https://doi.org/10.1109/IJCNN.2017.7966194.  
 
Muñoz, J.-M. et al. (2025) ‘Estimation of total body fat using symbolic regression and evolutionary algorithms’. arXiv. Available at: https://doi.org/10.48550/arXiv.2503.00594.  
 
Palechor, F.M. and Manotas, A.D.L.H. (2019) ‘Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico’, Data in Brief, 25, p. 104344. Available at: https://doi.org/10.1016/j.dib.2019.104344.  
 
Papapanagiotou, V. et al. (2025) ‘A system for objectively measuring behavior and the environment to support large-scale studies on childhood obesity’, IEEE Journal of Biomedical and Health Informatics, 29(4), pp. 2969–2980. Available at: https://doi.org/10.1109/JBHI.2025.3526794.  
 
Potdar, K., Pardawala, Taher, and Pai, Chimney (2017) ‘A Comparative Study of Categorical Variable Encoding Techniques for Neural Network Classifiers’, International Journal of Computer Applications, 175(4), pp. 7–9. Available at: https://doi.org/10.5120/ijca2017915495.  
 
Ruberts, A. (2020) K-Prototypes - Customer Clustering with Mixed Data Types, Well Enough. Available at: https://antonsruberts.github.io/kproto-audience/ (Accessed: 25 April 2025).  
 
Tanaka, K. and Nakanishi, T. (1996) ‘Obesity as a Risk Factor for Various Diseases: Necessity of Lifestyle Changes for Healthy Aging’, Applied Human Science, 15(4), pp. 139–148. Available at: https://doi.org/10.2114/jpa.15.139.  
 
Tiwari, A. and Balasundaram, P. (2023) ‘Public Health Considerations Regarding Obesity’, in StatPearls. Treasure Island (FL): StatPearls Publishing. Available at: http://www.ncbi.nlm.nih.gov/books/NBK572122/ (Accessed: 6 March 2025).  
 
Tuychiev, B. (2023) Learn XGBoost in Python: A Step-by-Step Tutorial, DataCamp. Available at: https://www.datacamp.com/tutorial/xgboost-in-python (Accessed: 27 April 2025).  
 
Tzenios, N. (2023) ‘Obesity as a risk factor for different types of cancer’, EPRA International Journal of Research & Development (IJRD)PRA International Journal of Research and Development, 8(2), pp. 97–100. Available at: https://doi.org/10.36713/epra12421.  
 
World Health Organisation (2022) A healthy lifestyle - WHO recommendations. Available at: https://www.who.int/europe/news-room/fact-sheets/item/a-healthy-lifestyle---who-recommendations (Accessed: 8 March 2025).  
 


