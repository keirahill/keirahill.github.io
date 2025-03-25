---
layout: post
title: Obesity Project
subtitle: Which combinations of lifestyle habits have an impact on obesity, and can obesity levels be predicted using these features?
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [clustering, classification]
author: Keira Hill
---

# Introduction
This project examines obesity levels and the influence of lifestyle factors. Over the past several decades, obesity has been increasing (Tzenios, 2023), posing a risk for chronic diseases such as cancer, diabetes, and heart disease (Abbott, Lemacks, and Greer, 2022; Tiwari and Balasundaram, 2023). These conditions contribute to high mortality rates and healthcare costs. Lifestyle factors play a crucial role in managing obesity (Tanaka and Nakanishi, 1996; Araromi et al., 2024; CDC, 2024). Understanding which lifestyle factors most significantly impact obesity can help individuals manage aspects within their control.

# Dataset
The dataset for this project, [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), is sourced from the UCI Machine Learning Repository. Data was collected via an anonymous survey from individuals in Peru, Mexico, and Colombia. A total of 485 respondents answered questions about their nutritional and physical behaviour. Classifications of nutritional status, as defined by WHO and provided in figure 1.4.1.1, were applied based on BMI calculated by weight and height, with an additional split on ‘Pre-obesity’.

![BMI](/assets/img/project_obesity/BMITable.png)


<figure>
  <img src="{{site.url}}/assets/img/project_obesity/BMITable.png" alt="BMI"/>
  <figcaption>Figure 1. (WORLD HEALTH ORGANISATION, 2022).</figcaption>
</figure>


| ![BMI](/assets/img/project_obesity/BMITable.png) |
|:-----:|
| *Figure 1. (WORLD HEALTH ORGANISATION, 2022)* |

Due to an imbalance towards 'normal' weight among respondents, additional data was synthetically generated using SMOTE, increasing the observations to 2,111 (Palechor and Manotas, 2019).

   
The data was explored using Python, revealing no missing values. However, some columns expected to contain object data were numeric, requiring further investigation. 



Most column names were not self-explanatory, so they were renamed for clarity and consistency. Ordinal features were categorized as categorical with specified orders.   

Categorical columns encoded as floats were analysed through histograms. The float values may have been generated during the SMOTE process used for data balancing. These columns were rounded to match the correct number of responses for each feature, allowing them to be interpreted according to the original response options.
An additional attribute for BMI, calculated using the formula weight/height², was added to support the exploratory data analysis (EDA). The final transformation involved splitting the data into training and testing datasets, a crucial step to avoid overfitting and to evaluate model performance. Studies suggest that allocating 20-30% of the data for testing yields optimal results (Gholamy, Kreinovich, and Kosheleva, 2018).
