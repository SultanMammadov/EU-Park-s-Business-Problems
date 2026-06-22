# EU-Park's Business Problems  
      
**Introduction:**  
EU Park is an amusement park that hosts many visitors daily. Recently, visitor satisfaction has declined due to long queues and inadequate restaurant menus.
Factors such as the day of the week, weather, temperature, and attraction popularity contribute to the long wait times.
Additionally, the restaurants are unable to offer customized menus because customer purchase patterns have not been analyzed.
These issues can be resolved by applying data analysis to the historical dataset.

**Objective:** The goal of this project is to address the following business problems:

1. How can we predict wait times, and what factors influence them?
2. Which combinations of food items are most popular?

**Data Collection:** The dataset is synthetic.

The data is clean and does not need any imputation. The outliers are visualised in the box plot below, before removal.

<img width="799" height="391" alt="download" src="https://github.com/user-attachments/assets/019cde44-f2d1-45cb-a775-27a694b853eb" />

Figure_1. "Before Outliers Removal"

The figure below shows the distribution of waiting times after removing outliers.

<img width="879" height="473" alt="download" src="https://github.com/user-attachments/assets/ba7db7a6-ed65-4033-8fac-2d1961e1d12f" />

Figure_2. "After Outliers Removal"

Before cleaning, the wait time data showed extreme outliers reaching up to 20,000, with most observations clustered near zero. 
After removing these outliers, the distribution becomes much clearer — the majority of wait times fall between 4 and 5, with a secondary concentration observed between 15 and 35. The data still exhibits a right-skewed, bimodal pattern, hinting at two distinct subgroups that are worth exploring further in the analysis.

 
The below correlation matrix displays the relationships between various variables in a dataset. 
It shows correlation coefficients that measure the strength and direction of these relationships. 
There is a moderate relationship between Hour and Wait Time (0.36), indicating that wait times increase as the evening approaches. 
The relationships between other variables are very weak.

<img width="711" height="609" alt="download" src="https://github.com/user-attachments/assets/94baee23-365c-42dd-9b1a-c6cbddace034" />

**Figure_3.** "Correlation Matrix of all variables"

**Feature engineering:**

Feature engineering was applied to transform categorical variables into numerical format suitable for machine learning models. Days of the week were mapped to ordered numerical values from 1 (Monday) to 7 (Sunday), preserving their natural sequence. Label Encoding was then applied to the remaining categorical columns — Attraction, Rain, and Date — converting each unique category into a corresponding numerical value. This step ensures that the model can interpret and process all variables effectively, as most algorithms require numerical input. Together, these transformations prepare the dataset for reliable and accurate model training.


**1st Model:** XGBRegressor:

The graph below shows the predictions of the XGB Regressor compared to the actual values. 
There is noticeable dispersion and outliers around the blue prediction line, but a strong linear relationship is evident as most data points are close to the line.
The below performance measurements confirm the strength of this relationship:

Mean Absolute Error (MAE): 2.68

Root Mean Squared Error (RMSE): 4.5

Mean Squared Error (MSE): 20.5

R-squared Score: 0.93

These metrics indicate a strong linear relationship, which may be due to the advanced learning capabilities of the XGB Regressor compared to a standard linear model.

<img width="690" height="550" alt="download" src="https://github.com/user-attachments/assets/8e8007c3-7fd0-4939-a4a0-bee50b3e47d0" />

**Figure_4.** "Predictions vs Actual Values"

The bar chart displays the contribution level of each feature to the model's output. 
"Hour" is the most important feature, as it has a strong relationship with Wait Time in the correlation matrix as well. 
"Day of Week" and "Attraction Type" are also significant factors influencing the model's output.

<img width="703" height="473" alt="download" src="https://github.com/user-attachments/assets/7fbb139e-1b32-4c6e-b74c-6a5a1eed6704" />
<img width="922" height="550" alt="download" src="https://github.com/user-attachments/assets/4b0a0a2a-1818-4fc8-8073-80db6dc44bfb" />


**Figure_5.** "Feature Importance for XGB Regressor"



**2nd Model:** KMeans:

The KMeans model was used to cluster the food sales data of EU Park. 
Since the dataset consists of 30 features, it cannot be visualized on a 2-dimensional graph. 
Therefore, Principal Component Analysis (PCA) was employed to reduce the dataset's dimensionality.

To determine the optimal number of clusters, the Elbow method was used and visualized in the graph below.
The relationship between the number of clusters and the Sum of Squared Distances to the center suggests that 4 clusters are optimal, indicating the need for 4 different menus.

![image](https://github.com/user-attachments/assets/8477480f-4140-414c-98a6-789ae36d1865)

**Figure_6.** "Elbow Method"

The visualization below shows that the 2nd cluster is highly compact and well-separated from the other clusters, making it a particularly reliable cluster for menu creation in this business case. 
Additionally, the other clusters are also compact and exhibit some degree of separation from each other, further supporting their viability for distinct menu combinations.

![image](https://github.com/user-attachments/assets/198741d0-2a4b-4e26-9094-fe0a6dba9e8b)

**Figure_7.** "Cluster Visualization of Food Sales""

The Silhouette Score and Davies-Bouldin Score are well-known metrics for assessing the separation and compactness of data in clusters. Both metrics were used in this analysis, and high scores were achieved:

- Silhouette Score: 0.7 (with 1 being the highest)
- Davies-Bouldin Score: 0.4 (with 0 being the highest)

These high-performance metrics indicate that the new menus, designed based on historical customer data patterns, are likely to meet expectations and satisfy customer needs.


**Business problems:**

1. How can we predict wait times, and what factors influence them?
   
The XGBoost Regression model enables the business to predict future wait times in advance, allowing proactive measures to be taken to maintain high customer satisfaction.
The feature importance analysis further revealed that the hour of the day, day of the week, and the specific attraction are the primary drivers of wait time, providing actionable insights for better crowd and queue management.

2. Which combinations of food items are most popular?

The KMeans model identified 4 distinct food combination clusters, each representing a different customer preference group:

- Cluster 1 – Health-Conscious Visitors predominantly order Water, Veggie Burger, Salad, and Juice, each accounting for 15–17% of orders. 
This group clearly favours light, healthy options.

- Cluster 2 – Snack & Drinks Lovers gravitate towards Nachos, Pretzel, Cocktail, Wine, and Beer, each contributing around 14–15% of orders. 
This group prefers savoury snacks paired with alcoholic beverages.

- Cluster 3 – Classic Comfort Food customers consistently order Hamburger, Cheeseburger, Fries, and Soft Drink, 
each at approximately 16% of orders — a traditional fast-food combination that remains highly popular.

- Cluster 4 – Sweet Treat Seekers favour Cookie, Coffee, Tea, and Cupcake, each ranging between 15–16% of orders, suggesting a group that visits primarily for desserts and hot beverages.


**How the analysis could be improved further ?**
- External Validation
- Experiment with Different Algorithms (Hierarchical Clustering or DBSCAN)
- Hyperparameter Tuning
- Check outliers further
- Explore feature engineering to create new features or transform existing ones      
