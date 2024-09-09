# EU-Park's Business Problems  
      
**Introduction:**  
EU Park is an amusement park that hosts many visitors daily. Recently, visitor satisfaction has declined due to long queues and inadequate restaurant menus.
Factors such as the day of the week, weather, temperature, and attraction popularity contribute to the long wait times.
Additionally, the restaurants are unable to offer customized menus because customer purchase patterns have not been analyzed.
These issues can be resolved by applying data analysis to the historical dataset.

**Objective:** The goal of this project is to address the following business issues:
-How can we predict wait times, and what factors influence them?
-Which combinations of food items are most popular?

**Data Collection:** The dataset is synthetic.
 
**1st Model:** XGBRegressor:

The below correlation matrix displays the relationships between various variables in a dataset. 
It shows correlation coefficients that measure the strength and direction of these relationships. 
There is a moderate relationship between Hour and Wait Time (0.36), indicating that wait times increase as the evening approaches. 
The relationships between other variables are very weak.

![image](https://github.com/user-attachments/assets/b40f3bb2-3ac4-48f6-8ded-ec21675d495a)

**Figure_1.** "Correlation Matrix of all variables"

The graph below shows the predictions of the XGB Regressor compared to the actual values. 
There is noticeable dispersion and outliers around the red prediction line, but a strong linear relationship is evident as most data points are close to the line.
The below performance measurements confirm the strength of this relationship:

Mean Absolute Error (MAE): 2.68

Root Mean Squared Error (RMSE): 4.5

Mean Squared Error (MSE): 20.5

R-squared Score: 0.93

These metrics indicate a strong linear relationship, which may be due to the advanced learning capabilities of the XGB Regressor compared to a standard linear model.

![image](https://github.com/user-attachments/assets/7b94540c-eabb-4bd2-87df-64eebc28f4b7)

**Figure_2.** "Predictions vs Actual Values"

The bar chart displays the contribution level of each feature to the model's output. 
"Hour" is the most important feature, as it has a strong relationship with Wait Time in the correlation matrix as well. 
"Day of Week" and "Attraction Type" are also significant factors influencing the model's output.

![image](https://github.com/user-attachments/assets/88923918-e406-4761-8d2e-ec266f6413bc)

**Figure_3.** "Feature Importance for XGB Regressor"



**2nd Model:** KMeans:

The KMeans model was used to cluster the food sales data of EU Park. 
Since the dataset consists of 30 features, it cannot be visualized on a 2-dimensional graph. 
Therefore, Principal Component Analysis (PCA) was employed to reduce the dataset's dimensionality.

To determine the optimal number of clusters, the Elbow method was used and visualized in the graph below.
The relationship between the number of clusters and the Sum of Squared Distances to the center suggests that 4 clusters are optimal, indicating the need for 4 different menus.

![image](https://github.com/user-attachments/assets/8477480f-4140-414c-98a6-789ae36d1865)

**Figure_4.** "Elbow Method"

The visualization below shows that the 2nd cluster is highly compact and well-separated from the other clusters, making it a particularly reliable cluster for menu creation in this business case. 
Additionally, the other clusters are also compact and exhibit some degree of separation from each other, further supporting their viability for distinct menu combinations.

![image](https://github.com/user-attachments/assets/198741d0-2a4b-4e26-9094-fe0a6dba9e8b)

**Figure_5.** "Cluster Visualization of Food Sales""

The Silhouette Score and Davies-Bouldin Score are well-known metrics for assessing the separation and compactness of data in clusters. Both metrics were used in this analysis, and high scores were achieved:

- Silhouette Score: 0.7 (with 1 being the highest)
- Davies-Bouldin Score: 0.4 (with 0 being the highest)

These high-performance metrics indicate that the new menus, designed based on historical customer data patterns, are likely to meet expectations and satisfy customer needs.


**How the analysis could be improved further ?**
-External Validation
-Experiment with Different Algorithms (Hierarchical Clustering or DBSCAN)
-Hyperparameter Tuning
-Check outliers further
-Explore feature engineering to create new features or transform existing ones


                                        











