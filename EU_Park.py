#!/usr/bin/env python
# coding: utf-8

# # Sultan Mammadov

# # Import Libraries

# In[2]:


pip install feature_engine


# In[2]:


# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import datasets
from xgboost import XGBRegressor

from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, mean_squared_error,mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Feature Selection
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Feature Engineering
from sklearn.preprocessing import LabelEncoder
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder

# Miscellaneous
from sklearn.preprocessing import minmax_scale
from scipy.stats import zscore


# # Data Import

# In[3]:


# Data Import
attractions = pd.read_csv("EU-park.csv")
customers = pd.read_csv("EU-Park-Customers.csv")
food = pd.read_csv("EU_park_food_sales.csv")


# # Data Cleaning

# In[4]:


# Check if there are any NULL values
print("Attractions", attractions.isnull().sum(), "\n")
print("Customers", customers.isnull().sum(), "\n")
print("Food", food.isnull().sum())


# In[8]:


# Assuming 'customers' is DataFrame
distance_data = customers[' Distance_from_Park_km ']

# Set the style for the plot
sns.set(style = "whitegrid")

# Create a histogram
plt.figure(figsize = (10, 6))
sns.histplot(distance_data, bins = 30, kde = True, color = 'skyblue')

# Add labels and title
plt.title('Distribution of Distance from Park')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# # XGBRegressor Model

# ### Predict wait times and understand what impacts wait times

# In[9]:


# Initial length of the dataset
len_atraction = len(attractions)
print("Length of the initial file: ", len_atraction)

attractions.describe()

# Finding negative numbers in the 'Attraction' column
negative_numbers = attractions[attractions['WaitTime'] < 0]['WaitTime']
print("Percent of negative numbers in WaitTime: ", len(negative_numbers), ", this is ", round(len(negative_numbers)/len(attractions)*100,2), "%")

# Deleted negative numbers
attractions_updated = attractions[attractions['WaitTime'] >= 0]

#Interquartile Range (IQR) Method:

#Calculate the IQR, which is the difference between the 75th percentile (Q3) and the 25th percentile (Q1).
#Define a threshold for outliers, for example, values outside the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
#Values outside this range can be considered outliers and can be treated accordingly (e.g., removed or adjusted).

Q1 = attractions['WaitTime'].quantile(0.25)
Q3 = attractions['WaitTime'].quantile(0.75)
IQR = Q3 - Q1

# Define the threshold for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Boxplot for visual inspection
sns.boxplot(x=attractions['WaitTime'])
plt.show()

# Histogram for visual inspection
sns.histplot(attractions['WaitTime'], bins=30, kde=True)
plt.show()

# Identify and handle outliers
# outliers = attractions_updated[(attractions_updated['WaitTime'] < lower_bound) | (attractions_updated['WaitTime'] > upper_bound)]

# Create a boolean mask for outliers
outlier_mask = (attractions_updated['WaitTime'] < lower_bound) | (attractions_updated['WaitTime'] > upper_bound)

# Remove rows with outliers
attractions_no_outliers = attractions_updated[~outlier_mask]

# Distribution plot for the 'WaitTime' column after removing outliers
sns.histplot(attractions_no_outliers['WaitTime'], bins=30, kde=True)
plt.title('Distribution of WaitTime without Outliers and negative values')
plt.xlabel('WaitTime')
plt.ylabel('Frequency')
plt.show()

print("Percent of deleted values: ", round((100 - len(attractions_no_outliers)/len_atraction*100), 2), "%")

#len(attractions_no_outliers)
# Native labeling
attractions_no_outliers['DayOfWeek'].replace(['Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday'], [3,4,5,6,7,1,2], inplace=True)

# Delete Date column
#attractions_no_outliers.drop(columns="Date", inplace = True)

# Create LabelEncoder instances
le_Attraction = LabelEncoder()
le_Date = LabelEncoder()

# Fit and transform each column
attractions_no_outliers['Attraction'] = le_Attraction.fit_transform(attractions_no_outliers['Attraction'])
attractions_no_outliers['Date'] = le_Date.fit_transform(attractions_no_outliers['Date'])

attractions_no_outliers.head()

# Prepare mask
matrix = attractions_no_outliers.corr().round(2)
mask = np.triu(np.ones_like(matrix, dtype=bool))

# Build
sns.heatmap(matrix, annot = True, vmax = 1, vmin = -1, cmap = 'vlag', mask = mask)

#Train Test Split
x, y = attractions_no_outliers.drop('WaitTime', axis = 1),  attractions_no_outliers['WaitTime']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

#XGBRegressor()
xgbr = xgb.XGBRegressor()

xgbr.fit(xtrain, ytrain)
pred = xgbr.predict(xtest)

score = xgbr.score(xtrain, ytrain)
print("Training score: ", score)

# - cross validataion
scores = cross_val_score(xgbr, xtrain, ytrain, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

mae = mean_absolute_error(ytest, pred)
print('MAE: ' + str(mae))

mse = mean_squared_error(ytest, pred)
print("RMSE: %.4f" % (mse**(1/2.0)))
print("MSE: %.6f" % mse)

#r^2
r2 = r2_score(ytest, pred)

print("R-squared Score:", r2)

# Visualizing the performance
plt.figure(figsize = (10, 6))  # Adjust the figure size as needed
plt.scatter(ytest, pred, alpha = 0.5, c = 'blue', label = 'Actual vs. Predicted')
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], linestyle = '--', color = 'red', linewidth = 2, label = 'Perfect Prediction')
plt.title('XGB.Regressor: Predictions vs. Actual Values')
plt.xlabel('Actual WaitTime')
plt.ylabel('Predicted WaitTime')
plt.legend()
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Calculate feature importance
feature_importance = xgbr.feature_importances_

# Sort features based on importance
sorted_idx = feature_importance.argsort()[::-1]

# Plot feature importance
plt.figure(figsize = (12, 6))

# Bar plot for individual feature importance
plt.subplot(1, 2, 1)
sns.barplot(x = feature_importance[sorted_idx], y = x.columns[sorted_idx], palette = 'viridis')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for XGBRegressor')
plt.grid(True)

# Cumulative importance plot
cumulative_importance = np.cumsum(feature_importance[sorted_idx])
plt.subplot(1, 2, 2)
sns.lineplot(x = range(1, len(cumulative_importance) + 1), y = cumulative_importance, color = 'orange')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.grid(True)

plt.tight_layout()
plt.show()


# ### KMeans Model

# ### If there is a group structure in the purchase then behavior that could drive the special offer for menu combinations

# In[10]:


# Data Import
food = pd.read_csv("EU_park_food_sales.csv")

# Determining the optimal number of clusters using the elbow method
SSE = []
k_range = range(1, 11)  # Trying with 1 to 10 clusters

for k in k_range:
    model = KMeans(n_clusters = k, random_state = 42)
    model.fit(food)
    SSE.append(model.inertia_)

# Plotting the Elbow Method Graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, SSE, marker='o', c='red')

plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Sum Of Squared Distance to Center')
plt.xticks(k_range)

plt.gca().set_facecolor('lightgray')
plt.grid(True)

plt.show()

# Applying PCA to reduce the data to two dimensions
pca = PCA(n_components = 2)
data_2d = pca.fit_transform(food)

# Applying KMeans clustering
model = KMeans(n_clusters = 4, random_state = 42)
clusters = model.fit_predict(data_2d)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c = clusters, cmap='viridis', marker='o')

#plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c = 'red', marker = "D", s = 100) # put centers

# Annotate cluster centers with numbers [0, 1, 2, 3]
for i, center in enumerate(model.cluster_centers_):
    plt.scatter(center[0], center[1], c = 'red', marker = "D", s = 100)
    plt.annotate(str(i), (center[0]+0.6, center[1]+0.6), color = 'orange', fontsize = 40, weight = 'bold')

plt.title('Cluster Visualization of Food Sales')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label = 'Cluster')

plt.gca().set_facecolor('beige')
plt.grid(True)
plt.show()

#Assign Clusters
food['Groups'] = clusters

# Filter groups based on Model Predictions
group_1 = food[food['Groups'] == 0]
group_2 = food[food['Groups'] == 1]
group_3 = food[food['Groups'] == 2]
group_4 = food[food['Groups'] == 3]

# Drop Groups Columns
group_1.drop(['Groups'], axis = 1, inplace = True)
group_2.drop(['Groups'], axis = 1, inplace = True)
group_3.drop(['Groups'], axis = 1, inplace = True)
group_4.drop(['Groups'], axis = 1, inplace = True)

# Create Data Frames showing sum of orders
group_1_df = pd.DataFrame({'Foods': [i for i in group_1.columns], 'Order_Counts': [group_1[i].sum() for i in group_1.columns]})
group_2_df = pd.DataFrame({'Foods': [i for i in group_2.columns], 'Order_Counts': [group_2[i].sum() for i in group_2.columns]})
group_3_df = pd.DataFrame({'Foods': [i for i in group_3.columns], 'Order_Counts': [group_3[i].sum() for i in group_3.columns]})
group_4_df = pd.DataFrame({'Foods': [i for i in group_4.columns], 'Order_Counts': [group_4[i].sum() for i in group_4.columns]})

#Remove orders less than 2
group_1_df = group_1_df[group_1_df['Order_Counts'] > 2]
group_1_df['Order_%'] = round(group_1_df['Order_Counts'] / group_1_df['Order_Counts'].sum()*100,0)

group_2_df = group_2_df[group_2_df['Order_Counts'] > 2]
group_2_df['Order_%'] = round(group_2_df['Order_Counts'] / group_2_df['Order_Counts'].sum()*100,0)

group_3_df = group_3_df[group_3_df['Order_Counts'] > 2]
group_3_df['Order_%'] = round(group_3_df['Order_Counts'] / group_3_df['Order_Counts'].sum()*100,0)

group_4_df = group_4_df[group_4_df['Order_Counts'] > 2]
group_4_df['Order_%'] = round(group_4_df['Order_Counts'] / group_4_df['Order_Counts'].sum()*100,0)

#Shows to 10 Orders
print()
print("Top 10 of Cluster/Menu 1")
print(group_1_df.sort_values('Order_%', ascending = False).head(10))

print()
print("Top 10 of Cluster/Menu 2")
print(group_2_df.sort_values('Order_%', ascending = False).head(10))

print()
print("Top 10 of Cluster/Menu 3")
print(group_3_df.sort_values('Order_%', ascending = False).head(10))

print()
print("Top 10 of Cluster/Menu 4")
print(group_4_df.sort_values('Order_%', ascending = False).head(10))

# Silhouette Score
print()
sc = round(silhouette_score(data_2d, clusters),2)
print('Silhouette Score: '+ str(sc))

# Davies Bouldin Score
print()
dbs = round(davies_bouldin_score(data_2d, clusters),2)
print('Davies Bouldin Score: '+ str(dbs))


# In[ ]:




