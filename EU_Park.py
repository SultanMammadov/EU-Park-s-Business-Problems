pip install feature_engine

# Import Libraries

# Data Manipulation
import os
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning 
from sklearn.model_selection import (train_test_split, cross_val_score, KFold)
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree as sklearn_plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import (XGBRegressor, XGBClassifier, plot_tree as xgb_plot_tree)

# Model Evaluation
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             balanced_accuracy_score, mean_squared_error, r2_score,
                             silhouette_score, davies_bouldin_score, mean_absolute_error,
                             mean_squared_error,
                             r2_score)
# Feature Selection
from sklearn.feature_selection import (mutual_info_classif, mutual_info_regression)

# Feature Engineering
from sklearn.preprocessing import (LabelEncoder, minmax_scale)
from feature_engine.encoding import (RareLabelEncoder, OrdinalEncoder)

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Statistics
from scipy.stats import zscore

# Data Import
path = input("Enter the path where the input files are saved: ")

# attractions in EU-park file
attractions_file = "EU-park.csv"
attractions_full_path = os.path.join(path, attractions_file)
attractions = pd.read_csv(attractions_full_path)

# food order details in EU_park_food_sales
food_file = "EU_park_food_sales.csv"
food_full_path = os.path.join(path, food_file)
food = pd.read_csv(food_full_path)


# Data Cleaning

# Check if there are any NULL values
print("Attractions", attractions.isnull().sum(), "\n")
print("Food", food.isnull().sum())


# Before Removing Negative Waiting Times
initial_length = len(attractions)

print(f"Initial dataset length: {initial_length}")

# Describe dataset
display(attractions.describe())

# Remove Negative Wait Times
negative_values = attractions.loc[attractions["WaitTime"] < 0, "WaitTime"]

negative_percentage = len(negative_values) / initial_length * 100

print(f"Negative WaitTime values: {len(negative_values)} " f"({negative_percentage:.2f}%)")

# Keep only valid WaitTime values
attractions_clean = attractions[attractions["WaitTime"] >= 0].copy()

# Outlier Detection using IQR

Q1 = attractions_clean["WaitTime"].quantile(0.25)
Q3 = attractions_clean["WaitTime"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")


# Visualize before removing outliers
plt.figure(figsize=(10,4))

sns.boxplot(x=attractions_clean["WaitTime"])

plt.title("WaitTime Distribution - Before Outlier Removal")
plt.show()


plt.figure(figsize = (10, 5))
sns.histplot(attractions_clean["WaitTime"], bins = 30, kde = True)

plt.title("WaitTime Distribution")
plt.xlabel("Wait Time")
plt.ylabel("Frequency")
plt.show()


# Remove outliers
outlier_mask = ((attractions_clean["WaitTime"] < lower_bound) | (attractions_clean["WaitTime"] > upper_bound))
outliers_removed = attractions_clean[outlier_mask]

attractions_clean = attractions_clean[~outlier_mask].copy()
print(f"Outliers removed: {len(outliers_removed)}")

deleted_percentage = ((initial_length - len(attractions_clean)) / initial_length * 100)
print(f"Total removed values: {deleted_percentage:.2f}%")


# Distribution After Cleaning

plt.figure(figsize = (10, 5))

sns.histplot(attractions_clean["WaitTime"], bins = 30, kde = True)

plt.title("WaitTime Distribution After Cleaning")

plt.xlabel("Wait Time")
plt.ylabel("Frequency")

plt.show()


# Feature Engineering

# Convert days into numerical values
day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7}

attractions_clean["DayOfWeek"] = (attractions_clean["DayOfWeek"].replace(day_mapping))

# Label Encoding for categorical variables
label_encoder_attraction = LabelEncoder()

attractions_clean["Attraction"] = (label_encoder_attraction.fit_transform(attractions_clean["Attraction"]))
attractions_clean["Rain"] = (label_encoder_attraction.fit_transform(attractions_clean["Rain"]))
attractions_clean["Date"] = (label_encoder_attraction.fit_transform(attractions_clean["Date"]))

display(attractions_clean.head())

# Correlation Matrix
corr_matrix = attractions_clean.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


#Model - XGBoost Regression
#predict wait times and understand what impacts wait times¶

# Prepare Data for Model

X = attractions_clean.drop("WaitTime", axis = 1)

y = attractions_clean["WaitTime"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# XGBoost Regression Model

xgbr = XGBRegressor(random_state = 42)

xgbr.fit(X_train, y_train)

predictions = xgbr.predict(X_test)
predictions

# Model Evaluation
train_score = xgbr.score( X_train, y_train)
print(f"Training R² Score: {train_score:.4f}")

cv_scores = cross_val_score(xgbr, X_train, y_train, cv = 5)
print(f"Cross Validation Mean Score: {cv_scores.mean():.4f}")

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# Actual vs Predicted Plot

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")

plt.title("XGBoost Regression: Actual vs Predicted")
plt.xlabel("Actual WaitTime")
plt.ylabel("Predicted WaitTime")
plt.show()

# Feature Importance

feature_importance = (xgbr.feature_importances_)
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
importance_df = (importance_df.sort_values("Importance", ascending=False))

plt.figure(figsize=(10,6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("XGBoost Feature Importance")
plt.show()


# Cumulative Feature Importance

importance_df["Cumulative"] = (importance_df["Importance"].cumsum())

plt.figure(figsize=(8,5))
sns.lineplot(data=importance_df, x=range(1, len(importance_df)+1), y="Cumulative")
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Importance")
plt.title("Cumulative Feature Importance")
plt.grid(True)
plt.show()


# Preparation and Model - KMeans
# if there is a group structure in the purchase
# behavior that could drive the special offer for menu combinations

# Determining the optimal number of clusters using the elbow method

SSE = []
k_range = range(1, 11)  # Trying with 1 to 10 clusters

for k in k_range:
    model = KMeans(n_clusters = k, random_state = 42)
    model.fit(food)
    SSE.append(model.inertia_)

# Plotting the Elbow Method Graph
plt.figure(figsize = (10, 6))
plt.plot(k_range, SSE, marker = 'o', c = 'red')

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
plt.scatter(data_2d[:, 0], data_2d[:, 1], c = clusters, cmap = 'viridis', marker = 'o')

# plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c = 'red', marker = "D", s = 100) # put centers

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














