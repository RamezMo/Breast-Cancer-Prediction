# Breast-Cancer-Predictor-Model : Machine Learning Model for Forecasting Breast Cancer with nearly 94% accuracy

## Introduction

Early detection and diagnosis are crucial in the fight against breast cancer, which is one of the most common cancers among women worldwide. This project utilizes machine learning techniques to predict breast cancer diagnosis. By analyzing attributes such as texture, perimeter, area, and smoothness, the model aims to assist medical professionals in making accurate and timely diagnoses. Achieving high accuracy in breast cancer prediction ensures early intervention and improves patient outcomes significantly.
## Table of Contents

1. [Introduction](#introduction)
2. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
3. [Data Preparation](#Data-Preparation)
4. [Model Evaluation](#Evaluate-models)



#Exploratory Data Analysis



### Importing Necessary Libraries
Importing Necessary Libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
```

##Load Breast Cancer Wisconsin (Diagnostic) Dataset
Load the dataset Load the dataset into DataFrame.

```python
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
```

## Display the first few rows of the dataset
inspect the first few rows to understand its structure.
```python
df.head(10)
```

![image](https://github.com/user-attachments/assets/ecda9cd1-cf14-47dc-bf68-ebbb31870175)


## How Many Instances and Features ?
Display the number of rows and columns in the dataset
```python
df.shape
```

#Exploratory-Data-Analysis

##Display Variables DataType and count of non-NULL values
```python
df.info()
```
![image](https://github.com/user-attachments/assets/4abba599-f93e-4b19-bd45-c9c3f83a02ac)

All variables DataType are Numerical except 'diagnosis' we will handle later
All variables Have 0 NULLs except 'Unnamed: 32' only have NULLs so we will drop it

## Discover The Categorical Variable
Show the Distribution of values in 'diagnosis' Column
```python
df.diagnosis.value_counts()
```


## Summary statistics

```python
df.describe()
```

![image](https://github.com/user-attachments/assets/bcaba26b-2972-43ae-be49-24cd4f88cb42)



## Create a correlation heatmap for the subset of features
#radius_mean have very strong positiver correlation with perimeter_mean,area_mean,radius_worst,area_worst and perimeter_worst
#perimeter_mean have very strong positiver correlation with radius_worst,area_worst, perimeter_worst and area_mean
#We will drop These Correlated Features using PCA Dimensionality Reduction Method

```python
plt.figure(figsize=(25,25))
sns.heatmap(df.corr(),annot=True,mask=np.triu(np.ones_like(df.corr(),dtype=bool)),fmt='.2f',annot_kws={'size':12})
```
![image](https://github.com/user-attachments/assets/2e1721fa-c0c5-424c-98f2-ee9f4307fa54)


# Data Preparation
## Data Transformation
Convert categorical variables into numerical ones for machine learning models.

```python
#we will convert it into 1 for M and 0 for B
df.replace({'M':1,'B':0},inplace=True)
```


#Now Let's Drop Columns that are not important like 'id' since it will introduce no information and 'Unnamed: 32' since it have only NULLs
```python
df.drop(['id','Unnamed: 32'] , axis=1 , inplace=True)
```

##Feature Selection
keep  features that have the highest importance Using PCA for Dimendionality Reduction

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Initialize PCA with the desired variance threshold
pca = PCA(0.97)

# Fit PCA to the scaled data
x_pca = pca.fit_transform(x_scaled)

# Convert transformed array back to DataFrame
x_pca = pd.DataFrame(data=x_pca, columns=[f"PC{i+1}" for i in range(x_pca.shape[1])])
```
![image](https://github.com/user-attachments/assets/8f355ab6-71c4-4628-ab60-28017f4fec83)



##Detect Outliers
Outliers are data points that deviate significantly from other observations in a dataset, potentially impacting statistical analyses and model performance by skewing results or introducing noise.handling outliers is critical to ensure data integrity and reliable analytical outcomes.
Use BoxPlot to See if we have Outliers
```python
# Plot boxplot to visualize distribution of features
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/user-attachments/assets/ce1289f4-6528-4c9c-b135-55e9a6d5dc09)


##Handle Outliers
```python

# Calculate the IQR for each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define a threshold to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df_clean = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Verify the shape of the dataframe after removing outliers
print("Original DataFrame shape:", df.shape)
print("DataFrame shape after removing outliers:", df_clean.shape)

```

Now let's Show the boxplot Distribution after Handling outliers
```python
# Plot boxplot to visualize distribution of features
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_clean)
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
plt.show()
```

![image](https://github.com/user-attachments/assets/35f7b1e3-c73a-42c1-b55a-7e528d14c594)



## Split data into Features and Target sets

```python
x = df_clean.drop('diagnosis',axis=1)
y = df_clean.diagnosis
```

## Split data into training, Validation and testing sets

```python
# Step 1: Split data into training and holdout (validation + testing) sets
x_train, x_holdout, y_train, y_holdout = train_test_split(x, y, test_size=0.5, random_state=42)

# Step 2: Further split holdout set into validation and testing sets
x_val, x_test, y_val, y_test = train_test_split(x_holdout, y_holdout, test_size=0.5, random_state=42)



# Print the shapes of each set to verify the splits
print("Training set:", x_train.shape, y_train.shape)
print("Validation set:", x_val.shape, y_val.shape)
print("Testing set:", x_test.shape, y_test.shape)


```
![image](https://github.com/user-attachments/assets/02ed3595-46b2-4c2e-89e5-3c25811e2b83)


## Evaluate models
after training the model and predicting it on test data it makes accuracy of nearly 94% Using LogisticRegression Machine Learning Algorithm

##Creating Confusion Matrix
it shows the predicted values Distribution

```python
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
```
![image](https://github.com/user-attachments/assets/aad814bf-8157-410e-8481-fa50466be670)

