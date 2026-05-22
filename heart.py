import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('./heart.csv')

#top 3 and bottom 3 rows of the dataset
print(data.head(3))
print(data.tail(3))
print(data.shape)

print(data.info())  # to get the information about the dataset
print(data.describe())  # to get the statistical summary of the dataset

print(data.isnull().sum())  # to check for missing values
print(data['HeartDisease'].value_counts())  # to check the distribution of the target variable

# #  sub plots for numeric columns
def plotting(var , num):
    plt.subplot(2, 2, num)
    sns.histplot(data[var], kde=True)
plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)
plt.tight_layout()
plt.show()


# countplot for Sex column

plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=data, palette='Set2', linewidth=1.5)
plt.title('Count of Sex')
plt.show()

# countplot for ChestPainType column
plt.figure(figsize=(6, 4))
sns.countplot(x='ChestPainType', data=data, palette='Set2', linewidth=1.5) # chest pain - ATA is  asymptomatic,
# NAP is non-anginal pain,TA is typical angina, and ASY is asymptomatic
plt.title('Count of Chest Pain Type')
plt.show()

# countplot for RestingECG column
plt.figure(figsize=(6, 4))
sns.countplot(x='RestingECG', data=data, palette='Set2', linewidth=1.5) # resting electrocardiographic results - normal, stt abnormality, lv hypertrophy
plt.title('Count of Resting ECG')
plt.show()

# countplot for ExerciseAngina column
plt.figure(figsize=(6, 4))
sns.countplot(x='ExerciseAngina', data=data, palette='Set2', linewidth=1.5) # exercise induced angina - Y is yes and N is no
plt.title('Count of Exercise Induced Angina')
plt.show()

# countplot for ST_Slope column
plt.figure(figsize=(6, 4))
sns.countplot(x='ST_Slope', data=data, palette='Set2', linewidth=1.5) # the slope of the peak exercise ST segment - upsloping, flat, downsloping    
plt.title('Count of ST Slope')
plt.show()


#count plot for HeartDisease column
plt.figure(figsize=(6, 4))
sns.countplot(x='HeartDisease', data=data, palette='Set2', linewidth=1.5) # target variable - 1 is presence of heart disease and 0 is absence of heart disease
plt.title('Count of Heart Disease')
plt.show()


# cleaning data

import numpy as np

data['Cholesterol'] = data['Cholesterol'].replace(0, np.nan)
data['Cholesterol'].fillna(data['Cholesterol'].mean(), inplace=True)

# boxplot for cholesterol column

sns.boxplot(x= "HeartDisease", y="Cholesterol", data=data)
plt.title('Boxplot of Cholesterol by Heart Disease')
plt.show()

# heatmap for correlation of the dataset
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(numeric_only=True) , annot=True, cmap='coolwarm')  # annot is used to display the correlation values on the heatmap
plt.title('Correlation Heatmap')
plt.show()

# Data preprocessing and cleaning


data_encoded = pd.get_dummies(data, drop_first=True)
print(data_encoded.head())
data_encoded = data_encoded.astype(int)
print(data_encoded.info())

from sklearn.preprocessing import StandardScaler
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])
print(data_encoded.head())

