import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('insurance.csv')

# top 3 rows and bottom 3 rows of the dataset
print(data.head(3))
print(data.tail(3))
print(data.shape)   

#EDA 
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# # numeric colummnns 

numeric_col = ['age', 'bmi', 'children', 'charges']
for col in numeric_col:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()  

# countplot for children column

plt.figure(figsize=(6, 4))
sns.countplot(x='children', data=data)
plt.title('Count of Children')
plt.show()

#countplot for smoker column

plt.figure(figsize=(6, 4))
sns.countplot(x='smoker', data=data)
plt.title('Count of Smokers')
plt.show()


for col in numeric_col:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# correlation of the dataset
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(numeric_only=True) , annot=True, cmap='coolwarm')  # annot is used to display the correlation values on the heatmap
plt.title('Correlation Heatmap')
plt.show()


# Data cleaning and preprocessing

#copy of the dataset and analyze the cleaned dataset
df_cleaned = data.copy()
print(df_cleaned.head())
print(df_cleaned.drop_duplicates(inplace=True))
print(df_cleaned.shape)


print(df_cleaned['sex'].value_counts())


# smoker column encoding
df_cleaned['smoker'] = df_cleaned['smoker'].map({'yes': 1, 'no': 0})
print(df_cleaned['smoker'].value_counts())


#  sex column encoding
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 1, 'female': 0})
print(df_cleaned['sex'].value_counts())


# region column encoding
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
print(df_cleaned.head())


df_cleaned= df_cleaned.astype(int)
print(df_cleaned)


# Feature Engineering and Extraction

df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, np.inf], 
    labels=['Underweight', 'Normal weight', 'Overweight', 'Obese'])
print(df_cleaned.head())


df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
print(df_cleaned.head())
print(df_cleaned.astype(int))

# colummns 
print(df_cleaned.columns)


from sklearn.preprocessing import StandardScaler

cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head(3))
print(df_cleaned.astype(int))


from scipy.stats import pearsonr

selected_features   = ['age', 'bmi', 'children', 'smoker', 'sex', 'region_northwest', 'region_southeast', 'region_southwest', 'bmi_category_Normal weight', 'bmi_category_Overweight', 'bmi_category_Obese']
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}

correlations_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation with Charges'])
print(correlations_df)

cat_features = ['smoker', 'sex', 'region_northwest', 'region_southeast', 'region_southwest', 'bmi_category_Normal weight', 'bmi_category_Overweight', 'bmi_category_Obese']

from scipy.stats import chi2_contingency
alpha = 0.05
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)

chi2_results = {}

for feature in cat_features:
    contingency_table = pd.crosstab(df_cleaned[feature], df_cleaned['charges_bin'])
    chi2, p,_,_ = chi2_contingency(contingency_table)
    decision = 'Reject Null (keep feature)' if p < alpha else 'Accept Null (drop feature)'
    chi2_results[feature] = {'Chi2 Statistic': chi2, 'p-value': p, 'Decision': decision}
chi2_results_df = pd.DataFrame.from_dict(chi2_results, orient='index').reset_index().rename(columns={'index': 'Feature'})
print(chi2_results_df)

final_df = df_cleaned[['age', 'bmi', 'children', 'smoker', 'sex',  'region_southeast',  'bmi_category_Obese', 'charges']]
print(final_df.head(3))
print(final_df.astype(int))