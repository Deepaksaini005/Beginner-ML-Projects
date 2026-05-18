import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('heart.csv')

#top 3 and bottom 3 rows of the dataset
print(data.head(3))
print(data.tail(3))
print(data.shape)

