import pandas as pd
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

## Load data
data = pd.read_csv("behaviorchi2.csv")
data.head()

## Chi2 with K-means output
X = data.drop(columns=['CLTR1', 'CLTR2', 'CLTR3', 'REG', 'SCO'])
y = data['CLTR1']

chi_scores = chi2(X,y)

p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()
plt.xlabel('Feature', fontsize=12)
plt.ylabel('P-value', fontsize=12)
plt.show()
