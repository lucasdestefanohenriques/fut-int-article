import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

## Load data
data = pd.read_csv("behavior.csv")
data.head()

X = data.drop(columns=['CLTR1', 'CLTR2', 'CLTR3', 'REG', 'SCO'])
y = data['CLTR3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_imp

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()

accuracy = clf.score(X_test, y_test)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=5)
scores = cross_val_score(clf, X, y, cv=cv)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
