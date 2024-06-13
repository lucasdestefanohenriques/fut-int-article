import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from operator import itemgetter

## Load data
data = pd.read_csv("behavior.csv")
data.head()

## RFE with K-means
X = data.drop(columns=['CLTR1', 'CLTR2', 'CLTR3','REG','SCO'])
y = data['CLTR1']
      
min_features_to_select = 1  # Minimum number of features to consider

clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
cv = StratifiedKFold(5)
rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
model = rfecv.fit(X, y)

n_scores = len(rfecv.cv_results_["mean_test_score"])

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("accuracy")
plt.errorbar(
    range(min_features_to_select, n_scores + min_features_to_select),
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"],
)
plt.show()

print("Num Features: %d" % model.n_features_)
print("Feature Ranking: %s" % model.ranking_)
features = X.columns.to_list()
for x, y in (sorted(zip(rfecv.ranking_ , features), key=itemgetter(0))):
    print(x, y)
