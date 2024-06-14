 ## First, you need to install !pip install fylearn

import numpy as np
import pandas as pd
from frlearn.base import select_class
from frlearn.classifiers import FRNN
from frlearn.feature_preprocessors import FRFS

for i in range(2,11):
  preprocessor = FRFS(n_features=i)
  behaviour_df = pd.read_csv('/behavior.csv')
  data = behavior_df.copy()
  X = data.drop(columns=['REG','SCO','CLTR1', 'CLTR2', 'CLTR3'])
  y = data['CLTR2']
  X = X.to_numpy()
  y = y.to_numpy()

  preprocessor = FRFS(n_features=i)
  model = preprocessor(X, y)
  X_pre = model(X)
  print(model.selection)

  X_train, X_test, y_train, y_test = train_test_split(X_pre, y, stratify=y, random_state=0)

# Create an instance of the FRNN classifier, construct the model, and query on the test set.
  k_values = [ 3, 5, 7, 9, 11]
  for i, k in enumerate(k_values):
    clf = FRNN(upper_weights=None, lower_weights=None, upper_k=k, lower_k=k)
  #clf = FRNN(preprocessors=(RangeNormaliser(), ))
    model = clf(X_train, y_train)
    scores = model(X_test)

# Convert scores to probabilities and calculate the AUROC.
    probabilities = probabilities_from_scores(scores)
    auroc = roc_auc_score(y_test, probabilities, multi_class='ovo')
    #print('AUROC:', auroc)

# Select classes with the highest scores and calculate the accuracy.
    classes = select_class(scores)
    accuracy = accuracy_score(y_test, classes)
    print('k:', k, 'accuracy:', accuracy)
