#ML Process
import pandas as pd
origin = pd.read_csv("/home/mbds/Documents/train_1/part-m-00000",delimiter=",")
data_2018 = pd.read_csv("/home/mbds/Documents/airline/2018_1/part-m-00000",delimiter=",")
cols = ['arrDelay','year','month','arr_flights','arr_del15']
train_y = origin['arr_del15']>15
train_x = origin[cols]

test_y = data_2018['arr_del15']>15
test_x = data_2018[cols]


import numpy as np

from sklearn import linear_model, metrics, svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import pandas as pd
import matplotlib.pyplot as plt

# Create logistic regression model with L2 regularization
clf_lr = linear_model.LogisticRegression(penalty='l2', class_weight='balanced')
clf_lr.fit(train_x, train_y)

# Predict output labels on test set
pr = clf_lr.predict(test_x)

# display evaluation metrics
cm = confusion_matrix(test_y, pr)
print("Confusion matrix")
print(pd.DataFrame(cm))
report_lr = precision_recall_fscore_support(list(test_y), list(pr), average='micro')
print "\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_lr[0], report_lr[1], report_lr[2], accuracy_score(list(test_y), list(pr)))

