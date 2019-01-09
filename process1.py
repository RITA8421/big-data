
import pydoop.hdfs as hdfs
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import sys
import random
import numpy as np

from sklearn import linear_model,  metrics, svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt

# function to read HDFS file into dataframe using PyDoop
import pydoop.hdfs as hdfs
def read_csv_from_hdfs(path, cols, col_types=None):
  files = hdfs.ls(path);
  pieces = []
  for f in files:
    fhandle = hdfs.open(f)
    pieces.append(pd.read_csv(fhandle, names=cols, dtype=col_types))
    fhandle.close()
  return pd.concat(pieces, ignore_index=True)


cols = ['delay','year','month','carrier','airport','arr_flight','arr_del15']
cols_types = {'delay':int,'year':int,'month':int,'carrier':str,'airport':str,'arr_flight':int,'arr_del15':int}
train_data = read_csv_from_hdfs('hdfs://localhost:8020/user/mbds/airline/train_1',cols,cols_types)
