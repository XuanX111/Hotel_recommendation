import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import h5py
import os
import data_io

# read in RF results
predshf = data_io.load_model('rf')
preds = 0.31*normalize(predshf.value, norm='l1', axis=1)

#read in xgb results
predshf = data_io.load_model('xgb')
preds += 0.39*normalize(predshf.value, norm='l1', axis=1)

#read in sgd results
predshf = data_io.load_model('sgd')
preds += 0.27*normalize(predshf.value, norm='l1', axis=1)

#read in nb results
predshf = data_io.load_model('nb')
preds += 0.03*normalize(predshf.value, norm='l1', axis=1)


print('generating submission')
# based on the probability give the final recommending hotel clusters
df_pred = pd.DataFrame(preds)
preds = []
for index, row in df_pred.iterrows():
    preds.append(list(row.nlargest(5).index))
print(preds)