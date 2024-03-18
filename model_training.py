import numpy as np
from tqdm import tqdm_notebook as tqdm

from qrbm.EncodedQRBM import QRBM


import pandas as pd

import random

from parsing_data import load_data, clean_data, load_config
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from sklearn.linear_model import LogisticRegression

# Variables
n_hidden = 300
epochs = 300
lr = 0.1
split_ratio = 0.8

qpu = False

data = pd.read_csv('encoded_data.csv')
cols = data.columns
cols = [x for x in cols if not x.startswith('Disease')]
cols = [x for x in cols if not '?' in x]
cols = [x for x in cols if not 'Treatment' in x]
data = data[cols]
num_train = int(len(data) * split_ratio)

config_file = 'config.json'
data_file = 'Simplified Dataset.xlsx'
sheet = 'Copy of Cases-Use'

config = load_config(config_file)
raw_data = load_data(data_file, sheet)
df = clean_data(raw_data, config)

encoder = OrdinalEncoder().fit(df[['Disease']])

y = encoder.transform(df[['Disease']])

indices = list(range(len(data)))

# test train split
random.shuffle(indices)
train = indices[:num_train]
test = indices[num_train:]
print(len(train), len(test))
X_train = data.values[train]
X_test = data.values[test]
y_train = y.ravel()[train]
y_test = y.ravel()[test]

# Training
bm = QRBM(X_train[0], n_hidden=n_hidden, qpu=qpu)
bm.tqdm = tqdm
bm.train(X_train, epochs = 300, lr = lr, lr_decay = 0.1)
bm.save('rbm_model')
clf = LogisticRegression(random_state=0)

clf = clf.fit(X_train, y_train)

g = 0
b = 0
for x, y in zip(X_test, y_test.tolist()):
    x = x.reshape(1, -1)
    y = np.array([[y]])
    print(encoder.inverse_transform(y))
    pred_indx = clf.predict(x)
    y_est = pred_indx.reshape((1, 1))
    print(encoder.inverse_transform(y_est), pred_indx)
    if y == y_est:
        g += 1
    else:
        b += 1
    print(clf.predict_proba(x))
    print(clf.score(x, y))
    print('===')
print(g, b, g/(g+b))