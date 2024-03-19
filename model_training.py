import numpy as np
from tqdm import tqdm_notebook as tqdm

from qrbm.EncodedQRBM import QRBM

import pandas as pd

import random

from parsing_data import load_data, clean_data, load_config
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LogisticRegression

# Variables
n_hidden = 50
epochs = 30
lr = 0.1
split_ratio = 0.8
n_folds = 5

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

Y = encoder.transform(df[['Disease']])

indices = list(range(len(data)))

#n-folds cross validation
scores = []
for i in range(n_folds):
    # test train split
    random.shuffle(indices)
    train = indices[:num_train]
    test = indices[num_train:]
    X_train = data.values[train]
    X_test = data.values[test]
    y_train = Y.ravel()[train]
    y_test = Y.ravel()[test]

    # Training
    bm = QRBM(X_train[0], n_hidden=n_hidden, qpu=qpu)
    bm.tqdm = tqdm
    bm.train(X_train, epochs = epochs, lr = lr, lr_decay = 0.1)
    bm.save(f'rbm_model_{i}')

    encoded_train = np.array([bm.sample_hidden(x) for x in X_train])
    encoded_test = np.array([bm.sample_hidden(x) for x in X_test])

    clf = LogisticRegression(random_state=0)

    clf = clf.fit(encoded_train, y_train)

    g = 0
    b = 0
    for x, y in zip(encoded_test, y_test.tolist()):
        x = x.reshape(1, -1)
        y = np.array([[y]])
        # print(encoder.inverse_transform(y))
        pred_indx = clf.predict(x)
        y_est = pred_indx.reshape((1, 1))
        # print(encoder.inverse_transform(y_est), pred_indx)
        if y == y_est:
            g += 1
        else:
            b += 1
            print(encoder.inverse_transform(y), encoder.inverse_transform(y_est))
        # print(clf.predict_proba(x))
        # print(clf.score(x, y))
        # print('===')
    print(g, b, g/(g+b))
    scores.append([g/(g+b)])
print(np.mean(scores))
