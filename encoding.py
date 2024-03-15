import numpy as np
from sklearn import preprocessing

def binary_encode(values):
    uniq = sorted(list(set(values)))
    result = np.array(values == uniq[0]).astype(int)
    return [result]


def trinary_encode(values, col, how='signal', s_prefix='N'):
    uniq = sorted(list(set(values)))
    if how == 'signal':
        s = [x for x in uniq if x.startswith(s_prefix)]
        if len(s) == 1:
            sig = s[0]
        else:
            sig = uniq[0]
        alt = [x for x in uniq if x != sig]
        bit0 = np.array(values != sig).astype(int)
        bit1 = np.array(values == alt[0]).astype(int) - np.array(values != sig).astype(int)
    return [f'{col}-{sig}', f'{col}-{alt[0]}_vs_{alt[1]}'], [bit0, bit1]


def one_hot_encode(values, col):
    uniq = sorted(list(set(values)))
    names = []
    bits = []
    for u in uniq:
        names.append(f'{col}-{u}')
        bits.append(np.array(values != u).astype(int))
    return names, bits


def bit_encode(data, col):
    uniq = set(data[col])
    if len(uniq) == 2:
        return [col], binary_encode(data[col])
    elif len(uniq) == 3:
        return trinary_encode(data[col], col)
    else:
        return one_hot_encode(data[col], col)


def make_encoders(data, config):
    encoders = {k: preprocessing.LabelEncoder() for k in config['col_types'].keys()}
    for k in encoders.keys():
        encoders[k].fit(data[k])
    return encoders
