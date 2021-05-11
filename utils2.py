import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def encode_num_from_pred(encode):
    nus = []

    for data in encode:
        num.append(data[0])

    return nums

def encode_label_from_pred(encode):
    label = []

    for data in encode:
        labels.append(data[1])

    return lables

def encode_to_one_hot(encode):
    label_encoder = sklearn.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(numpy.array(encode))
    print(integer_encoded)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(one_hot_encoded)