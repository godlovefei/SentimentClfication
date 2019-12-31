#!/usr/bin/env python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def train_valid_test_split(x_data, y_data,
                           validation_size=0.1, test_size=0.1, shuffle=True):
    x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
    valid_size = validation_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size, shuffle=shuffle)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def save_file(x_data, y_data):
    x_train, x_valid, x_test, y_train, y_valid, y_test = train_valid_test_split(x_data, y_data, 0.1, 0.1)

    train = pd.DataFrame({'label': y_train, 'x_train': x_train})
    train.dropna()
    train.to_csv("./data/train.csv", index=False)

    valid = pd.DataFrame({'label': y_valid, 'x_valid': x_valid})
    valid.dropna()
    valid.to_csv("./data/dev.csv", index=False)

    test = pd.DataFrame({'label': y_test, 'x_test': x_test})
    test.dropna()
    test.to_csv("./data/test.csv", index=False)
# label,x_test
def get_predict():
    test_id = []
    test_text = []
    test_label = []
    with open('./data/Test/Test_DataSet.csv', encoding='utf8') as f:
        for idx, i in enumerate(f):
            if idx == 0:
                pass
            else:
                ik = str(i).split(',')[0]
                i = str(i).split(',')[1:]
                test_id.append(ik)
                test_text.append((
                    ','.join(i).replace('\\n', ' ').replace('\\r\\n', '').replace(' ', '').replace('.', '')).strip())
                test_label.append("1")
    test = pd.DataFrame()
    test['label'] = test_label
    # test['id'] = test_id
    test['x_test'] = test_text

    test.to_csv('./data/test.csv',encoding='utf-8',index=False)
    return test


def read_file():
    train_id = []
    train_text = []
    with open('./data/Train/Train_DataSet.csv', encoding='utf8') as f:
        for idx, i in enumerate(f):
            if idx == 0:
                pass
            else:
                ik = str(i).split(',')[0]
                i = str(i).split(',')[1:]
                train_id.append(ik)
                train_text.append((
                    ','.join(i).replace('\\n', ' ').replace('\\r\\n', '').replace(' ', '').replace('.', '')).strip())
    train = pd.DataFrame()
    train['id'] = train_id
    train['text'] = train_text
    train_label = pd.read_csv('./data/Train/Train_DataSet_Label.csv', sep=',')
    train = pd.merge(train, train_label, on=['id'], copy=False)
    # train.to_csv('./data/a.csv',encoding='utf-8',index=False)
    return train


if __name__ == '__main__':
    # train_file = read_file()
    #
    # x_data, y_data = train_file["text"], train_file["label"]
    #
    # save_file(x_data, y_data)

    get_predict()

