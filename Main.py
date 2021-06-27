import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Node import Node

#Read data from dataset and display data
df = pd.read_csv('mushrooms.csv')
print(df.head(20))

#View each column and the number of their unique values
for c in df.columns:
    print('{0} : {1}'.format(c, df[c].unique()))

#Among the columns, it is seen that veil-type column has one unique value
#Hence, veil-type column is dropped
df = df.drop(columns=['veil-type'])
print('veil-type dropped...')

#Split dataset into training and testing
split_point = (7 * df.shape[0]) // 10
df_train = df.iloc[0:split_point,:].reset_index(drop=True)
df_test = df.iloc[split_point:,:].reset_index(drop=True)
print('Split data into training and testing...')
print('Total data: {0}'.format(df.shape[0]))
print('No. of training data: {0} (70%)'.format(df_train.shape[0]))
print('No. of testing data: {0} (30%)'.format(df_test.shape[0]))

#Initializing the root of the decision tree
root = Node()
minimum_count = (5 * df_train.shape[0]) // 1000

def find_entropy(df):
    norm_vals = df['class'].value_counts(normalize=True)
    entropy = norm_vals.apply(lambda x: x * math.log2(x)).sum() * (-1)
    return entropy

def create_node(node, df):
    entropy_class = find_entropy(df)
    features = df.iloc[:,1:].columns.tolist()
    info_gain_max = 0
    splitting_feature = ''
    for f in features:
        print(f)
        entropy_feature = 0
        vals = df[f].unique().tolist()
        for v in vals:
            df_temp = df.loc[df[f] == v].reset_index(drop=True)
            entropy_val = find_entropy(df_temp)
            entropy_feature = entropy_feature + (entropy_val * (df[f].value_counts(normalize=True))[v])
        info_gain = entropy_class - entropy_feature
        if info_gain > info_gain_max:
            info_gain_max = info_gain
            splitting_feature = f
    print('Max {0}'.format(splitting_feature))
    node.split_on = splitting_feature

create_node(root, df_train)
