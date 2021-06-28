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
root = Node(0)
minimum_count = (5 * df_train.shape[0]) // 1000

def find_entropy(df):
    norm_vals = df['class'].value_counts(normalize=True)
    entropy = norm_vals.apply(lambda x: x * math.log2(x)).sum() * (-1)
    return entropy

def create_node(node, df):
    print('At depth {0} and columns {1}'.format(node.depth, df.shape[1]))
    print('No of data {0}'.format(df.shape[0]))
    splitting_feature = ''
    if df.shape[1] > 2:
        #print(node.depth)
        entropy_class = find_entropy(df)
        features = df.iloc[:,1:].columns.tolist()
        info_gain_max = 0
        for f in features:
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
    else:
        splitting_feature = df.columns.tolist()[1]
    print('Max {0}'.format(splitting_feature))
    node.split_on = splitting_feature
    values = df[splitting_feature].unique().tolist()
    print(values)
    for val in values:
        df_temp = df.loc[df[splitting_feature] == val].reset_index(drop=True)
        #print(df_temp.shape[0])
        if len(df_temp['class'].unique().tolist()) == 1:
            print('1st')
            print(val)
            node.class_of[val] = df_temp['class'].unique().tolist()[0]
        elif (df_temp.shape[0] <= minimum_count) or (len(df.columns.tolist()) == 2):
            print('2nd')
            print(val)
            node.class_of[val] = df_temp['class'].mode()
        else:
            print('3rd')
            print(val)
            node.is_leaf = False
            node.class_of[val] = Node(node.depth + 1)
            df_temp_next = df_temp.drop(columns=[splitting_feature]).reset_index(drop=True)
            create_node(node.class_of[val], df_temp_next)
            print('Back to depth {0}'.format(node.depth))
            print('columns are {0}'.format(df.shape[1]))
        #print(val)
        #print(node.class_of[val])

create_node(root, df_train)
