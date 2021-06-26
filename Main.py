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

def create_node(node, df):
    if (len(df['class'].unique().tolist()) > 1) and (df.shape[0] > minimum_count):
        print(minimum_count)

create_node(root, df_train)
