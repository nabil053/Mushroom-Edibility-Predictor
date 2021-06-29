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
df = df.sample(frac=1)

#Split dataset into training and testing
split_point = (7 * df.shape[0]) // 10
df_train = df.iloc[0:split_point,:].reset_index(drop=True)
df_test = df.iloc[split_point:,:].reset_index(drop=True)
print('Split data into training and testing...')
print('Total data: {0}'.format(df.shape[0]))
print('No. of training data: {0} (70%)'.format(df_train.shape[0]))
print('No. of testing data: {0} (30%)'.format(df_test.shape[0]))

#Initializing the root of the decision tree and the minimum number of data to split on
root = Node(0)
minimum_count = (5 * df_train.shape[0]) // 1000

#Function to find the entropy based on class
def find_entropy(df):
    norm_vals = df['class'].value_counts(normalize=True)
    entropy = norm_vals.apply(lambda x: x * math.log2(x)).sum() * (-1)
    return entropy

#Function for generating the decision tree
def generate_tree(node, df):
    #If there are at least two features, find entropy of each feature and determine the corresponding information gain
    if df.shape[1] > 2:
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
                node.split_on = f
    #If there is only one feature, select that feature to classify on
    else:
        node.split_on = df.columns.tolist()[1]
    values = df[node.split_on].unique().tolist()
    #Classify based on the selected feature
    for val in values:
        df_temp = (df.loc[df[node.split_on] == val]).reset_index(drop=True)
        #If there is only one class, associate the feature value with that class
        if len(df_temp['class'].unique().tolist()) == 1:
            node.class_of[val] = df_temp['class'].unique().tolist()[0]
        #Else if there is too few data on the dataset or just one class, assign the class value of highest frequency to feature value
        elif (df_temp.shape[0] <= minimum_count) or (df.shape[1] == 2):
            node.class_of[val] = df_temp['class'].mode()[0]
        #Otherwise generate sub-table based on feature value and recursively generate tree on that sub-table
        else:
            node.is_leaf = False
            node.class_of[val] = Node(node.depth + 1)
            df_temp_next = df_temp.drop(columns=[node.split_on]).reset_index(drop=True)
            generate_tree(node.class_of[val], df_temp_next)

#Generate the decision tree
print('Generating tree...')
generate_tree(root, df_train)
print('Tree has been generated.')

