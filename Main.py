import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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






