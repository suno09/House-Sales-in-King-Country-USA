# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RandomizedLasso, LinearRegression
import itertools
import statistic
from time import time
import sys
from threading import Thread
from queue import Queue

# In[6]:


progress = 0


def show_progress(total):
    global progress
    progress += 1
    progress_value = progress * 100. / total
    sys.stdout.write("\r")
    sys.stdout.write("Progression |%-100s| %.2f %%" %
                     ("\u2588" * int(progress_value), progress_value)
                     )
    sys.stdout.flush()


# In[7]:


# read csv file and show head
dataframe = pd.read_csv('kc_house_data.csv')
# delete id and date columns
dataframe = dataframe.drop(['id', 'date'], axis=1)

# ### Feature Selection ### #
# extract the input and target
dataframe_input = dataframe.drop(['price'], axis=1)
Y = dataframe.price.values

len_cols = dataframe_input.columns.__len__()
nbr_tests = sum(
    len(list(itertools.combinations(range(len_cols), nbr_features))) for
    nbr_features in range(16, len_cols + 1)
)
current_nbr_tests = 0.
results = []

# In[15]:


current_nbr_tests = 0
start = time()
for nbr_features in range(16, len_cols + 1):
    for indexes_cols in itertools.combinations(range(len_cols), nbr_features):
        current_nbr_tests += 1
        X = dataframe_input.iloc[:, list(indexes_cols)].as_matrix()
        lin_reg = LinearRegression()
        lin_reg.fit(X, Y)
        adj_r2 = statistic.adjusted_r2(
            lin_reg.score(X, Y),
            X.shape[1],
            X.shape[0]
        ) * 100.
        results.append({
            'index_test': current_nbr_tests,
            'indexes_cols': indexes_cols,
            'algo': "Linear Regression",
            'adj_r2': adj_r2
        })
        progress_value = current_nbr_tests * 100. / nbr_tests
        sys.stdout.write("\r")
        sys.stdout.write("Progression |%-100s| %.2f %%" %
                         ("\u2588" * int(progress_value), progress_value)
                         )
        sys.stdout.flush()

max_adj_r2 = max([d['adj_r2'] for d in results])
best_lin_reg = min(
    filter(lambda d: d['adj_r2'] == max_adj_r2, results),
    key=lambda d: len(d['indexes_cols'])
)
end = time()
print("\ntime %.2f seconds" % (end - start))

