import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RandomizedLasso, LinearRegression
import itertools
import statistic
from time import time
import sys
from threading import Thread, Lock
from queue import Queue


print_lock = Lock()


def thread_lin_reg(nbr_tests, indexes_cols, data_pandas, X, Y,
                   queue: Queue):
    """
    parameters in keys:
    :param nbr_tests: nbr of all threads
    :param indexes_cols: the list of chosen columns indexes
    :param data_pandas: the data in DataFrame type of pandas library
    :param X: matrix of input if exist
    :param Y: matrix of output if exist
    :param queue: queue which save the result
    """
    # print(index_test)
    if X is None:
        X = data_pandas.iloc[:, list(indexes_cols)].as_matrix()
    lin_reg = LinearRegression()
    lin_reg.fit(X, Y)
    adj_r2 = statistic.adjusted_r2(
        lin_reg.score(X, Y),
        X.shape[1],
        X.shape[0]
    ) * 100.

    queue.put({
        'indexes_cols': indexes_cols,
        'algo': "Linear Regression",
        'adj_r2': adj_r2
    })

    progress_value = queue.qsize() * 100. / nbr_tests
    with print_lock:
        sys.stdout.write("\r")
        sys.stdout.write("Progression |%-100s| %.2f %%" %
                         ("\u2588" * int(progress_value), progress_value)
                         )
        sys.stdout.flush()


print()
queue = Queue()
threads_lin_reg = []

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

for nbr_features in range(16, len_cols + 1):
    for indexes_cols in itertools.combinations(range(len_cols), nbr_features):
        threads_lin_reg.append(
            Thread(
                target=thread_lin_reg,
                args=(
                    nbr_tests,
                    indexes_cols,
                    dataframe_input,
                    None,
                    Y,
                    queue
                )
            )
        )

start_time = time()

_ = [thread.start() for thread in threads_lin_reg]
_ = [thread.join() for thread in threads_lin_reg]
results = [queue.get() for _ in range(threads_lin_reg.__len__())]
end_time = time()

max_adj_r2 = max([d['adj_r2'] for d in results])
best_lin_reg = min(
    filter(lambda d: d['adj_r2'] == max_adj_r2, results),
    key=lambda d: len(d['indexes_cols'])
)

# print("\nThe best model is %s" % repr(
#     dataframe_input.columns[[best_lin_reg['indexes_cols']]]))
print("\nThe best Adjusted R² is %.2f %%" % max_adj_r2)
print("The duration of execution : %.2f seconds" % (end_time - start_time))
print("Number of tests = %d tests" % nbr_tests)

file_write = open("results/lin_reg.txt", "w")
file_write.write("\nThe best model is %s\n" % repr(
    dataframe_input.columns[[best_lin_reg['indexes_cols']]]))
file_write.write("The best Adjusted R² is %.2f %%\n" % max_adj_r2)
file_write.write(
    "The duration of execution : %.2f seconds\n" % (end_time - start_time))
file_write.write("Number of tests = %d tests" % nbr_tests)

file_write.close()

###############################################################################
# rfe = RFE(estimator=lin_reg, n_features_to_select=1, verbose=3)
# rfe.fit(X, Y)
# ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), names=dataframe_input.columns, order=-1)
#
#
# # In[12]:
#
#
# f, pval = f_regression(X, Y, center=True)
# ranks['corr'] = ranking(ranks=f, names=dataframe_input.columns)
#
#
# # In[13]:
#
#
# # Linear Regression
# lin_reg = LinearRegression()
# lin_reg.fit(X, Y)
# ranks["lin_reg"] = ranking(ranks=lin_reg.coef_, names=dataframe_input.columns)
#
#
# # In[14]:
#
#
# # Decision tree Regression
# # Fitting Decision Tree Regression to the dataset
# tree_reg = DecisionTreeRegressor(random_state = 0)
# tree_reg.fit(X, Y)
# ranks['tree'] = ranking(ranks=tree_reg.feature_importances_, names=dataframe_input.columns)
#
#
# # In[15]:
#
#
# means = {}
# for feature in dataframe_input.columns:
#     means[feature] = np.mean([algo[feature] for algo in ranks.values()])
# ranks['mean'] = means
#
#
# # In[16]:
#
#
# print(" " * 20, "".join(map(lambda k: "%-15s " % k, ranks)))
# for feature in sorted(dataframe_input.columns, key=lambda col: ranks['mean'][col], reverse=True):
#     print("%-20s" % feature, end='')
#     print("".join(map(lambda k: "%-15f " % ranks[k][feature], ranks)))
#
