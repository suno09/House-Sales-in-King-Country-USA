import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RandomizedLasso, LinearRegression
import itertools
import statistic
from time import time
import sys

# read csv file and show head
dataframe = pd.read_csv('kc_house_data.csv')
# delete id and date columns
dataframe = dataframe.drop(['id', 'date'], axis=1)

# ### Feature Selection ### #
# extract the input and target
dataframe_input = dataframe.drop(['price'], axis=1)
Y = dataframe.price.values

len_cols = dataframe_input.columns.__len__()
max_adj_r2 = 0.
max_indexes_adj_r2 = []

nbr_tests = sum(
    len(list(itertools.combinations(range(len_cols), nbr_features))) for
    nbr_features in range(15, len_cols + 1)
)
current_nbr_tests = 0.

file_write = open("results/lin_reg.txt", "w")
start_time = time()

for nbr_features in range(15, len_cols + 1):
    for indexes_cols in itertools.combinations(range(len_cols), nbr_features):
        current_nbr_tests += 1
        progress = current_nbr_tests * 100. / nbr_tests
        sys.stdout.write("\r")
        sys.stdout.write("Progression [%-100s] -> %.2f %%" % ("=" * int(progress), progress))
        sys.stdout.flush()
        file_write.write(repr(dataframe_input.columns[[indexes_cols]]) + "\n")
        X = dataframe_input.iloc[:, list(indexes_cols)].as_matrix()
        # Linear Regression
        lin_reg = LinearRegression()
        lin_reg.fit(X, Y)
        adj_r2 = statistic.adjusted_r2(
            lin_reg.score(X, Y),
            X.shape[1],
            X.shape[0]
        ) * 100.
        if max_adj_r2 < adj_r2:
            max_adj_r2 = adj_r2
            max_indexes_adj_r2 = indexes_cols

        file_write.write("adjusted r^2 = %.2f %%\n" % adj_r2)
        file_write.write("%s\n" % ("-" * 100))

end_time = time()

print("\nThe best model is %s" % repr(dataframe_input.columns[[max_indexes_adj_r2]]))
print("The best Adjusted R² is %.2f %%" % max_adj_r2)
print("The duration of execution : %.2f seconds" % (end_time - start_time))
print("Number of tests = %d tests" % nbr_tests)

file_write.write("\nThe best model is %s\n" % repr(dataframe_input.columns[[max_indexes_adj_r2]]))
file_write.write("The best Adjusted R² is %.2f %%\n" % max_adj_r2)
file_write.write("The duration of execution : %.2f seconds\n" % (end_time - start_time))
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
