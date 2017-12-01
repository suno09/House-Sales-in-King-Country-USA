from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# #### data preprocessing ####
# read csv file
dataset = pd.read_csv("kc_house_data.csv")
# transform the date of selling to year
dataset.date = [int(str(d)[:4]) for d in dataset.date]
# delete ID
dataset = dataset.drop("id", axis=1)

# the target is in the third column
X = dataset.iloc[:, [0, *range(2, dataset.shape[1])]].values
y = dataset.iloc[:, 1].values

# Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X[:, 12]), max(X[:, 12]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X[:, 12], y, color='red')
plt.plot(X[:, 12], regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
