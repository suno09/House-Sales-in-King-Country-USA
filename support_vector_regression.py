from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd

from matplotlib import pyplot as plt


# #### data preprocessing ####
# read csv file
dataset = pd.read_csv("kc_house_data.csv")
# transform the date of selling to year
dataset.date = [int(str(d)[:4]) for d in dataset.date]
# make the year renovated like is renovated or not
dataset.yr_renovated = [0 if year == 0 else 1 for year in
                        dataset.yr_renovated]
# change the date was sold and built by duration
duration = pd.Series(
    [ys - yb for ys, yb in zip(dataset.date, dataset.yr_built)])
dataset = dataset.assign(duration=duration)
# delete ID, date, yr_built
dataset = dataset.drop(["id", "date", "yr_built"], axis=1)

# the target is in the third column
X = dataset.iloc[:, range(1, dataset.shape[1])].values
# X = dataset.iloc[:50, [18]].values
y = dataset.iloc[:, [0]].values

# #### Support vector regression ####
# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X_svr = sc_X.fit_transform(X)
y_svr = sc_y.fit_transform(y)

svr = SVR()
svr.fit(X_svr, y_svr)

plt.scatter(X[:, -1], y, marker='^')
plt.scatter(
    X[:, -1],
    sc_y.inverse_transform(svr.predict(sc_X.transform(X))),
    marker='o'
)

plt.show()
