from sklearn.linear_model import LinearRegression
import pandas as pd

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

# #### Multiple Linear regression ####
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# plot with multiple figures
row_size = 10
column_index_view = 14
len_figs, idx_fig_x, idx_fig_y = 36, 6, 6

fig = plt.figure()
for idx_fig in range(len_figs):
    indexes = range(row_size * idx_fig, (idx_fig + 1) * row_size)
    axe = fig.add_subplot(idx_fig_x, idx_fig_y, idx_fig + 1)
    axe.scatter(
        X[indexes, column_index_view],
        y[indexes],
        marker='^'
    )
    axe.scatter(
        X[indexes, column_index_view],
        lin_reg.predict(X[indexes]),
        marker='^'
    )

    axe.set_xticks([])
    axe.set_yticks([])

fig.canvas.set_window_title('Visualization Scatter prediction MLR')
plt.show()
