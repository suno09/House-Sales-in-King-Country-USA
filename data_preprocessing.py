import pandas as pd
from matplotlib import pyplot as plt


# #### data preprocessing ####
# read csv file
dataset = pd.read_csv("kc_house_data.csv")
# plot bar (sqft_living, sqft_above, sqft_basement)
dataset[100:120].sort_values("price").plot(
    x="price",
    y=['sqft_living', 'sqft_above', 'sqft_basement'],
    kind="bar"
)
dataset[100:120].sort_values("price").plot(
    x="price",
    y=['bedrooms', 'bathrooms', 'floors'],
    kind="bar"
)
dataset[100:120].sort_values("price").plot(
    x="price",
    y=['view', 'condition', 'grade'],
    kind="bar"
)

plt.show()
