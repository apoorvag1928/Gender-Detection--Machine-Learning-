import pandas as pd
import numpy as np

data=pd.read_csv("imdb4p5.csv")

label=data['gender']
features=data.drop(['age', 'gender', 'Unnamed: 0'], axis=1)

np.save("x.npy",features)
np.save("y.npy",label)
