import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import strpdate2num

convert = lambda s: float(int(strpdate2num('%Y-%m-%d')(s)) % 7)

dataset = np.genfromtxt(
	'data/data_train.csv',
	dtype=float,
	delimiter=',',
	converters={
		0: convert
	},
  usecols=(0, 23),
	encoding='utf-8'
)

print(dataset)

X = dataset[:, 0]
y = dataset[:, 1]

plt.scatter(X, y)
plt.show()
