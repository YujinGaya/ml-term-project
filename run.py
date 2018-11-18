import numpy as np
from xgboost import XGBClassifier
from matplotlib.dates import strpdate2num

convert = lambda s: strpdate2num('%Y-%m-%d')(s) if s != '' else 0.0

dataset = np.genfromtxt(
	'data/data_train.csv',
	dtype=float,
	delimiter=',',
	converters={
		0: convert,
		18: convert
	},
	encoding='utf-8'
)

print("genfromtxt: end")

np.random.shuffle(dataset)

X = dataset[:, 0:22]
y = dataset[:, 22]

test_data_number = int(dataset.shape[0] * 0.1)
train_data_number = dataset.shape[0] - test_data_number
test_X, train_X = X[:test_data_number], X[test_data_number:]
test_y, train_y = y[:test_data_number], y[test_data_number:]

print("dataset ready. Starting XGBoost...")

model = XGBClassifier()
model.fit(train_X, train_y)
train_y_hat = model.predict(train_X)
print('>> train error:',
			np.linalg.norm(train_y_hat - train_y, ord=1) / train_data_number)

# test model
test_y_hat = model.predict(test_X)
print('>> test error:',
			np.linalg.norm(test_y_hat - test_y, ord=1) / test_data_number)

# plot feature importance
from xgboost import plot_importance
plot_importance(model)
from matplotlib import pyplot
pyplot.show()




# count = 0
# def debug(s):
# 	global count
# 	count = count + 1
# 	print(count)
# 	return s

# zeroIfNullF = lambda s: s if s != '' else 0.0
# zeroIfNullI = lambda s: s if s != '' else 0

# names=['Contract Date', 'Latitude', 'Longitude', 'Altitude', '1st class Region Id', '2nd class Region Id', 'Road Id', 'Apartment Id', 'Floor', 'Angle', 'Area', 'Parking Lot No', 'Parking Lot Area', 'External Vehicle', 'Fee', '# household', 'Age', 'Builder Id', 'Construction Date', 'Built Year', 'Schools', 'Bus Stations', 'Subway Stations', 'Price'],

# dataset = np.loadtxt(
# 	'data/data_train.csv',
# 	dtype=None,
# 	# {
# 	# 	'names': (	'Contract Date',	'Latitude', 'Longitude', 	'Altitude', '1st class Region Id', 	'2nd class Region Id',	'Road Id', 'Apartment Id',	'Floor', 	'Angle',	'Area', 		'Parking Lot No', 'Parking Lot Area', 'External Vehicle', 'Fee',		'# household', 	'Age',			'Builder Id', 'Construction Date',	'Built Year',	'# Schools',	'# Bus Stations', '# Subway Stations'),
# 	# 	'formats': (np.float32,				np.float32,	np.float32,		np.float32,	np.int32,								np.int32,								np.int32,		np.int32,				np.int32,	np.int32,	np.float32,	np.int32,					np.int32,						np.int32,						np.int32,	np.int32,				np.float32,	np.int32,			np.float32,								np.int32,				np.int32,				np.int32,	np.int32)
# 	# }
# 	delimiter=',',
# 	converters={
# 		0: convert,    		# 1980-03-16,
# 		1: debug,    		# 40.8201,
# 		2: zeroIfNullF,    		# -73.9495,
# 		3: zeroIfNullF,    		# 46.1,
# 		4: zeroIfNullI,    		# 3,
# 		5: zeroIfNullI,    		# 28,
# 		6: zeroIfNullI,    		# 139,
# 		7: zeroIfNullI,    		# 216,
# 		8: zeroIfNullI,    		# 6,
# 		9: zeroIfNullI,    		# 149,
# 		10: zeroIfNullF,    		# 94.696,
# 		11: zeroIfNullI,    		# 6661,
# 		12: zeroIfNullI,    		# 98361,
# 		13: zeroIfNullI,    		# 1,
# 		14: zeroIfNullI,    		# 420,
# 		15: zeroIfNullI,    		# 1321,
# 		16: zeroIfNullF,    		# 45.4,
# 		17: zeroIfNullI,    		# 14,
# 		18: convert,    		# 1964-01-01,
# 		19: zeroIfNullI,    		# 1964,
# 		20: zeroIfNullI,    		# 3,
# 		21: zeroIfNullI,    		# 5,
# 		22: zeroIfNullI,    		# 0,
# 		23: zeroIfNullI,    		# 14040
# 	}
# )
