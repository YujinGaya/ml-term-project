import numpy as np
import xgboost as xgb
from matplotlib.dates import strpdate2num

convert = lambda s: strpdate2num('%Y-%m-%d')(s) if s != '' else 0.0

dataset = np.genfromtxt(
	'data/data_train.csv',
	dtype=float,
	delimiter=',',
	usecols=(0, 1, 2, 3, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23),
	converters={
		0: convert,
		18: convert
	},
	encoding='utf-8'
)

print("genfromtxt: end")
print("shape: ", dataset.shape)
print(dataset[0])

np.random.shuffle(dataset)

# 🔥🔥🔥🔥🔥🔥🔥 조심하자!!
X = dataset[:, 0:17]
y = dataset[:, 17] / (500000 * np.ones(dataset.shape[0]))
print(X, y)
# X = np.hstack((
# 	X,
# 	np.remainder(
# 		X[:, 0],
# 		np.ones(X.shape[0]) * 365
# 	)[:, None],
# 	np.remainder(
# 		X[:, 0],
# 		np.ones(X.shape[0]) * 7
# 	)[:, None]
# ))

test_data_number = int(dataset.shape[0] * 0.1)
train_data_number = dataset.shape[0] - test_data_number
test_X, train_X = X[:test_data_number], X[test_data_number:]
test_y, train_y = y[:test_data_number], y[test_data_number:]

print("dataset ready. Starting XGBoost...")

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

features = [
	'Contract Date', # 0
	'Latitude', # 1
	'Longitude', # 2
	'Altitude', # 3
	'Floor', # 4
	'Angle', # 5
	'Area', # 6
	'No Parking Lot', # 7
	'Area Parking Lot', # 8
	'Management Fee', # 9
	'No Households', # 10
	'Age', # 11
	'Completion Date', # 12
	'Built Year', # 13
	'No Schools', # 14
	'No Bus Stations', # 15
	'No Subway Stations' # 16
]

# train = xgb.DMatrix(train_X, label=train_y, feature_names=features)
# test = xgb.DMatrix(test_X, label=test_y, feature_names=features)

train = xgb.DMatrix(train_X, label=train_y)
test = xgb.DMatrix(test_X, label=test_y)


evallist = [(test, 'eval'), (train, 'train')]

bst = xgb.train(param, train, 100, evals=evallist)
bst.save_model('0001.model')


train_y = train_y * (500000 * np.ones(train_data_number))
print('train y is...\n', train_y)
# train_y_hat = bst.predict(xgb.DMatrix(train_X, feature_names=features)) * (500000 * np.ones(train_data_number))
train_y_hat = bst.predict(xgb.DMatrix(train_X)) * (500000 * np.ones(train_data_number))
print('train_y_hat y is...\n', train_y_hat)

print('>> test performance:',
	1 - (
		np.linalg.norm(
			(train_y_hat - train_y) / train_y,
			ord=1
		) / train_data_number
	)
)


test_y = test_y * (500000 * np.ones(test_data_number))
print('test y is...\n', test_y)
# test_y_hat = bst.predict(xgb.DMatrix(test_X, feature_names=features)) * (500000 * np.ones(test_data_number))
test_y_hat = bst.predict(xgb.DMatrix(test_X)) * (500000 * np.ones(test_data_number))
print('test_y_hat y is...\n', test_y_hat)

print('>> test performance:',
	1 - (
		np.linalg.norm(
			(test_y_hat - test_y) / test_y,
			ord=1
		) / test_data_number
	)
)

import matplotlib.pyplot as plt

xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=0)
xgb.plot_tree(bst, num_trees=1)
xgb.plot_tree(bst, num_trees=2)

plt.show()

# model = xgb.XGBClassifier(silent=False, gamma=1)
# model.fit(train_X, train_y)
# train_y_hat = model.predict(train_X)
# print('>> train error:',
# 			np.linalg.norm(train_y_hat - train_y, ord=1) / train_data_number)

# print('train_y', train_y)

# print('>> train performance:',
# 	1 - (
# 		np.linalg.norm(
# 			(train_y_hat - train_y) / train_y,
# 			ord=1
# 		) / train_data_number
# 	)
# )

# # test model
# test_y_hat = model.predict(test_X)
# print('>> test error:',
# 			np.linalg.norm(test_y_hat - test_y, ord=1) / test_data_number)

# print('test_y', test_y)

# print('>> test performance:',
# 	1 - (
# 		np.linalg.norm(
# 			(test_y_hat - test_y) / test_y,
# 			ord=1
# 		) / test_data_number
# 	)
# )

# # plot feature importance
# from xgboost import plot_importance
# plot_importance(model)
# from matplotlib import pyplot
# pyplot.show()




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
