import numpy as np   # Linear Algebra
import pandas as pd  # Data load and handle
import xgboost as xgb
# from matplotlib.dates import strpdate2num
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# Name of columns
col_names = ['contract_date', 'latitude', 'longitude', 'altitude', '1st_region_id', '2nd_region_id', 'road_id', 'apart_id', 'floor', 'angle', 'area', 'num_parking_lot', 'total_parking_lot', 'external_vehicle', 'manage_fee', 'num_household', 'resident_age', 'builder_id', 'completion_date', 'built_year', 'schools', 'bus_stations', 'subway_stations']

##### performance: (list-of-float, list-of-float) -> float #####
##### To calculate performance training:validation = 5:1 ##########
##### 1-abs((prediction-answer)/prediction)/40099 ##############
def performance(y_validation, y_test):
    y_validation_array = np.array(y_validation)
    output_error = np.abs(y_validation - y_test)
    output_error = output_error / y_validation
    output_error_sum = np.sum(output_error)
    n_validation_data = y_validation.size
    accuracy = 1 - (output_error_sum / n_validation_data)
    print("Performance is... %f" % accuracy)
    return accuracy


##### data_setting: (empty) -> three matrix and three list-of-float #####
##### To load training data, give column names, and split data into training data and validation data
##### And handle missing data with SimpleImputer
def data_setting():
    data_train = pd.read_csv('./data/data_train.csv', names = col_names+['price'])
    data_train = shuffle(data_train)
    data_train['contract_date'] = pd.to_datetime(data_train['contract_date'])
    data_train['completion_date'] = pd.to_datetime(data_train['completion_date'])
    data_train['contract_year'] = data_train['contract_date'].dt.year
    data_train['contract_month'] = data_train['contract_date'].dt.month
    data_train['contract_day'] = data_train['contract_date'].dt.day
    data_train['completion_year'] = data_train['completion_date'].dt.year
    data_train['completion_month'] = data_train['completion_date'].dt.month
    data_train['completion_day'] = data_train['completion_date'].dt.day
    data_train = data_train.drop(columns=['contract_date', 'completion_date'])
    
    #### data_train.columns ####
    # 0: latiitude,
    # 1: longitude,
    # 2: altitude,
    # 3: 1st_region_id,
    # 4: 2nd_region_id,
    # 5: road_id,
    # 6: apart_id,
    # 7: floor,
    # 8: angle,
    # 9: area,
    # 10: num_parking_lot,
    # 11: total_parking_lot
    # 12: external_vehicle,
    # 13: manage_fee,
    # 14: num_household,
    # 15: resident_age,
    # 16: builder_id,
    # 17: built_year,
    # 18: schools,
    # 19: bus_stations,
    # 20: subway_stations,
    # 21: price,
    # 22: contract_year,
    # 23: contract_month,
    # 24: contract_day,
    # 25: completion_year
    ############################


    ################# Split data into training data and validation data ################
    X_train = data_train.drop(columns=['price'])
    y_train = data_train['price']
    y_train_divided = y_train / 500000

    global test_data_number, train_data_number
    test_data_number = int(data_train.shape[0] * 0.1)
    train_data_number = data_train.shape[0] - test_data_number

    X_training = X_train[test_data_number:][:]
    y_training = y_train_divided[test_data_number:][:]
    X_validation = X_train[:test_data_number][:]
    y_validation = y_train_divided[:test_data_number][:]
    

    ############### Set test dataset (data_test.csv) #############
    data_test = pd.read_csv('./data/data_test.csv', names = col_names)
    data_test['contract_date'] = pd.to_datetime(data_test['contract_date'])
    data_test['completion_date'] = pd.to_datetime(data_test['completion_date'])
    data_test['contract_year'] = data_test['contract_date'].dt.year
    data_test['contract_month'] = data_test['contract_date'].dt.month
    data_test['contract_day'] = data_test['contract_date'].dt.day
    data_test['completion_year'] = data_test['completion_date'].dt.year
    data_test['completion_month'] = data_test['completion_date'].dt.month
    data_test['completion_day'] = data_test['completion_date'].dt.day
    X_test = data_test.drop(columns=['contract_date', 'completion_date'])
    ###########################################################

    print("dataset ready. Starting XGBoost...")

    return X_train, y_train, X_training, y_training, X_validation, y_validation, X_test


def train_n_predict(train_X, train_y, test_X, test_y, X_test):
    train = xgb.DMatrix(train_X, label=train_y)
    test = xgb.DMatrix(test_X, label=test_y)

    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'

    evallist = [(test, 'eval'), (train, 'train')]

    bst = xgb.train(param, train, 500, evals=evallist)
    bst.save_model('base.model')
    bst.dump_model('base.dump.raw.txt')

    train_y_hat = bst.predict(xgb.DMatrix(train_X))
    validate_y_hat = bst.predict(xgb.DMatrix(test_X))
    test_y_hat = bst.predict(xgb.DMatrix(X_test))

    return train_y_hat, validate_y_hat, test_y_hat


def main():
    print("Start data setting...")
    # data_setting()
    X_train, y_train, X_training, y_training, X_validation, y_validation, X_test = data_setting()

    train_y_hat, validate_y_hat, test_y_hat = train_n_predict(X_training, y_training, X_validation, y_validation, X_test)
    
    y_training = y_training * 500000
    y_validation = y_validation * 500000
    train_y_hat = train_y_hat * 500000
    validate_y_hat = validate_y_hat * 500000
    test_y_hat = test_y_hat * 500000

    print("Performance calculation for training data")
    performance(y_training, train_y_hat)
    print("Performance calculation for validation data")
    performance(y_validation, validate_y_hat)

    f = open("aaa.csv", 'w')
    for i in range(9647):
      f.write(str(test_y_hat[i]))
      f.write("\n")
    f.close()

if __name__ == "__main__":
    main()
