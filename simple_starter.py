import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss
import numpy as np
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, Embedding
from keras.utils import np_utils

random.seed(2016)

def run_xgb(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 3
    subsample = 0.7
    colsample_bytree = 0.7
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "learning_rate" : 0.5,
        "n_estimators" : 1000,
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 500
    early_stopping_rounds = 50
    test_size = 0.3

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(train[features])
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])


    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    score = log_loss(y_valid.tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score

from sklearn import neighbors


def run_knn(train, test, features, target, n_neighbors=3, weights='uniform'):

    train_X = train.as_matrix(columns=features)
    test_X = test.as_matrix(columns=features)

    train_y = train[target].values

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(train_X, train_y)

    predictions = clf.predict_proba(test_X)
    print(predictions)
    return predictions


def run_nn(train, test, features, target, batch_size=25):

    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    nb_classes = np.max(train[target]) + 1
    print(nb_classes, 'classes')
    Y_train = np_utils.to_categorical(train[target].values, nb_classes)

    train = train.as_matrix(columns=features)
    test = test.as_matrix(columns=features)

    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print(train[:10])
    train = train.reshape(train.shape + (1,))
    test = test.reshape(test.shape + (1,))
    print('Building model...', train.shape)
    print(train.shape[1])
    model = Sequential()
    model.add(Convolution1D(
        nb_filter=128,
        filter_length=3,
        input_shape=(len(features), 1),
        input_length=train.shape[0],
        activation='relu',
        init='uniform'))
    model.add(Activation('relu'))
    model.add(Convolution1D(128, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.25))

    #model.add(Convolution1D(128, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(Convolution1D(128, 3))
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_length=2))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])

    history = model.fit(train, Y_train,
                        nb_epoch=150, batch_size=batch_size,
                        verbose=1, validation_split=0.4)
    print("Predicting... \n")

    predictions = model.predict_proba(test)

    return predictions


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

def read_train_test():
    # Events
    print('Read events...')
    events = pd.read_csv("data/events.csv", dtype={'device_id': np.str})
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
    #events['timestamp'] = split_datetime(events['timestamp'])
    events['device_id_prob'] = events.groupby('device_id')['event_id'].transform('mean')
    #train['device_model_freq'] = train.groupby('device_model')['event_id'].transform('count')
    #train['device_model_prob'] = train.groupby('device_model')['event_id'].transform('mean')
    #train['phone_brand_freq'] = train.groupby('phone_brand')['event_id'].transform('count')
    #train['phone_brand_prob'] = train.groupby('phone_brand')['event_id'].transform('mean')

    events['year'] = events['timestamp'].str[:4].astype(int)
    events['month'] = events['timestamp'].str[5:7].astype(int)
    events['day'] = events['timestamp'].str[8:10].astype(int)
    events['hour'] = events['timestamp'].str[11:13].astype(int)
    events['minute'] = events['timestamp'].str[14:16].astype(int)
    events['second'] = events['timestamp'].str[17:].astype(int)

    print(np.unique(events['year']))
    print(np.unique(events['day']))

    events_small = events[['device_id', 'counts', 'year',
                           'month', 'day', 'hour', 'minute', 'device_id_prob']].drop_duplicates('device_id', keep='first')

    print(events_small.head())
    # import sys
    #
    # def show_sizeof(x, level=0):
    #
    #     print("\t" * level, x.__class__, sys.getsizeof(x), x)
    #     if hasattr(x, '__iter__'):
    #         if hasattr(x, 'items'):
    #             for xx in x.items():
    #                 show_sizeof(xx, level + 1)
    #         else:
    #             for xx in x:
    #                 show_sizeof(xx, level + 1)
    #
    # show_sizeof(3.14159265358979323846264338327950288)
    # Read apps
    app_labels = pd.read_csv("data/app_labels.csv", dtype={'app_id': np.str, 'label_id': np.str})
    app_events = pd.read_csv("data/app_events.csv", dtype={'event_id': np.str, 'app_id': np.str})
    #
    # print(app_events.head())
    # print(app_labels.head())
    #apps = pd.merge(app_events, app_labels, how='left', on='app_id', left_index=True)

    # print(apps.head())
    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("data/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)

    print(pbd.head())
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    print(pbd.head())
    # Train
    print('Read train...')
    train = pd.read_csv("data/gender_age_train.csv", dtype={'device_id': np.str})

    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    train = train.drop(['gender'], axis=1)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)

    train['device_model_freq'] = train.groupby('device_model')['day'].transform('count')
    train['device_model_prob'] = train.groupby('device_model')['day'].transform('mean')

    train.fillna(-1, inplace=True)


    print(train.head())

    # Test
    print('Read test...')
    test = pd.read_csv("data/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)

    test['device_model_freq'] = test.groupby('device_model')['day'].transform('count')
    test['device_model_prob'] = test.groupby('device_model')['day'].transform('mean')

    test.fillna(-1, inplace=True)

    # Features
    features = list(test.columns.values)
    features.remove('device_id')

    return train, test, features


train, test, features = read_train_test()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_predictions, score = run_xgb(train, test, features, 'group', 25)
#test_predictions = run_nn(train, test, features, 'group', 25)
#test_predictions = run_knn(train, test, features, 'group', n_neighbors=3, weights='distance')
#print("LS: {}".format(round(score, 5)))
create_submission(0.0, test, test_predictions)
