# Created by Yuexiong Ding
# Date: 2018/9/30
# Description: 
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers import Masking
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from MyModule import data
from MyModule import evaluate
import os
import tensorflow as tf
from keras import backend as K
import gc


def load_model(model_path, weight_path):
    if os.path.exists(model_path):
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        # 有参数则加载
        if os.path.exists(weight_path):
            print('load weights ' + weight_path)
            model.load_weights(weight_path)
        return model
    else:
        exit('找不到模型' + model_path)


def main(conf, is_train=True):
    df_raw = data.get_raw_data(conf['data_conf']['data_path'], usecols=conf['data_conf']['usecols'])
    X_train, X_test, y_train, y_test, y_scaler = data.process_data_for_lstm(df_raw, conf['model_conf'])

    if is_train:
        if os.path.exists(model_path):
            json_string = open(model_path).read()
            model = model_from_json(json_string)
            # 有参数则加载
            if os.path.exists(weight_path):
                print('load weights ' + weight_path)
                model.load_weights(weight_path)
        else:
            model = Sequential()
            model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dense(units=64, activation='linear'))
            model.add(Dense(units=1))
            open(model_path, 'w').write(model.to_json())
        model.compile(loss='mse', optimizer='RMSprop')
        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, epochs=conf['model_conf']['epochs'],
                            batch_size=conf['model_conf']['batch_size'], validation_data=(X_test, y_test), verbose=1,
                            callbacks=callbacks_list, shuffle=False)

        evaluate.draw_loss_curve(figure_num=conf['model_conf']['target_col'], train_loss=history.history['loss'],
                                 val_loss=history.history['val_loss'])
    else:
        json_string = open(model_path).read()
        model = model_from_json(json_string)
        model.load_weights(weight_path)
        y_pred = model.predict(X_test)
        y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        df_all_metrics = evaluate.all_metrics(y_true[: len(y_true) - 1], y_pred[1:])
        evaluate.draw_fitting_curve(y_true[: len(y_true) - 1], y_pred[1:])


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    data_path = '../DataSet/Processed/Train/train_v1.csv'
    cols = ['Site_060371103', 'Site_060370016', 'Site_060371201', 'Site_060374004', 'Site_060376012']
    # df_data = pd.read_csv('../DataSet/Processed/Train/train_v1.csv', usecols=cols)
    model_path = '../Models/LSTM/model_epochs1000_batch512.best.json'
    weight_path = '../Models/LSTM/weights_epochs1000_batch512.best.hdf5'
    max_time_lag = 48
    min_corr_coeff = 0.3
    target_offset = 1
    test_split = 0.4
    epochs = 100
    batch_size = 1024
    config = {
        'data_conf': {
            'data_path': data_path,
            'usecols': cols
        },
        'model_conf': {
            'time_steps': {},
            'target_col': 'Site_060371103',
            'other_cols': cols,
            'max_time_lag': max_time_lag,
            'test_split': test_split,
            'min_corr_coeff': min_corr_coeff,
            'target_offset': target_offset,
            'model_path': model_path,
            'weight_path': weight_path,
            'epochs': epochs,
            'batch_size': batch_size,
        }
    }
    # data = process_data_for_lstm(df_data, model_config)
    # main(config, is_train=True)
    main(config, is_train=False)
