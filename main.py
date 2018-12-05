# Import necessary packages
import argparse
import scipy
import os
import math
import numpy as np
import argparse
import json

import keras
from keras.callbacks import LambdaCallback
from keras.models import load_model
from utils.parse_args import parse_args, get_logs_dir
from utils.load_data import load_data
from utils.assemble_model import assemble_model

def log_callback(epoch, logs, logs_dir):
    dic = {}
    with open(logs_dir+'/loss_log.json', 'r') as json_log:
        dic = json.loads(json_log.read())
        dic[epoch] = {'loss': logs['loss'], 'accuracy': logs['acc']}
    with open(logs_dir+'/loss_log.json', 'w') as json_log_w:
        json.dump(dic,json_log_w)

def fit_model(model, X,Y, args):
    logs_dir = get_logs_dir(args)+"/train"
    with open(logs_dir+'/loss_log.json', 'w') as outfile:
        json.dump({}, outfile)
    json_logging_callback = LambdaCallback(
        on_epoch_end= lambda epoch, logs: log_callback(epoch, logs,logs_dir)
    )
    model.fit(X,Y, batch_size = args.batch_size, epochs=args.epochs, 
        callbacks=[keras.callbacks.TensorBoard(log_dir=logs_dir),
        json_logging_callback])

    model.save(logs_dir+'/model.h5')
    return model

def evaluate_model(model, X, Y, args):
    loss_train, acc_train = model.evaluate(X, Y, verbose=0)
    loss_eval, acc_eval, loss_test, acc_test = None, None, None, None
    logs_dir = get_logs_dir(args)
    records = {"train":{"loss":loss_train, "accuracy":acc_train}}
    if args.eval:
        X,Y = load_data(args, data_partition="val")
        loss_eval, acc_eval = model.evaluate(X, Y, verbose=0)
        records["eval"] = {"loss":loss_eval, "accuracy":acc_eval}
    if args.test:
        X,Y = load_data(args, data_partition="test")
        loss_test, acc_test = model.evaluate(X, Y, verbose=0)
        records["test"] = {"loss":loss_test, "accuracy":acc_test}
    
    with open(logs_dir+'/log.txt', mode='w') as json_log:
        json.dump(records, json_log)

def main(args): 
    X,Y = load_data(args)
    print("Data Loaded") 
    
    model = None
    if args.train:
        model = assemble_model(args)
        model.compile("adam", "categorical_crossentropy", ["accuracy"])
        model = fit_model(model, X, Y, args)
        print("Model was fitted!")
    else:
        logs_dir = get_logs_dir(args)+"/train"
        model = load_model(logs_dir+'/model.h5')

    evaluate_model(model, X, Y, args)

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    main(args)




