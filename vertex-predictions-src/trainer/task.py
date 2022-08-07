#!/usr/bin/env python3
#@author: Jorge III Altamirano-Astorga
import tensorflow as tf
import re, os, sys, shelve, time, dill, io, logging
import argparse #args from cli
from pickle import PicklingError
from dill import Pickler, Unpickler
shelve.Pickler = Pickler
shelve.Unpickler = Unpickler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
tf.get_logger().setLevel('ERROR')

def evaluate_model(
    input_dataset,
    #output_datastore,
    model_file,
    mse=True,
    mae=True,
    debug=False):
    
    if debug:
        logging.info("Entering evaluate_model()")
    
    if not model_file.startswith('/'):
        logging.error('Model file should be on local storage. No URL gs:// supported!')
        return 255
    if mse == False and mae == False:
        logging.error('Stopping as no models need to be evaluated!')
        return 255
    
    base_dir = re.sub('^(.*/)[^.]+\.h5$', r'\1', model_file)
    output_datastore = base_dir
    modelname = re.sub('^.*/([^.]+)\.h5$', r'\1', model_file)
    logging.info("Processing base input dir: %s"%base_dir)
    logging.info("Processing base name: %s"%modelname)
    
    tsparams = get_tsparams(base_dir, modelname, debug)
    
    test3_iaq = get_timeseries_dataset_from_array(input_dataset, tsparams, debug)
    
    logging.info('Loading H5 model...')
    model = tf.keras.models.load_model(os.path.join(base_dir, f"{modelname}.h5"))
    logging.info('Done loading H5 model!')
    logging.info('Predicting... (this may take a while)')
    Y_hat = model.predict(test3_iaq, batch_size=tsparams["batch_size"])
    logging.info('Y_hat.shape = %s'%str(Y_hat.shape))
    logging.info('Done predictions!')
    
    logging.info('Creating Ground Truth Y...')
    Y_gt = np.array([])
    for batch in test3_iaq:
      _, targets = batch
      Y_gt = np.append(Y_gt,targets.numpy())
    logging.info('Y_gt.shape = %s'%str(Y_gt.shape))
    logging.info('Done creating Ground Truth Y!')
    
    returnValue = 0 
    if mse:
        logging.info('Getting MSE...')
        metric_filename = os.path.join(
            output_datastore,
            f'{modelname}.mse.pickle.gz'
        )
        mse = get_mse(Y_gt, Y_hat, metric_filename, debug)
        logging.info('Done MSE!')
    if mae:
        logging.info('Getting MAE...')
        metric_filename = os.path.join(
            output_datastore,
            f'{modelname}.mae.pickle.gz'
        )
        mae = get_mae(Y_gt, Y_hat, metric_filename, debug)
        logging.info('Done MAE!')
        
    #### Start Section: Hyperparameter Tuning
    try: 
        
        import hypertune

        hpt = hypertune.HyperTune()

        if mse:
            hpt.report_hyperparameter_tuning_metric(
                  hyperparameter_metric_tag='mse',
                  metric_value=mse,
                  global_step=1
                  )
        if mae:
            hpt.report_hyperparameter_tuning_metric(
                  hyperparameter_metric_tag='mae',
                  metric_value=mae,
                  global_step=1
                  )
    except Exception as e:
      logging.error("Start error of hypertune")
      logging.error(e)
      logging.error("End error of hypertune")
      pass
    #### End Section: Hyperparameter Tuning
    
    return returnValue

def get_mse(Y_gt, Y_hat, metric_filename, debug=False):
    if debug:
        logging.info("Entering get_mse()")
    A = []
    m = tf.keras.metrics.MeanSquaredError()

    for i in range(0, Y_hat.shape[1]):
      m.update_state(Y_gt, Y_hat[:,i])
      A.append(m.result().numpy())
      m.reset_states()
    
    logging.info("Saving MSE metrics to file: %s"%metric_filename)
    with io.open(metric_filename, 'wb') as metric_file:
        dill.dump(A, metric_file)
    logging.info("Saved %s"%metric_filename)
    A = np.array(A).mean()
    
    logging.info("Returning MSE mean: %f"%A)
    return A

def get_mae(Y_gt, Y_hat, metric_filename, debug=False):
    if debug:
        logging.info("Entering get_mae()")
    A = []
    m = tf.keras.metrics.MeanAbsoluteError()

    for i in range(0, Y_hat.shape[1]):
      m.update_state(Y_gt, Y_hat[:,i])
      A.append(m.result().numpy())
      m.reset_states()
    
    logging.info("Saving MAE metrics to file: %s"%metric_filename)
    with io.open(metric_filename, 'wb') as metric_file:
        dill.dump(A, metric_file)
    logging.info("Saved %s"%metric_filename)
    A = np.array(A).mean()
    
    logging.info("Returning MAE mean: %f"%A)
    return A

def get_timeseries_dataset_from_array(input_dataset, tsparams,
                                      debug=True):
    if debug:
        logging.info("Entering get_timeseries_dataset_from_array()")
        
    logging.info('Loading dataset: %s'%input_dataset)
    data =  pd.read_pickle(input_dataset)
    data = data[~data.isna().any(axis=1)]
    excluded_columns = ["iaqAccuracy", "wind_speed", "wind_deg"]
    if debug:
        logging.info('Splitting input dataset into Train and Test datasets.')
    train, test = train_test_split(data[[x
                                            for x in data.columns
                                            if x not in excluded_columns]],
                                   train_size=0.7, random_state=175904, shuffle=False)
    scaler = None
    scaler = MinMaxScaler()
    if debug:
        logging.info('MinMaxScale-ing data.')
    scaler_f = scaler.fit(train)
    test2 = scaler_f.transform(test)
    #with io.open(os.path.join("data/output-hyper05min-w07-stride2-samplingrate2/", "scaler.dill"), 'rb') as scalerfile:
    #  scaler = dill.load(scalerfile)
    X_cols = [i for i, x in enumerate(train.columns)
              if x not in ["IAQ", "gasResistance"]]
    Y_cols = [i for i, x in enumerate(train.columns)
              if x in ["IAQ", "gasResistance"]]
    X_test  = test2[:, X_cols]
    Y_test = test2[:, Y_cols]

    if debug:
        logging.info('Executing timeseries_dataset_from array...')
    test3_iaq = tf.keras.preprocessing.timeseries_dataset_from_array(
      X_test,
      Y_test[:, 1],
      sequence_length=tsparams["sequence_length"],
      sampling_rate=tsparams["sampling_rate"],
      sequence_stride=tsparams["stride"],
      batch_size=tsparams["batch_size"],
      seed=175904
    )
    if debug:
        logging.info('Done timeseries_dataset_from array...')
    
    return test3_iaq

def get_tsparams(base_dir, modelname, debug=False):
    if debug:
        logging.info("Entering get_tsparams()")
    tsparamsfilename = os.path.join(base_dir, f"{modelname}.tsparams.dill")
    with io.open(tsparamsfilename, 'rb') as tsparamsfile:
        tsparams = dill.load(tsparamsfile)
    logging.info("Time series parameters:")
    logging.info("{ %s }"%tsparams)
    return tsparams

def setup_logger(logname="logger_jorge3a", loglevel=logging.INFO):
    # Define the log format
    log_format = '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s'

    # Define basic configuration
    logging.basicConfig(
        # Define logging level
        level=loglevel,
        # Declare the object we created to format the log messages
        format=log_format,
        # Declare handlers
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Define your own logger name
    logger = logging.getLogger(logname)
    

def usage(argv):
    parser = argparse.ArgumentParser(
        description="""
        Creates a files saved into a Datastore with the 
        Mean Squared Errors and Mean Absolute Error using
        an H5 Model File evaluated with the input dataset.""",
        epilog="Example: %(prog)s -m trained/model.h5 https://data.example.com/data/data.pickle.gz", 
        prefix_chars='-')
            
    parser.add_argument('--nomse', '-S', action="store_true",  default=False,
                        help='Avoid calculating Mean Squared Error (MSE).')
    parser.add_argument('--nomae', '-A', action="store_true",  default=False,
                        help='Avoid calculating Mean Squared Error (MSE).')
    parser.add_argument('--debug', '-d', action="store_true",  default=False,
                        help='Enable debugging.')
    parser.add_argument('--model', '-m', nargs=1, required=True,
                        help="""H5 **trained** model file.""")
    parser.add_argument('input_dataset', nargs=1, default="",
                        help="""Input dataset URL or Location of the File.""")
    #parser.add_argument('output_datastore', nargs=1, default="", 
    #                    help="""Output dataset URL or Location of the File.
    #                    """)
    return parser.parse_args()

def main(argv):
    
    ### read cli arguments
    args = usage(argv)
    
    if args.debug: 
        setup_logger(argv[0], logging.DEBUG)
    else:
        setup_logger(argv[0], logging.INFO)
        
    logging.info("MSE: %s"%(not args.nomse))
    logging.info("MAE: %s"%(not args.nomae))
    logging.info("Debug: %s"%args.debug)
    logging.info("Model Filename: %s"%args.model[0].strip())
    logging.info("Input Dataset: %s"%args.input_dataset[0].strip())
    logging.info("Output Datastore: %s"%re.sub('^(.*/)[^.]+\.h5$', r'\1', args.model[0].strip()))
    logging.info("")
    #print(args)
    
    result = evaluate_model(
        input_dataset=args.input_dataset[0].strip(),
        #output_datastore=args.output_datastore[0].strip(),
        model_file=args.model[0].strip(),
        mse=(not args.nomse),
        mae=(not args.nomae),
        debug=args.debug
    )
    
    exit(result)
    
    
if __name__ == "__main__":
    main(sys.argv)