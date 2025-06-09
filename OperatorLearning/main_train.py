import numpy as np
from functions import *
import matplotlib.pyplot as plt
import json
import sys
from distutils.util import strtobool

# ------------------------------------------------------
# Execution(N: id of train settings):
#
# python main_train.py train_settingsN.yaml 
#               or 
# python main_train.py train_settingsN.yaml [True/False]
#
#   True -> Use the model with optimized hyperparameters. (True can be applied only after hyper parameter optimization)
#   False or nothing -> Use the model with hyperparameters defined in train_settingsN.yaml.
#
# [train_settingsN.yaml]
#   - ep: # of epochs (e.g., 1000)
#   - model_name: name of deep learning model (e.g., "fno")
#   - sys_name: name of habituation system (e.g., "TysonAL2D")
#   - sub_int: an integer that determines the coarseness of downsampling (e.g., 5 ---> # of data points in [0,1] is num_dp/2^5)
#   - modes: # of modes in Neural Operator (e.g., 3)
#   - recov: whether input data set contain repetitive stimuli with recovery stimulus (0: no recovery stimuli, 1: all input functions contain recovery stimuli, 2: some of input functions contain recovery stimuli)
#   - learning_rate : Initial learning rate (e.g., 0.001)
#   - step_size: step size (Decays the learning rate of each parameter group every step_size epochs.) (e.g, 50)
#   - gamma: gamma (Decays the learning rate of each parameter group by gamma every step_size epochs.) (e.g., 0.5)
#   - L: # of layers in Neural Operator (e.g., 4)
#   - width : # width of each layer in Neural Operator (e.g., 64)
# ------------------------------------------------------


# Specify json and yaml files
with open("../data_generation/path.json", 'r', encoding='utf-8') as f:
    JSON = json.load(f)
global_path = "../data_generation/"+JSON["global_path"]

args = sys.argv
print(args)
if len(args) == 1:
    yaml_path ="train_settings0.yaml"
else:
    yaml_path = args[1]
    if len(args) > 2:
        hpo = bool(strtobool(args[2]))
    else:
        hpo = False
exid = yaml_path[-6]
# Load settings
ep, sub_int, model_name, sys_name, _, n_modes, recov, learning_rate, step_size, gamma, L, width = get_learn_settings(yaml_path)
json_path = os.path.join("../data_generation/", sys_name+".json")
F, _, num_dp, ntrain, ntest, bs, grids, f_mean, f_var, numi, numo, _, _, _ = get_settings(yaml_path, json_path, global_path)

if model_name == "donet":
    from functions_donet import * 

# Training
if model_name == "donet":
    # Load data sets (without Fs, Params, Habituation features)
    x_train, y_train, x_test, y_test= load_data_donet(sys_name, ntrain=ntrain, ntest=ntest, num_dp = num_dp, sub=2**sub_int, batch_size=bs, freq_mean = f_mean, freq_var = f_var, Flevel =F, numi=numi, numo=numo, recov=recov, display_data = True)
    if hpo == True: # With HPO
        d = np.load("HPO/train"+str(exid)+"_"+model_name+".npy")
        L = d[0]
        width = d[1]
    # Train model
    train_model_donet(exid, x_train, y_train, x_test, y_test, learning_rate = learning_rate, epochs = ep, batch_size = bs, step_size = step_size, gamma = gamma, L = L, width = width, sub_int = sub_int, model_name = model_name, sys_name= sys_name)
else: 
    # Load data sets
    train_loader, test_loader, x_train, y_train, x_test, y_test, params, Fs = load_data(sys_name, ntrain=ntrain, ntest=ntest, num_dp = num_dp, sub=2**sub_int, batch_size=bs, freq_mean = f_mean, freq_var = f_var, Flevel =F, numi=numi, numo=numo, recov=recov, display_data = True)
    # number of evaluation points in each data instance
    x,_ = list(train_loader)[0]
    s = x.shape[1]-numi   
    train_model(exid, train_loader, test_loader, x_train, y_train, x_test, y_test, learning_rate = learning_rate, epochs = ep, batch_size = bs, step_size = step_size, gamma = gamma, L = L, modes = n_modes, width = width, sub_int = sub_int, model_name=model_name, sys_name= sys_name, numi=numi, numo=numo, s =s, hpo = hpo)
