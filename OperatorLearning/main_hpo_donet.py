import numpy as np
from functions import *
from functions_donet import donet
import matplotlib.pyplot as plt
import torch.nn as nn
import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from hpo_config import Parser
import time
import sys
import argparse

# ------------------------------------------------------
# Execution(N: id of train settings):
# python main_hpo_donet.py train_settingsN.yaml
# ------------------------------------------------------

args = sys.argv
print(args)
if len(args) == 1:
    yaml_path ="train_settings0.yaml"
else:
    yaml_path = args[1]
exid = yaml_path[-6]
with open(yaml_path, 'rb') as f:
  yml = yaml.safe_load(f)
model_name = yml["model_name"]

ITERATION = 0

# L = (2,3,4,5) 
# width = (32, 64, 96, 128)

dim_layers = Integer(low=1, high=5, name="Layers")
dim_width =  Categorical([32, 64, 96, 128],name="Widths")
default_parameters =[4, 64]

dimensions = [
    dim_layers,
    dim_width
]


@use_named_args(dimensions=dimensions)
def fitness(Layers, Widths):
    start_time = time.time()
    test = Parser(yaml_path)
    config = test.config
    config.Layers = Layers
    config.Widths = Widths

    global ITERATION

    # Load settings
    ep, sub_int, model_name, sys_name, _, n_modes, recov, learning_rate, step_size, gamma, L, width = get_learn_settings(yaml_path)
    json_path = os.path.join("../data_generation/", sys_name+".json")
    with open("../data_generation/path.json", 'r', encoding='utf-8') as f:
        JSON = json.load(f)
    global_path = "../data_generation/"+JSON["global_path"]
    F, _, num_dp, ntrain, ntest, bs, grids, f_mean, f_var, numi, numo, _, _, _ = get_settings(yaml_path, json_path, global_path)

    config.name = "gp-" +sys_name+"-"+model_name+"-"+str(ITERATION)
    print(config.name, "config.name")
    print(ITERATION, "it number")

    # Print the hyper-parameters.
    print("Layers:", config.Layers)
    print("Width:", config.Widths)
    print()

    # Load data sets (without Fs, Params, Habituation features)
    x_train, y_train, x_test, y_test= load_data_donet(sys_name, ntrain=ntrain, ntest=ntest, num_dp = num_dp, sub=2**sub_int, batch_size=bs, freq_mean = f_mean, freq_var = f_var, Flevel =F, numi=numi, numo=numo, recov=recov, display_data = True)
    # Train model
    out = train_model_donet(exid, x_train, y_train, x_test, y_test, learning_rate = learning_rate, epochs = ep, batch_size = bs, step_size = step_size, gamma = gamma, L = config.Layers, width = config.Widths, sub_int = sub_int, model_name = model_name, sys_name= sys_name)

    print(out)
    end_time = time.time()
    print(str((end_time-start_time)/60)+"min")
    return out

if __name__ == "__main__":
  n_calls = 11

  #FNO: L, modes, width,
  #LNO: L, rank, width

  test = Parser(yaml_path)
  config = test.config
  np.int = int

  search_result = gp_minimize(
      func=fitness,
      dimensions=dimensions,
      acq_func="EI",  # Expected Improvement.
      n_calls=n_calls,
      x0=default_parameters,
      random_state=123,
  )

  #name = "HPO/" + config.name
  parent = "HPO"
  if not os.path.exists(parent):
    os.makedirs(parent)
  name = parent + "/train"+str(exid)+"_"+model_name
  print(search_result)
  print(search_result.x)
  skopt.dump(search_result, name + ".pkl")
  np.save(name+".npy", search_result.x)

