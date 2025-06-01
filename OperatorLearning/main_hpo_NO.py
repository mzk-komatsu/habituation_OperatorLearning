import numpy as np
from functions import *
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
# python main_hpo_NO.py train_settingsN.yaml
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

# L = (2,3,4,5) <-FNO, LNO
# dl = (32,64,128) <- FNO,LNO
# Modes = (10,15,20,25,30) <- FNO
# rank = (1,2,3,4,5) <- LNO

dim_layers = Integer(low=1, high=5, name="Layers")
dim_width =  Categorical([32, 64, 96, 128],name="Widths")
if model_name == "fno":
  dim_mr = Categorical([10, 15, 20, 25, 30], name="MR")
  default_parameters =[4, 64, 25]
elif model_name == "lno":
  dim_mr = Integer(low=1, high=5, name="MR")
  default_parameters =[4, 64, 2]

dimensions = [
    dim_layers,
    dim_width,
    dim_mr,
]


@use_named_args(dimensions=dimensions)
def fitness(Layers, Widths, MR):
    start_time = time.time()
    test = Parser(yaml_path)
    config = test.config
    config.Layers = Layers
    config.Widths = Widths
    config.MR = MR

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
    print("MR:", config.MR)
    print()

    # Load data sets
    train_loader, test_loader, x_train, y_train, x_test, y_test, params, Fs = load_data(sys_name, ntrain=ntrain, ntest=ntest, num_dp = num_dp, sub=2**sub_int, batch_size=bs, freq_mean = f_mean, freq_var = f_var, Flevel =F, numi=numi, numo=numo, recov=recov, display_data = True)

    # number of evaluation points in each data instance
    x,_ = list(train_loader)[0]
    s = x.shape[1]-numi

    out = train_model_hpo(exid, train_loader, test_loader, x_train, y_train, x_test, y_test, learning_rate = learning_rate, epochs = ep, batch_size = bs, step_size = step_size, gamma = gamma, L = config.Layers, modes = config.MR, width = config.Widths, sub_int = sub_int, model_name =model_name,  numi=numi, numo=numo, s = s)


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



#FNO: L, modes, width,
#LNO: L, rank, width


"""
https://github.com/pescap/HPOMax/blob/main/HPO-PINNs/Dirichlet/dirichlet_gp.py

FNO, LNO
- L = (2,3,4,5) <-FNO, LNO
- d0 = dL1 = (32, 64, 96, 128)
- dl = (32,64,128) <- FNO,LNO
- Modes = (10,15,20,25,30) <- FNO
- rank = (1,2,3,4) <- LNO
- dL2 = 2*dL1

DeepONet
- width = (32, 64, 96, 128)

"""
