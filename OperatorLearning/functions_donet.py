import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LRNO, FNO1d, LRNOL, FNO1dL, LowRank1d, SpectralConv1d
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import os, sys
import matplotlib.pyplot as plt
from torchsummary import summary
import seaborn as sns
from sklearn.metrics import r2_score
import shutil
import yaml, json
import pickle
import random
from scipy.integrate import trapz
from sklearn.metrics import mean_absolute_percentage_error
import math
import skopt
import deepxde as dde

sys.path.append("../fno")
from utilities3 import *
from Adam import Adam

sys.path.append("../analysis")
import tools

# Get ht_thres (thresholds used to evaluate habituation time)
with open("../data_generation/path.json", 'r', encoding='utf-8') as f:
    JSON = json.load(f)
global_path = JSON["global_path"]
with open("../data_generation/"+global_path, 'r', encoding='utf-8') as f:
    gJSON = json.load(f)
ht_thres = gJSON["ht_thres"]


def load_data_donet(sys_name, ntrain=1000, ntest=1000, sub=2**3, h=1024, num_dp=1024, batch_size=20, freq_mean = 10, freq_var = 1, Flevel = 30, T=1,rand=False, numi=2, numo=4, recov=0, display_data = False):
  json_file = "../data_generation/"+sys_name+".json"
  with open(json_file, 'r', encoding='utf-8') as f:
    jsondata = json.load(f)
  data_dir = "../data_generation/data_"+jsondata["config_name"]+"/"
  if recov == 0:
      fullpath_train = data_dir+"train_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square"
      fullpath_test = data_dir+"test_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square"
  elif recov == 1:
      fullpath_train= data_dir+"train_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_recov"
      fullpath_test= data_dir+"test_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_recov"
  elif recov == 2:
      fullpath_train= data_dir+"train_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_mix"
      fullpath_test= data_dir+"test_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_mix"

  fullpath = [fullpath_train, fullpath_test]
  for j in range(len(fullpath)):
      fp = fullpath[j].split('.')
      if display_data == True:
        print(fullpath[j])
      d = np.load(fullpath[j]+".npz", allow_pickle=True)
      X = (d["Xf"].astype(np.float32), d["Xt"].astype(np.float32))
      state = d["state"].astype(np.float32)
      y = d["y"].astype(np.float32)
      params = d["params"].astype(np.float32)
      Fs =  d["Fs"].astype(np.float32)
      x_input = X[0]
      x_time = X[1]
      y_output = y
      if j == 1: #Regarding test datasets, use full batches only; skip any remaining incomplete batch)
        ntest = int(len(Fs)-(len(Fs)%batch_size))
        params = params[:ntest]
        Fs = Fs[:ntest]
        x_input = x_input[:ntest,:]
        y_output = y_output[:ntest,:]
      
      x_data_input = x_input # x_input = [parametes of repetitive stimuli | repetitive stimuli (input function)]
      x_data_time = x_time 
      y_data = y_output # y_output = [parametes of response | response (output function)]
      
      # Split parameters and input/output functions
      ps_data = x_data_input[:,0:numi]
      x_data_input = x_data_input[:,numi:]
      habit_data = y_data[:,:numo]
      y_data = y_data[:,numo:]

      # Downsample
      x_data_input = x_data_input[:,::sub]
      x_data_time = x_data_time[::sub]
      y_data = y_data[:,::sub]

      # Concatenate parameters and downsampled input/output functions
      #p_expanded = np.repeat(ps_data[:,0].reshape(-1,1), x_data.shape[1], axis=1)
      #x_data = torch.cat([ps_data, x_data],1)
      #y_data = torch.cat([habit_data, y_data],1)
      #x_data = x_data.to(device)
      #y_data = y_data.to(device)
      if j == 0:
        x_train = (x_data_input, x_data_time)
        y_train = y_data
        #x_train = x_train.reshape(ntrain,x_train.shape[1],1)
      else:
        x_test = (x_data_input, x_data_time)
        y_test = y_data
        #x_test = x_test.reshape(ntest,x_test.shape[1],1)

  return x_train, y_train, x_test, y_test

def train_model_donet(exid, x_train, y_train, x_test, y_test, learning_rate = 0.001, epochs = 500, batch_size = 20, step_size = 50, gamma = 0.5, L = 4, width = 64, sub_int = 3, model_name = "donet", sys_name= "TysonAL2D"):
  print(x_train[0].shape)
  print(x_train[1].shape)
  data = dde.data.TripleCartesianProd(
      X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test
  )
  save_dir = "model_"+model_name
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # Build a model
  m = len(x_train[1]) 
  dim_x = 1 # time
  branch = [m]
  trank = [dim_x]
  for l in range(L):
    branch.append(width)
    trank.append(width)
  net = dde.nn.DeepONetCartesianProd(
      branch,
      trank,
      "tanh",
      "Glorot normal",
  )
  model = dde.Model(data, net)
  
  # Compile and Train
  model.compile("adam", lr=learning_rate, loss = 'mean l2 relative error', metrics=["mean l2 relative error", "mse"])
  save_path = save_dir+"/"+"train"+str(exid)+"_"+(model_name)+"_"+sys_name+"_epochs"+str(epochs)+"_sub"+str(2**sub_int)
  losshistory, train_state = model.train(iterations=epochs, model_save_path=save_path)
  
  # Plot the loss trajectory
  #dde.utils.plot_loss_history(losshistory)
  dde.utils.save_loss_history(losshistory, fname=save_path+"_loss")
  #plt.savefig(fname)
  losses = np.array(losshistory.loss_train)
  min_train_loss = np.min(losses[:, 0])
  return min_train_loss

# TODO 
def predict_plot_donet():
  return
   
def predict_donet(exids, sys_name, x_train, y_train, x_test, y_test, T, ep, num_dp, sub, sub_int, L, width, recov, hpo = False):
    # Load data sets
    num_grids_T = T*num_dp
    Xf = x_train[0]
    Xf = Xf.squeeze()
    m = Xf.shape[1]
     
    ng = y_train.shape[1]//T
    Xt = x_train[1]
    X_train = (Xf, Xt)

    Xf = x_test[0]
    Xf = Xf.squeeze()
    X_test = (Xf, Xt)
    data = dde.data.Triple(
     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # Save folder
    parent = "train"+str(exids[1])+"_"+"pred"+str(exids[0])+"_donet_"+sys_name
    if not os.path.exists(parent):
      os.makedirs(parent)
    if recov == 0:
      save_parent_dir = parent+"/pred"
    elif recov == 1:
      save_parent_dir = parent+"/recov"
    elif recov == 2:
      save_parent_dir = parent+"/mix"
    if not os.path.exists(save_parent_dir):
      os.makedirs(save_parent_dir)
    
    # Select optimzied hyperparameters if hpo == True
    if hpo == True: # With HPO
      d = np.load("HPO/train"+str(exids[1])+"_donet.npy")
      L = d[0]
      width = d[1]

    # Build a model
    m = len(x_train[1]) 
    dim_x = 1 # time
    branch = [m]
    trank = [dim_x]
    for l in range(L):
      branch.append(width)
      trank.append(width)
    net = dde.nn.DeepONetCartesianProd(
        branch,
        trank,
        "tanh",
        "Glorot normal",
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, loss = 'mean l2 relative error', metrics=["mse", "mean l2 relative error"])
    
    # Copy trained model (.pt)
    resultname = "train"+str(exids[1])+"_donet_"+sys_name+"_epochs"+str(ep)+"_sub"+str(sub)
    load_dir = "model_donet"
    save_path = load_dir+"/"+resultname +"-"+str(ep)+".pt"
    if not os.path.exists(save_path):
      print(save_path)
      return [None for i in range(4)]
    print("(L, width)=(",str(L),",",str(width),")")
    model.restore(save_path = save_path)

    # Prediction
    y_pred = model.predict(x_test)
    l2_test = 0
    mse_test = 0
    mae_test = 0
    for i in range(len(y_test)):
      l2_test += dde.metrics.l2_relative_error(y_test, y_pred)
      mse_test += dde.metrics.mean_squared_error(y_test, y_pred)
      mae_test += np.mean(np.abs(y_test - y_pred))
    l2_test /= len(y_test)
    mse_test /= len(y_test)
    mae_test /= len(y_test)
    sub_at_train = 2**sub_int
    print("train(#grids "+str(num_dp//sub_at_train)+"), test(#grids "+str((y_test.shape[1]//2))+"): pred", " MSE/#grids", '{:.3e}'.format(mse_test)," MAE/#grids", '{:.3e}'.format(mae_test), "L2loss",'{:.3e}'.format(l2_test))
    num_dmodel_params = m*width+width*width*L+dim_x*width+width*width*L
    return mse_test, l2_test, num_dmodel_params, y_pred
