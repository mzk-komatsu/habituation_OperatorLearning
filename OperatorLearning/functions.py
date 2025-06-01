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


# Loas parameters related to habituation systems and data sets
def get_settings(train_path, json_path, global_path):
    print("Training:", train_path)
    #Load parameters related to habituating systems
    with open(json_path, 'r', encoding='utf-8') as f:
        sys_JSON = json.load(f)
    F = sys_JSON["F"]
    T = sys_JSON["T"]
    num_dp = sys_JSON["num_dp"]
    ntest = int(sys_JSON["num_input_func"]*sys_JSON["rate"][0])
    ntrain = int(sys_JSON["num_input_func"]*sys_JSON["rate"][1])
    bs = int((20/1000)*(ntrain+ntest))
    f_mean = sys_JSON["freq"][0]
    f_var = sys_JSON["freq"][1]

    #Load parameters related to data set
    with open(global_path, 'r', encoding='utf-8') as f:
        gJSON = json.load(f)
    numi = gJSON["numi"]
    numo = gJSON["numo"]
    pred_ntrain =int(gJSON["pred_rate"]*ntrain)
    pred_ntest = int(gJSON["pred_rate"]*ntest)
    split = gJSON["split"]

    # Define grids
    with open(train_path, 'rb') as f:
        yml = yaml.safe_load(f)
    sub_int = yml['sub_int']
    grids = str(T*num_dp//(num_dp**sub_int))

    return F, T, num_dp, ntrain, ntest, bs, grids, f_mean, f_var, numi, numo, pred_ntrain, pred_ntest, split

# Load parameters related to training and prediction
def get_learn_settings(train_path):
  with open(train_path, 'rb') as f:
      yml = yaml.safe_load(f)
  ep = yml['ep']
  model_name = yml["model_name"]
  sys_name = yml["sys_name"]
  if model_name == "fno":
      model_id = 1
  elif model_name == "lno":
      model_id = 2
  else:
      model_id = 0
  sub_int = yml['sub_int']
  n_modes = yml['modes']
  recov = yml['recov']
  learning_rate = yml['learning_rate']
  step_size = yml['step_size']
  gamma= yml['gamma']
  L= yml['L']
  width = yml['width']
  return ep, sub_int, model_name, sys_name, model_id, n_modes, recov, learning_rate, step_size, gamma, L, width


# Load specified data sets
def load_data(sys_name, ntrain=1000, ntest=1000, sub=2**3, h=1024, num_dp=1024, batch_size=20, freq_mean = 10, freq_var = 1, Flevel = 30, T=1,rand=False, numi=2, numo=4, recov=0, display_data = False):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  json_file = "../data_generation/"+sys_name+".json"
  with open(json_file, 'r', encoding='utf-8') as f:
    jsondata = json.load(f)
  data_dir = "../data_generation/data_"+jsondata["config_name"]+"/"
  if recov == 0:
      fullpath_train = data_dir+"train_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square.npz"
      fullpath_test = data_dir+"test_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square.npz"
  elif recov == 1:
      fullpath_train= data_dir+"train_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_recov.npz"
      fullpath_test= data_dir+"test_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_recov.npz"
  elif recov == 2:
      fullpath_train= data_dir+"train_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_mix.npz"
      fullpath_test= data_dir+"test_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(ntrain+ntest)+"-F"+str(Flevel)+"-freq"+str(freq_mean)+"-"+str(freq_var)+"square_mix.npz"

  fullpath = [fullpath_train, fullpath_test]
  for j in range(len(fullpath)):
      fp = fullpath[j].split('.')
      if display_data == True:
        print(fullpath[j])
      d = np.load(fullpath[j], allow_pickle=True)
      X = (d["Xf"].astype(np.float32), d["Xt"].astype(np.float32))
      state = d["state"].astype(np.float32)
      y = d["y"].astype(np.float32)
      params = d["params"].astype(np.float32)
      Fs =  d["Fs"].astype(np.float32)
      x_input = X[0]
      y_output = y
      if j == 1: #Regarding test datasets, use full batches only; skip any remaining incomplete batch)
        ntest = int(len(Fs)-(len(Fs)%batch_size))
        params = params[:ntest]
        Fs = Fs[:ntest]
        x_input = x_input[:ntest,:]
        y_output = y_output[:ntest,:]
      x_data = torch.from_numpy(x_input) # x_input = [parametes of repetitive stimuli | repetitive stimuli (input function)]
      y_data = torch.from_numpy(y_output) # y_output = [parametes of response | response (output function)]
      if rand == True:
        idx = sorted(random.sample(range(T*num_dp), k=T*int(num_dp/sub)))
        x_data = x_data[:,idx]
        y_data = y_data[:,idx]
      else:
        # Split parameters and input/output functions
        ps_data = x_data[:,0:numi]
        x_data = x_data[:,numi:]
        habit_data = y_data[:,:numo]
        y_data = y_data[:,numo:]

        # Downsample
        x_data = x_data[:,::sub]
        y_data = y_data[:,::sub]

        # Concatenate parameters and downsampled input/output functions
        p_expanded = np.repeat(ps_data[:,0].reshape(-1,1), x_data.shape[1], axis=1)
        x_data = torch.cat([ps_data, x_data],1)
        y_data = torch.cat([habit_data, y_data],1)
        x_data = x_data.to(device)
        y_data = y_data.to(device)
      if j == 0:
        x_train = x_data
        y_train = y_data
        x_train = x_train.reshape(ntrain,x_train.shape[1],1)
      else:
        x_test = x_data
        y_test = y_data
        x_test = x_test.reshape(ntest,x_test.shape[1],1)

  # Loader
  train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)
  return train_loader, test_loader, x_train, y_train, x_test, y_test, params, Fs


def my_reshape(x,y):
  bs = x.shape[0]
  return x.view(bs,-1), y.view(bs,-1)

def set_model_params(model_name, model_params):
    if model_name == "fno":
        dict_model_params = {
                        "L": model_params[0],
                        "width": model_params[1],
                        "modes": model_params[2],
                        }
    elif model_name == "lno":
        dict_model_params = {
                        "L": model_params[0],
                        "width": model_params[1],
                        "rank": model_params[2],
                        }
    return dict_model_params

def build_model(model_name, exid, hpo, s, dict_model_params):
    if model_name == "fno":
        if hpo == True: # After HPO
            d = np.load("HPO/train"+str(exid)+"_"+model_name+".npy")
            L = d[0]
            width = d[1]
            modes = d[2]
        else:
            modes = dict_model_params["modes"]
            width = dict_model_params["width"]
            L = dict_model_params["L"]
        convs = [SpectralConv1d(width, width, modes) for _ in range(L)]
        ws = [nn.Conv1d(width, width, 1) for _ in range(L)]
        model = FNO1dL(modes, width, convs, ws).to(device)
        #model = FNO1d(modes, width).to(device) # If L is fixed to 4
        #print("fno(L =", str(L)+", width=", str(width), "modes =", str(modes),")")
    elif model_name == "lno":
        if hpo == True: # After HPO
            d = np.load("HPO/train"+str(exid)+"_"+model_name+".npy")
            L = d[0]
            width = d[1]
            rank = d[2]
        else:
            rank = dict_model_params["rank"]
            width = dict_model_params["width"]
            L = 4
        nets = [LowRank1d(width, width, s, width, rank) for _ in range(L)]
        ws = [nn.Linear(width, width) for _ in range(L)]
        bs = [nn.BatchNorm1d(width) for _ in range(L-1)]
        model = LRNOL(nets, ws, bs, width, rank).to(device)
        #model = LRNO(s, width, rank).to(device) # If L is fixed to 4
        #print("lno(L =", str(L)+", width=", str(width), "rank =", str(rank),")")
    return model

# Training
def train_model(exid, train_loader, test_loader, x_train, y_train, x_test, y_test, learning_rate = 0.001, epochs = 500, batch_size = 20, step_size = 50, gamma = 0.5, L = 4, modes = 16, width = 64, sub_int = 3, model_name="fno", sys_name= "TysonAL2D", numi=2, numo=4, s = 128, hpo=False):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(0)
  np.random.seed(0)
  true_ht = 0
  myloss = LpLoss(p=2, size_average=True)

  # Prepare for building model
  if model_name == "fno":
      model_params = [L, width, modes]
  elif model_name == "lno":
      model_params = [L, width, modes]
  dict_mp = set_model_params(model_name, model_params)
  print(dict_mp,exid,hpo,s)
  # Build model
  model = build_model(model_name, exid, hpo, s, dict_mp)
  save_dir = "model_"+model_name
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # training and evaluation
  optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0) #1e-4)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  for ep in range(epochs):
      # Training
      model.train()
      t1 = default_timer()
      train_mse = 0 #MSE(data) evaluated for each batch
      train_l2 = 0 #L2(data) + l0*L2(ht0) + l1*L2(ht1) + l2*L2(ht2) + l3*L2(ht3) evaluated for each batch
      test_mse = 0
      test_l2 = 0
      l2_temp = 0
      train_temp = 0
      for x, y in train_loader:
          x, y = x.to(device), y.to(device)
          x.requires_grad = True
          optimizer.zero_grad()
          thres = y[:, 0].reshape((batch_size,1)).to(x.device)
          out = model(x)
          y_mse = y[:, numo:]
          y_ht = y[:, :numo]
          out_mse = out
          mse = F.mse_loss(out_mse.view(batch_size, -1), y_mse.view(batch_size, -1), reduction='mean')
          mse_item = mse.item()

          out_r, y_r = my_reshape(out_mse, y_mse)
          data_loss = myloss(out_r, y_r)

          l2 = data_loss
          l2.backward()

          optimizer.step()
          train_mse += mse_item
          train_l2 += l2.item()
      scheduler.step()

      # Evaluation
      model.eval()
      test_l2 = 0.0
      with torch.no_grad():
          for x, y in test_loader:
              x, y = x.to(device), y.to(device)
              x.requires_grad = True
              out = model(x)
              y_mse = y[:, numo:]
              y_ht = y[:, :numo]
              out_mse = out
              mse = F.mse_loss(out_mse.view(batch_size, -1), y_mse.view(batch_size, -1), reduction='mean')
              mse_item = mse.item()
              out_r, y_r = my_reshape(out_mse, y_mse)
              data_loss = myloss(out_r, y_r)
              l2 = data_loss

              test_mse += mse_item
              test_l2 += l2.item()

      train_mse /=(len(train_loader))
      test_mse /=(len(test_loader))
      train_l2 /=(len(train_loader))
      test_l2 /=(len(test_loader))

      t2 = default_timer()
      print(ep, "L2(train, test):", '{:.3e}'.format(train_l2), '{:.3e}'.format(test_l2),":",data_loss.item(), "MSE(train, test):", '{:.3e}'.format(train_mse), '{:.3e}'.format(test_mse))

  save_path = save_dir+"/"+"train"+str(exid)+"_"+(model_name)+"_"+sys_name+"_epochs"+str(epochs)+"_sub"+str(2**sub_int)+".pt"
  torch.save(model.state_dict(), save_path)

  return


def train_model_hpo(exid, train_loader, test_loader, x_train, y_train, x_test, y_test, learning_rate = 0.001, epochs = 500, batch_size = 20, step_size = 50, gamma = 0.5, L = 4, modes = 16, width = 64, sub_int = 3,  model_name = "fno", numi=2, numo=4, s = 128):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(0)
  np.random.seed(0)
  true_ht = 0
  myloss = LpLoss(p=2, size_average=True)

  if model_name == "fno":
      convs = [SpectralConv1d(width, width, modes) for _ in range(L)]
      ws = [nn.Conv1d(width, width, 1) for _ in range(L)]
      model = FNO1dL(modes, width, convs, ws).to(device)
      #model = FNO1d(modes, width).to(device)
  elif model_name == "lno":
      rank = modes
      nets = [LowRank1d(width, width, s, width, rank) for _ in range(L)]
      ws = [nn.Linear(width, width) for _ in range(L)]
      bs = [nn.BatchNorm1d(width) for _ in range(L-1)]
      model = LRNOL(nets, ws, bs, width, modes).to(device)
  # training and evaluation
  optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0) 
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  l2loss_log = torch.zeros((1,epochs))
  for ep in range(epochs):
      model.train()
      t1 = default_timer()
      train_mse = 0 
      train_l2 = 0
      test_mse = 0
      test_l2 = 0
      l2_temp = 0
      train_temp = 0
      for x, y in train_loader:
          x, y = x.to(device), y.to(device)
          x.requires_grad = True
          optimizer.zero_grad()
          thres = y[:, 0].reshape((batch_size,1)).to(x.device)
          out = model(x)
          y_mse = y[:, numo:]
          y_ht = y[:, :numo]

          out_mse = out
          mse = F.mse_loss(out_mse.view(batch_size, -1), y_mse.view(batch_size, -1), reduction='mean')
          mse_item = mse.item()

          out_r, y_r = my_reshape(out_mse, y_mse)
          data_loss = myloss(out_r, y_r)

          l2 = data_loss
          l2.backward()

          optimizer.step()
          train_mse += mse_item
          train_l2 += l2.item()

      scheduler.step()

      train_mse /=(len(train_loader))
      train_l2 /=(len(train_loader))
      l2loss_log[0,ep] = train_l2
      t2 = default_timer()
      print(ep, "L2:", '{:.3e}'.format(train_l2), "MSE:", '{:.3e}'.format(train_mse))
  save_path = "exid"+str(exid)+"_"+model_name+"_"+str(epochs)+"_sub"+str(2**sub_int)+".pt"
  torch.save(model.state_dict(), save_path)
  min_l2loss = torch.min(l2loss_log.flatten()).item()
  return min_l2loss


# ----------------------------------------------------
# Prediction
# [Output]
#   pred: predicted responses,
#   pred_ht: parameters of predicted respnoses
#   pred_mse: test errors
#   pred_l2: test
#   num_dmodel_params: # of parameters of NO/DoNet
# ----------------------------------------------------
def predict(exids, sys_name, num_data, test_loader, bs, sub_at_train, sub_int, freq_name, Flevel, T, model_id, model_name, num_dp = 1024, epochs = 500, L = 4, modes = 16, width = 64, numi=2, numo=4, recov=0, hpo = False):
  num_grids = int(num_dp//(2**sub_int)) # Number of grids in [0, 1]
  num_grids_T = T*num_grids # Number of grids in simulation period

  # Setting for directories (save results of prediction)
  # Ex)
  # train0_pred1_fno_TysonAL2D
  #   - pred
  #       - freq9.5-0.5_epoch10000_numgrids2048
  #       - freq9.5-0.5_epoch10000_numgrids1024
  #   - recov
  #       -
  #   - mix
  #       -
  parent = "train"+str(exids[1])+"_"+"pred"+str(exids[0])+"_"+(model_name)+"_"+sys_name
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
  save_dir = "freq"+str(freq_name)+"_epoch"+str(epochs)+"_numgrids"+str(num_grids)
  if not os.path.exists("./"+save_parent_dir+"/"+save_dir):
    os.makedirs("./"+save_parent_dir+"/"+save_dir)
  savedirname = "./"+save_parent_dir+"/"
  resultname = "train"+str(exids[1])+"_"+(model_name)+"_"+sys_name+"_epochs"+str(epochs)+"_sub"+str(sub_at_train)
  load_dir = "model_"+model_name

  # Copy trained model (.pt)
  save_path = load_dir+"/"+resultname +".pt"
  shutil.copy(save_path, "./"+save_parent_dir)

  # Load data used in prediction
  # timestamps of stimuli)
  dataname = "train_"+sys_name+"-nd"+str(num_dp)+"-nf"+str(num_data)+"-F"+str(Flevel)+"-freq"+freq_name
  if recov == 0:
      dataname += "square"
  elif recov == 1:
      dataname += "square_recov"
  elif recov == 2:
      dataname += "square_mix"
  json_file = "../data_generation/"+sys_name+".json"
  with open(json_file, 'r', encoding='utf-8') as f:
    jsondata = json.load(f)
  data_folder_name = "data_"+jsondata["config_name"]
  fr = open("../data_generation/"+data_folder_name+"/"+dataname+"_s.txt", "rb")
  stimulus_t = pickle.load(fr)

  # Prepare for building model
  model_params = [L, width, modes]
  dict_mp = set_model_params(model_name, model_params)
  # Define and load model trained
  model = build_model(model_name, exids[1], hpo, num_grids_T, dict_mp)
  model.load_state_dict(torch.load(save_path,weights_only=True))
  model.eval()

  # Prediction settings
  myloss = LpLoss(p=2, size_average=True)
  num_minibatch = len(test_loader)
  pred = torch.zeros((num_minibatch, bs, num_grids_T))  # Array to store prediction
  pred_ht = torch.zeros((num_minibatch, bs, numo))  # Array to store parameters of predicted response
  y_ht = torch.zeros((num_minibatch,bs, numo))  # Array to store parameters of true response

  # Predicton
  index = 0
  pred_mse = 0
  pred_l2 = 0
  pred_mae = 0
  with torch.no_grad():
    for x, y in test_loader:
      test_l2 = 0
      x, y = x.to(device), y.to(device)
      out = model(x)
      pred_y = out.squeeze()
      pred[index] = pred_y

      stimulus_t = stimulus_t[:]
      max_ys, ht_ys, ht_s, energy = tools.get_output_params(stimulus_t, x[0,numi:].cpu().numpy().squeeze(), pred[index].cpu().numpy(),ht_thres,T,num_grids,bs)
      temp= np.concatenate([max_ys.reshape(-1,1), ht_ys.reshape(-1,1), ht_s.reshape(-1,1), energy.reshape(-1,1)],axis=1)
      temp = temp.astype(np.float32)
      pred_ht[index] = torch.from_numpy(temp).clone()
      y_mse = y[:,numo:]
      y_ht[index] = y[:,:numo]
      mse = F.mse_loss(pred_y.view(bs, -1), y_mse.view(bs, -1), reduction='mean')
      pred_mse += mse.item()
      mae = F.l1_loss(pred_y.view(bs, -1), y_mse.view(bs, -1), reduction='mean')
      pred_mae += mae.item()
      data_loss = myloss(pred_y.view(bs, -1), y_mse.view(bs, -1)).item()
      pred_l2 += data_loss
      index = index + 1

    # Evaluation
    pred_l2 /= num_minibatch
    pred_mse /= num_minibatch
    pred_mae /= num_minibatch
    print("train(#grids "+str(num_dp//sub_at_train)+"), test(#grids "+str(num_grids)+"): pred", " MSE/#grids", '{:.3e}'.format(pred_mse), " MAE/#grids", '{:.3e}'.format(pred_mae), "L2loss",'{:.3e}'.format(pred_l2))

  num_dmodel_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
  return pred, pred_ht, pred_mse, pred_l2, num_dmodel_params


# Plot parameters of (true, predicted) response vs (true, predicted) repetitive stimuli
def predict_analysis(x_test_in_batch, yt_ht_arr, pred_ht_arr, freq_name, save_parent_dir,save_dir):
    # Parameters of repetitive stimuli
    input_F = x_test_in_batch[:,1].to('cpu').detach().numpy().copy() # Strength of repetitive stimuli
    input_Freq = x_test_in_batch[:,0].to('cpu').detach().numpy().copy() # Frequency of repetitive stimuli

    # Parameters of predicted/true response
    pred_ht_arr = pred_ht_arr.reshape(yt_ht_arr.shape)
    max_y = yt_ht_arr[:,0] # Max. of true response
    Max_pred = pred_ht_arr[:,0] # Max. of predicted response
    rel_y = (yt_ht_arr[:,0]-yt_ht_arr[:,1])/yt_ht_arr[:,0] # Relative attenuation rate of true response
    rel_pred =  (pred_ht_arr[:,0]-pred_ht_arr[:,1])/pred_ht_arr[:,0] # Relateive attenuation rate of predicted response
    ht_y = yt_ht_arr[:,2] # Habituation time of true response
    ht_pred = pred_ht_arr[:,2] # Habituation time of predicted response

    # Plot relative attenuation rate
    plt.scatter(input_Freq, rel_y, color='black', marker='o',label="true")
    plt.scatter(input_Freq, rel_pred, color='red', marker='+' ,label="pred")
    halflen = int(len(rel_y)/2)
    mape1 = mean_absolute_percentage_error(rel_y[:halflen], rel_pred[:halflen])
    mape2 = mean_absolute_percentage_error(rel_y[halflen:], rel_pred[halflen:])
    mape = mean_absolute_percentage_error(rel_y, rel_pred)
    tit = '{:.3f}'.format(mape1)+","+'{:.3f}'.format(mape2)+","+'{:.3f}'.format(mape)
    plt.xlabel("Frequency")
    plt.ylabel("Attenuation Rate")
    plt.ylim([0.3, 0.6])
    fm_fv = freq_name.split("-")
    plt.xlim([float(fm_fv[0])-float(fm_fv[1]), float(fm_fv[0])+float(fm_fv[1])])
    plt.legend()
    plt.savefig("./"+save_parent_dir+"/"+save_dir+"/Freq-RelAtten"+'{:.3f}'.format(mape)+".png")
    plt.close()

    # Plot habituation time
    plt.scatter(input_Freq, ht_y, facecolors='k', edgecolors='k',label="true")
    plt.scatter(input_Freq, ht_pred, facecolors='r', edgecolors='r',label="pred")
    mape1 = mean_absolute_percentage_error(ht_y[:halflen], ht_pred[:halflen])
    mape2 = mean_absolute_percentage_error(ht_y[halflen:], ht_pred[halflen:])
    mape = mean_absolute_percentage_error(ht_y, ht_pred)
    tit = '{:.3f}'.format(mape1)+","+'{:.3f}'.format(mape2)+","+'{:.3f}'.format(mape)
    plt.title(tit)
    plt.title(tit)
    plt.xlabel("Frequency")
    plt.ylabel("Habituation time")
    plt.ylim([0, 1.0])
    plt.xlim([float(fm_fv[0])-float(fm_fv[1]), float(fm_fv[0])+float(fm_fv[1])])
    plt.legend()
    plt.savefig("./"+save_parent_dir+"/"+save_dir+"/HT.png")
    plt.close()
    return

# Plot repetitive stimuli (input) and predicted/true response (ouput)
# (Some of the input-putput pairs in the first minibatch of datasets for prediction)
def predict_plot(exids, sys_name, num_data, x_test, y_test, test_loader, bs, params, Fs, sub_at_train, sub_int, freq_name, F, T, model_id, model_name, num_dp = 1024, epochs = 500, L = 4, modes = 16, width = 64, num_display=5, numi=2, numo=4, recov=0, hpo = False):
  save_path = model_name+str(epochs)+".pt"
  num_grids = int(num_dp//(2**sub_int))
  num_grids_T = T*num_grids
  pred, pred_ht, pred_mse, pred_l2, num_dmodel_params = predict(exids, sys_name, num_data, test_loader, bs, sub_at_train, sub_int, freq_name, F, T, model_id, model_name, num_dp = num_dp, epochs = epochs, L = L, modes = modes, width = width,  numi=numi, numo=numo,recov=recov, hpo = hpo)
  save_dir = "freq"+str(freq_name)+"_epoch"+str(epochs)+"_numgrids"+str(num_dp//(2**sub_int))
  parent = "train"+str(exids[1])+"_"+"pred"+str(exids[0])+"_"+(model_name)+"_"+sys_name
  if recov == 0:
    save_parent_dir = parent+"/pred"
  elif recov == 1:
    save_parent_dir = parent+"/recov"
  elif recov == 2:
    save_parent_dir = parent+"/mix"
  pred3d = pred
  cnt= 0
  tmp = iter(test_loader)
  x_test_in_batch = x_test
  y_test_in_batch = y_test
  pred = pred3d[0,:,:,] #the first minibatch

  num_minibatch = len(test_loader)
  num_display = min(bs,num_display)
  yt_ht_arr = y_test_in_batch[:,:numo].cpu().numpy()
  pred_ht_arr = pred_ht[0,:,:numo].cpu().numpy()
  r2_arr = np.zeros((numo,1))

  #Use the first minibatch to plot prediction graphs
  for i in range(num_display):
    plt.figure()
    yval_temp = y_test_in_batch[i].cpu().numpy()
    xval_temp = x_test_in_batch[i].cpu().numpy()
    predval = pred[i].cpu().numpy()

    t = np.arange(0, T, 1/num_grids)
    yval = yval_temp[numo:]
    plt.plot(t,yval, 'k-', label='true')
    plt.plot(t,predval,'r:', label='pred')
    plt.axvline(x=yt_ht_arr[i,2], color="k", linestyle="dashed")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.savefig("./"+save_parent_dir+"/"+save_dir+"/pred_"+str(i)+"{:.2f}".format(params[i])+".png")
    plt.close()

    xval = xval_temp[numi:]
    plt.plot(t,xval, 'k-')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.savefig("./"+save_parent_dir+"/"+save_dir+"/input_"+str(i)+"{:.2f}".format(params[i])+".png")
    plt.close()

  # Plot parameters of (true, predicted) response vs (true, predicted) repetitive stimuli
  predict_analysis(x_test_in_batch, yt_ht_arr, pred_ht[:,:,:numo].cpu().numpy(), freq_name, save_parent_dir,save_dir)

  for i in range(num_display):
    predval = pred[i].cpu().numpy()
    t = np.arange(0,T,1/num_grids)

  #Use all of the minibatches to plot R2 graphs
  yt_ht_arr = y_test[:, :numo].cpu().numpy()
  pred_ht_arr = pred_ht.view(-1, pred_ht.size(2)).cpu().numpy()
  yval = y_test[:,numo:].cpu().numpy()
  predval = pred3d.view(-1, pred3d.size(2)).cpu().numpy()
  x_ht_arr = x_test[:,:numi].cpu().numpy() #

  # Save predicted response, parameters of predicted response and prediction error (mse, l2) and # of parameters in NO/donet
  np.savez("./"+save_parent_dir+"/"+save_dir+"/prediction", r2_arr = r2_arr, pred_ht_arr = pred_ht_arr, yt_ht_arr = yt_ht_arr, yval=yval, predval=predval)
  return pred, pred_ht_arr, pred_mse, pred_l2, num_dmodel_params

