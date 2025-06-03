import numpy as np
from numpy import sqrt
from scipy.integrate import odeint
import bokeh.application
import bokeh.application.handlers
import bokeh.io
import bokeh.models
from scipy import signal
import os
import sys
import pickle
import json
from sklearn import preprocessing as prep
import input_functions
import data_utils

sys.path.append("../systems")
sys.path.append("../analysis")
from tools import get_output_params
import systems as hs

def get_rhs(sys):
  if sys == "TysonAL2D":
    rhs_inp = hs.TysonAL2D_input_interporate
  elif sys == "NegativeFB":
    rhs_inp = hs.NegativeFB_input_interporate
  elif sys == "GMinimal":
    rhs_inp = hs.GMinimal_input_interporate
  elif sys == "CMinimal":
    rhs_inp = hs.CMinimal_input_interporate
  elif sys == "Test":
    rhs_inp = hs.Test_input_interporate
  return rhs_inp
  
args = sys.argv
print(args)
if len(args) == 1:
  dim_sys, sys, params_sys = data_utils.get_sysparams()
  T, num_dp, num_input_func, F, rate, w, freq, input_func, out_var = data_utils.get_params_inputdata()
  dataname = data_utils.get_dataname()
  data_folder_name = "./"
else:
  dim_sys, sys, params_sys = data_utils.get_sysparams(args[1])
  T, num_dp, num_input_func, F, rate, w, freq, input_func, out_var = data_utils.get_params_inputdata(args[1])
  dataname = data_utils.get_dataname(args[1])
  data_folder_name = args[2]

with open("path.json", 'r', encoding='utf-8') as f:
    JSON = json.load(f)
global_path = JSON["global_path"]
with open(global_path, 'r', encoding='utf-8') as f:
    gJSON = json.load(f)
SPLIT = gJSON["split"]
RECOV_RATE = gJSON["recov_rate"]
numi = gJSON["numi"]
numo = gJSON["numo"]
ht_thres = gJSON["ht_thres"]

t = np.arange(0,SPLIT*T, T/num_dp)
if numi == 2:
    Fs = np.random.uniform(F-0,F+0, num_input_func)
    params = freq[0]-freq[1]*np.random.rand((num_input_func))

# Define habituating system
rhs_inp = get_rhs(sys)

"""
Generate input data (input to branch: Xf_*, input to trunk: Xt_*, * is whether train or test)
"""
num_train_data = int(num_input_func*rate[1])
num_test_data = int(num_input_func*rate[0])
Xt_train = np.linspace(0, SPLIT*T, SPLIT*num_dp, endpoint=False).reshape(-1,1)
Xt_test = np.linspace(0, SPLIT*T, SPLIT*num_dp, endpoint=False).reshape(-1,1)
#input_functions.generate_input_t_data(SPLIT*T, SPLIT*num_dp)
#Xt_test = input_functions.generate_input_t_data(SPLIT*T, SPLIT*num_dp)
if input_func == "square":
  Xf_train, stimulus_t_train = input_functions.approx_delta(T, num_dp, Fs, params[:num_train_data], w, split= SPLIT)
  Xf_test, stimulus_t_test = input_functions.approx_delta(T, num_dp, Fs, params[num_train_data:], w, split= SPLIT)
elif input_func == "square_recov" or input_func == "square_mix":
  if input_func == "square_recov":
      RECOV_RATE = 1
  Xf_train, stimulus_t_train = input_functions.approx_delta_recov(T, num_dp, Fs[:num_train_data], params[:num_train_data], w, recov_rate = RECOV_RATE, split = SPLIT, recov_stim_range=[1.3, 1.9])
  Xf_test, stimulus_t_test = input_functions.approx_delta_recov(T, num_dp, Fs[num_train_data:], params[num_train_data:], w, recov_rate = RECOV_RATE, split = SPLIT, recov_stim_range=[1.3, 1.9])

if numi == 2:
  Xf_train = np.concatenate([params[:num_train_data].reshape(-1,1), Fs[:num_train_data].reshape(-1,1), Xf_train],axis=1)
  Xf_test = np.concatenate([params[num_train_data:].reshape(-1,1), Fs[num_train_data:].reshape(-1,1), Xf_test],axis=1)



"""
Generate output data
"""
tt = t.tolist()
x0 = np.zeros(dim_sys)
y_train = np.zeros((num_train_data, len(t)))
y_test = np.zeros((num_test_data, len(t)))
ps = []
ts = []
state_train = np.zeros((num_train_data, len(t), dim_sys))
state_test = np.zeros((num_test_data, len(t), dim_sys))
for i in range(len(params)):
  if i < num_train_data:
    input_t = Xt_train.T[0]
    if numi == 2:
      input_f = Xf_train[i,numi:]
    x = odeint(rhs_inp, x0, t, (params_sys, input_t.tolist(), input_f.tolist()))
    state_train[i] = x
    y_train[i] = x[:,out_var]
  else:
    input_t = Xt_test.T[0]
    if numi == 2:
      #input_f = Xf_test[i-num_train_data,:]
      input_f = Xf_test[i-num_train_data,numi:]
    x = odeint(rhs_inp, x0, t, (params_sys, input_t.tolist(), input_f.tolist()))
    state_test[i-num_train_data] = x
    y_test[i-num_train_data] = x[:,out_var]

if numo == 4:
    max_ys, ht_ys, ht_s, energy = get_output_params(stimulus_t_train, t, y_train, ht_thres, T, num_dp, num_train_data)
y_train = np.concatenate([max_ys.reshape(-1,1), ht_ys.reshape(-1,1), ht_s.reshape(-1,1), energy.reshape(-1,1), y_train],axis=1)
if numo == 4:
    max_ys, ht_ys, ht_s, energy = get_output_params(stimulus_t_test, t, y_test, ht_thres, T, num_dp, num_test_data)
y_test = np.concatenate([max_ys.reshape(-1,1), ht_ys.reshape(-1,1), ht_s.reshape(-1,1), energy.reshape(-1,1), y_test],axis=1)

np.savez(data_folder_name+"/train_"+dataname, Xf = Xf_train, Xt = Xt_train, y = y_train, state = state_train, params = params[:num_train_data], Fs = Fs[:num_train_data])
np.savez(data_folder_name+"/test_"+dataname, Xf=Xf_test, Xt=Xt_test, y = y_test, state = state_test, params = params[num_train_data:], Fs = Fs[num_train_data:])

f = open(data_folder_name+"/train_"+dataname+"_s.txt", 'wb')
pickle.dump(stimulus_t_train, f)
f = open(data_folder_name+"/test_"+dataname+"_s.txt", 'wb')
pickle.dump(stimulus_t_test, f)
