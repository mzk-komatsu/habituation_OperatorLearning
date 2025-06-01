import json
import sys
import numpy as np
import os
import sys
import pickle
import input_functions, data_utils

def get_sysparams(data_path='../data_settings.json'):
  json_file = open(data_path, 'r')  
  j_data = json.load(json_file)
  dim_sys = j_data["dim_sys"] #2
  sys =j_data["sys"]
  params_sys = j_data["sys_params"]
  return dim_sys, sys, params_sys

def get_params_inputdata(data_path='../data_settings.json'):
  json_file = open(data_path, 'r')  
  j_data = json.load(json_file)
  T = j_data["T"] #1
  num_dp =  j_data["num_dp"] #100
  num_input_func = j_data["num_input_func"] #100
  F = j_data["F"] #25 #intensity
  rate = j_data["rate"] #[0.8, 0.2]
  w = j_data["w"] #3 only need for approx_delta input
  freq = j_data["freq"]#freqency freq[0]-freq[1]*np.random.rand((num_input_func))
  input_func = j_data["input_func"] #gauss or square
  out_var = j_data["out_var"]
  return T, num_dp, num_input_func, F, rate, w, freq, input_func, out_var

def get_dataname(data_path='../data_settings.json'):
  dim_sys, sys, params_sys = get_sysparams(data_path)
  T, num_dp, num_input_func, F, rate, w, freq, input_func, _ = get_params_inputdata(data_path)
  dataname = sys+"-nd"+str(num_dp)+"-nf"+str(num_input_func)+"-F"+str(F)+"-freq"+str(freq[0])+"-"+str(freq[1])+input_func
  return dataname
