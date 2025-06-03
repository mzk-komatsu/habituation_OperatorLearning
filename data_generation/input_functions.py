import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import sin, sqrt
from scipy.integrate import odeint
import bokeh.application
import bokeh.application.handlers
import bokeh.io
import bokeh.models
import bokeh.plotting
from sklearn.preprocessing import StandardScaler
from matplotlib import animation, rc
from IPython.display import HTML
from scipy import interpolate, signal
from scipy.signal import find_peaks
import random

def generate_input_t_data(T, num_dp):
  return np.arange(0,T,T/num_dp).reshape(num_dp,1)


def approx_delta(T, num_dp, Fs, params, w, split = 1):
  time = np.linspace(0, split*T, split*T*num_dp, endpoint=False)
  input_func_data = np.zeros((len(params),split*T*num_dp))
  stimulus_time = []
  for i in range(len(params)):
    input_t_idx = np.arange(0,num_dp,int(num_dp/params[i]))
    temp = []
    for j in range(w):
      temp = temp + (input_t_idx + j).tolist()
    on_t_idx = np.array(list(set(temp)))
    on_t_idx = np.delete(on_t_idx, np.where(on_t_idx >= num_dp))
    input_func_data[i,on_t_idx] = Fs[i]
    stimulus_time.append((time[input_t_idx]).tolist())
  return input_func_data, stimulus_time


def approx_delta_recov(T, num_dp, Fs, params, w, recov_rate = 1, split = 2, recov_stim_range=[1.3, 1.9]):
  rng = np.random.default_rng()
  rt = rng.uniform(recov_stim_range[0], recov_stim_range[1], int(len(params)/(1/recov_rate)))
  time = np.linspace(0, split*T,  split*num_dp, endpoint=False)
  input_func_data = np.zeros((len(params),split*T*num_dp))
  stimulus_time = []
  for i in range(len(params)):
    input_t_idx = np.arange(0, num_dp, int(num_dp/params[i]))
    if i%recov_rate == 0:
        input_t_idx = np.append(input_t_idx, int(rt[i//int(1/recov_rate)]*num_dp))
    temp = []
    for j in range(w):
      temp = temp + (input_t_idx + j).tolist()
    on_t_idx = np.array(list(set(temp)))
    on_t_idx = np.delete(on_t_idx, np.where(on_t_idx >= split*T*num_dp))
    input_func_data[i,on_t_idx] = Fs[i]
    stimulus_time.append((time[input_t_idx]).tolist())
  return input_func_data, stimulus_time

def display_input(T=1,num_dp=1000,num_input_func=10, F=10,freq=[12,2],method="gauss",save =1):
  t = np.arange(0,T,T/num_dp)
  params = freq[0]-freq[1]*np.random.rand((num_input_func))
  if method == "square":
    # square
    fname = method
    ww=0.03
    w = int(ww/(T/num_dp))
    Xf, stimulus_t = approx_delta(T, num_dp, F, params[num_input_func-1:], w)
  elif method == "gauss":
    # gaussian
    fname = "gauss-num_dp" + str(num_dp)+ ""
    Xf, stimulus_t = approx_delta_gauss(T, num_dp, F, params, t)

  plt.plot(t, Xf[0])
  plt.scatter(stimulus_t[0],F*np.ones(len(stimulus_t[0])))
  plt.title(fname)
  plt.xlabel("t")
  plt.ylabel("u")
  if save == 1:
    plt.savefig(fname+".png")
  plt.show()
  return Xf, stimulus_t
