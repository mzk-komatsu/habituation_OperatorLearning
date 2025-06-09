import json
import sys
import numpy as np
import os
import sys
from scipy.signal import find_peaks
from scipy.integrate import simpson
sys.path.append("../data_generation")
import input_functions


#Get habituation(num), y's peaks (index), habituation time (real time)
def get_peaks_and_ht(stimulus_t, y, T, num_dp, threshold=0.01):
  t, pt, hp = get_habituation_time(stimulus_t, y, threshold, T, num_dp)
  return t, hp,  hp/num_dp #hp*T/num_dp


# ---------------------------------------------
# parameters of output function
# maximum values of response, response at habituation time, habituation time, integral of response function
# ---------------------------------------------
def get_output_params(ss,t, ys,thres,T,num_dp,num_data):
  max_ys =np.zeros((num_data,1))
  ht_ys =np.zeros((num_data,1))
  ht_s =np.zeros((num_data,1))
  area = np.zeros((num_data,1))
  for i in range(num_data):
    habituation_num, peak_tidx, habituation_time = get_peaks_and_ht(ss[i], ys[i], T, num_dp)
    peaks, throughs = get_peak_trough(ys[i])
    max_y = np.max(peaks[1])
    ht_idx_peaks = np.where(np.array(peaks[0])==peak_tidx)[0]
    if len(ht_idx_peaks) == 0:
        ht_idx_peaks = 0
    else:
        ht_idx_peaks =  ht_idx_peaks[0] # index of habituation time
    ht_y = peaks[1][ht_idx_peaks] # response at habituation time
    after_ht = ys[i,ht_idx_peaks:] # response after habituation
    max_ys[i] = max_y
    ht_ys[i] = ht_y
    #ht_ys[i] = (max_y-ht_y)/max_y
    ht_s[i] = habituation_time
    #energy[i] = np.sum(np.abs(after_ht)**2)/len(after_ht)
    area[i] = simpson(ys[i], x=t) # integral of response function
  return max_ys, ht_ys, ht_s, area


def get_peak_trough(ydata):
  """
  Detect peaks and troughts for given output trajectory
  peaks[0]: time points where output trajectory takes its local peaks
  peaks[1]: the values of the output trajectory at peaks[0]
  same applies for troughts
  """
  peaks = []
  troughs = []
  peaks_t, _ = find_peaks(ydata, plateau_size=1)
  troughs_t, _= find_peaks(-ydata, plateau_size=1)

  peaks_t_fixed = [i for i in peaks_t if ydata[i]>0.1]
  #troughs_t_fixed = [i for i in troughs_t if -ydata[i]>-0.1]

  peaks.append(peaks_t_fixed)
  peaks.append(ydata[peaks_t_fixed].tolist())
  troughs.append(troughs_t.tolist())
  troughs.append(ydata[troughs_t].tolist())

  return peaks, troughs

def get_cumnum_stimulus(stimulus_t, habiuated_timepoint):
  """
  In: Timepoints of stimulus
  Out: Cumulutive number of stimulus until a time opint (habituated_timepoint)
  """
  num = np.where(stimulus_t>habiuated_timepoint)
  return num[0][0]-1

def get_habituation_time(stimulus_t, ydata, threshold, T, num_dp):
  """
  Compute habituation time as defined in the following thesis 
  https://vcp.med.harvard.edu/papers/thesis-lina-eckert-MSc-ETH.pdf
  """
  peaks, _ = get_peak_trough(ydata)
  peaks_t = np.array(peaks[0])
  peaks_val = np.array(peaks[1])
  #troughs_t = np.array(troughs[0])
  #troughs_val = np.array(troughs[1])
  d = -np.diff(peaks_val)/peaks_val[:-1]
  temp = np.where(d<threshold)
  if len(temp[0]) != 0:
    ht_idx = temp[0][0] # habituation
    ht = get_cumnum_stimulus(stimulus_t, (T/num_dp)*peaks_t[ht_idx])
    return ht, peaks_t, peaks_t[ht_idx]
  else:
    ht=0 #no habituation
    return ht, peaks_t, 0

def get_recovery_time(data):
  # TODO
  return rt


# Given response trajectory,
# compute habituation/recovery features and flags
# Ref. ) https://www.pnas.org/doi/pdf/10.1073/pnas.2409330121 (Hallmark 1)
def check_habituation(y, eps = 0):
    peaks_t, tmp = find_peaks(y)
    habituation_flag1 = False
    habituation_flag2 = True
    habituation_flag = False
    peaks = [y[i] for i in peaks_t]
    for i in range(len(peaks_t)):
        if i < len(peaks_t)-1:
            peak = peaks[i] #y[peaks_t[i]]
            peak_next = peaks[i+1] #y[peaks_t[i+1]]
            if peak_next < peak: # Strict decrement condition
                if i > 0 and habituation_flag1 == False:
                    habituation_flag1 = True
                    dt_idx = i+2
                    dt_time = peaks_t[dt_idx-1]
                    dt_peak = y[dt_time]
            if peak_next > peak + eps: # Weak decrement condition (eps is introduced considering numerical error)
                habituation_flag2 = False
    if habituation_flag1 == True and habituation_flag2 == True:
        habituation_flag = True
    else:
        dt_idx = -1
        dt_time = 0
        dt_peak = 0
    dt_all = [dt_idx, dt_time, dt_peak] # Number of stimuli, time index, peak value
    return [habituation_flag1, habituation_flag2, habituation_flag], dt_all, peaks_t, peaks
