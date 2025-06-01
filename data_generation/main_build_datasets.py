import subprocess
import numpy as np
import json
import tempfile
import os
from datetime import date
from datetime import datetime
import shutil
import sys
import argparse


# ---------------------------------------------------------------------------------
# Execution:
# E.g.)
# python main_build_datasets.py --json config_T.json --gjson config_global.json
# ---------------------------------------------------------------------------------

"""

Content of json file:
    "config_name": identifier of dataset folder (e.g., sample)
    "T": Standard time duration [0, 1] (e.g., 1)
    "num_dp": # of discretization points in [0, T] (e.g., 2048)
    "num_input_func": # of input functions (e.g., 1000)
    "rate": Ratio of # of training data, test data (e.g., [0.8, 0.2])
    "sys": Habituation model (e.g., "TysonAL2D")
    "sys_params": Parameters in Habituation model (e.g., [1, 2, 1, 2, 7])
    "dim_sys": # dimension of state variables of Habituation model (e.g., 2)
    "input_func": # Type of the input function (square or square_recov or square_mix)
    "F": Strength of repetitive stimuli (e.g., 30)
    "w": Width of a stimulus (e.g., 60, depends on num_dp)
    "freq": Mean (and width) of # of stimuli in [0, T] (e.g., [9.5, 1.5], # of stimuli in [0,T] is (9.5-1.5, 9.g+1.5) 
    
Content of gjson file:
  "split": Solve habituating system over [0, T*split] (e.g., 2)
  "recov_rate": Rate of the existence of recovery stimuli in (e.g., ),
  "numi": # of paramaeters in repetitive stimuli such as F and freq[0] (use 2)
  "numo": # of parameters in habituating responses such as Max attenuation rate (use 4)
  "pred_rate": The size of datasets used for prediction is pred_rate*num_input_func (e.g., 0.25)
  "ht_thres": A parameter used to compute habituation time (e.g., 0.01)
}
 
"""

today = date.today()
now = datetime.now().isoformat()

###
# Parameters related to the frequency of repetitive stimuli
freq_lb = 8
freq_ub = 11
freq_global_mean = (freq_lb+freq_ub)/2
freq_global_width = freq_global_mean - freq_lb
freq_step = 1.0
freq_width = freq_step/2
freq_start = freq_lb+freq_width
num_freqs = int((freq_ub-freq_lb)/freq_step)
###

# Load parameters required for data generation
parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default='config.json')
parser.add_argument('--gjson', type=str, default='config_global.json')
args = parser.parse_args()
config_path = args.json
config_global = args.gjson
with open(config_path, 'r', encoding='utf-8') as f:
    JSON = json.load(f)
JSON["freq"] =[freq_global_mean, freq_global_width]
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(JSON, f, ensure_ascii=False, indent=2)

# Load global setting
with open(config_global, 'r', encoding='utf-8') as f:
    gJSON = json.load(f)
pred_rate = gJSON["pred_rate"]
pred_ntrain = int(pred_rate*JSON["num_input_func"]*JSON["rate"][1])
pred_ntest = int(pred_rate*JSON["num_input_func"]*JSON["rate"][0])

# Directory to store data sets   
data_folder_name = "data_"+JSON["config_name"]
os.makedirs(data_folder_name, exist_ok=True)

# Generate data set varying the range of the frequency
ranges = []
# ntrain ntest
num_data = [[int(JSON["num_input_func"]*JSON["rate"][1]), int(JSON["num_input_func"]*JSON["rate"][0])]]
for i in range(num_freqs+1):
    if i != 0:
       center_freq = freq_start + (i-1) * freq_step
       JSON["freq"] = [center_freq, freq_width]
       JSON["num_input_func"] = pred_ntrain + pred_ntest
       num_data.append([pred_ntrain,pred_ntest])
    print(config_path,"\n",JSON)
    # Generate Json file for each data set
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", dir=data_folder_name) as temp_json_file:
        json.dump(JSON, temp_json_file, ensure_ascii=False, indent=2)
        temp_json_file.write("\n")
        temp_json_file_path = temp_json_file.name
    ranges.append(JSON["freq"])
    subprocess.run(['python', 'dataset_generation.py', temp_json_file_path, data_folder_name, config_global])


JSON["ranges"] = ranges
JSON["num_data"] = num_data
JSON["num_input_func"] = num_data[0][0] + num_data[0][1]
JSON["freq"] =[freq_global_mean, freq_global_width]

save_path = os.path.join("./", JSON["sys"]+".json")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(JSON, f, ensure_ascii=False, indent=2)
    f.write("\n")

# Save path for the json file of glocal setting
temp = {
  "global_path": config_global
}
with open('path.json', 'w', encoding='utf-8') as f:
    json.dump(temp, f, ensure_ascii=False, indent=2)