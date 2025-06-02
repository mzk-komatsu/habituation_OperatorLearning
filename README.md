# habituation_OperatorLearning 

Implementation of our paper, 
M. Komatsu, T. Yasui, T. Ohkawa, C. Budd, *"Data-driven modeling of habituation with its frequency-dependent hallmark based on Fourier Neural Operator,"* Nonlinear Theory and Its Applications, IEICE (NOLTA), to appear.

## Environments

This project was executed in the environment described below.

- Ubuntu 20.04.6 LTS

- Python 3.8.20
- PyTorch 2.4.1
- CUDA version 11.8
- cuDNN version 90100
- GPU device: NVIDIA RTX A600

## Project Structure

```
habituation-OperatorLearning/
├── data_generation/  # Directory with data generation scripts
│   ├── main_build_datasets.py  # Run this file for dataset generation
│   ├── dataset_generation.py   
│   ├── input_functions.py       
│   ├── data_utils.py            
│   ├── path.json               # Specify path for "config_global.json"
│   ├── config_global.json      # Global configuration
│   ├── other json files        # Configuration on habituating system,
│   │                           #  "config_[Habituation sys].json" 
│   │                             ([Habituation sys].json is automatically generated)
│   └── data_sample             # Example of datasets
├── OperatorLearning/  # Directory with scripts related to Operator Learning
│   ├── main_hpo_NO.py     # Run this file for HPO of FNO/LNO 
│   ├── main_hpo_donet.py  # Run this file for HPO of DeepONet
│   ├── main_train.py      # Run this file for training FNO/LNO/DeepONet
│   ├── main_pred.py       # Run this file for evaluating FNO/LNO/DeepONet
│   ├── functions.py
│   ├── functions_donet.py
│   ├── models.py          # FNO/LNO is defined here
│   │                   (partially based on External Repository, see below.)
│   ├── hpo_config.py
│   ├── train_settingsN.yaml # Configuration on training (N = 0,1,...)
│   └── pred_settingsM.yaml  # Configuration on evaluation (M = 0,1,...)
├── systems/        # Habituation systems are defined here
├── analysis/       # Analysis 
└── fno/       		# External Repository (See below.)
```



## External Repository Inclusion

- This repository includes the external repository *"fourier_neural_operator"* (https://github.com/wesley-stone/fourier_neural_operator/) by Prof. Zongyi Li, which is licensed under the MIT License. 
- The folder `fno` contains the entire contents of  the repository *"fourier_neural_operator"* as-is.
- `habituation-OperatorLearning/OperatorLearning/models.py` includes the portion of codes of "fourier_neural_operator". 



## Preparation

### Configuration for dataset generation (habituation-OperatorLearning/data_generation)

- `path.json` 
  - Store path for `config_global.json`
- `config_global.json` 
  - **split**: Solve habituating system over [0, T * split] (e.g., 2)
  - **recov_rate**: Rate of the existence of recovery stimuli in (e.g., 0.1),
  - **numi**: # of paramaeters in repetitive stimuli such as F and freq[0] (use 2)
  - **numo**: # of parameters in habituating responses such as Max attenuation rate (use 4)
  - **pred_rate**: The size of datasets used for prediction is pred_rate*num_input_func (e.g., 0.25)
  - **ht_thres**: A parameter used to compute habituation time (e.g., 0.01)
- `config_HabituationSys.json` (HabituationSys : `TysonAL2D`/`NegativeFB`/`GMinimal`/`CMinimal`)
  - **config_name**: identifier of dataset folder (e.g., sample)
  - **T**: Standard time duration [0, 1] (e.g., 1)
  - **num_dp**: # of discretization points in [0, T] (e.g., 2048)
  - **num_input_func**: # of input functions (e.g., 1000)
  - **rate**: Ratio of # of training data, test data (e.g., [0.8, 0.2])
  - **sys**: Habituation model (e.g., "TysonAL2D")
  - **sys_params**: Parameters in Habituation model (e.g., [1, 2, 1, 2, 7])
  - **dim_sys**: # dimension of state variables of Habituation model (e.g., 2)
  - **input_func**: # Type of the input function (square or square_recov or square_mix)
  - **F**: Strength of repetitive stimuli (e.g., 30)
  - **w**: Width of a stimulus (e.g., 60, depends on num_dp)
  - **freq**: Mean (and width) of # of stimuli in [0, T] (e.g., [9.5, 1.5], # of stimuli in [0,T] is (9.5-1.5, 9.5+1.5)
    - freq does not need to be specified manually, as it will be automatically included when `main_build_datasets.py` is executed.   
          

```json
# Example of config_TysonAL2D.json
{
  "config_name": "sample",
  "data_folder_name": "data_sample",
  "T": 1,
  "num_dp": 2048,
  "num_input_func": 1000,
  "rate": [
    0.8,
    0.2
  ],
  "sys": "TysonAL2D",
  "sys_params": [
    1,
    2,
    1,
    2,
    7
  ],
  "dim_sys": 2,
  "input_func": "square",
  "F": 30,
  "w": 60,
  "out_var": 1
}
```



### Configuration for Operator Learning (habituation-OperatorLearning/OperatorLearning)

- `train_settingsN.yaml`

  - **ep**: # of epochs (e.g., 1000)

  - **model_name**: name of deep learning model (e.g., "fno")

  - **sys_name**: name of habituation system (e.g., "TysonAL2D")

  - **sub_int**: log2 of the # of time points in [0, 1] (e.g., 5 ---> 2^5 time points in [0, 1])

  - **modes**: #  of modes in Neural Operator (e.g., 3)

  - **recov**: whether input data set contain repetitive stimuli with recovery stimulus (0: no recovery stimuli, 1: all input functions contain recovery stimuli, 2: some of input functions contain recovery stimuli)

  - **learning_rate** : Initial learning rate (e.g., 0.001)

  - **step_size**: step size (Decays the learning rate of each parameter group every step_size epochs.) (e.g, 50)

  - **gamma**: gamma (Decays the learning rate of each parameter group by gamma every step_size epochs.) (e.g., 0.5)

  - **L**: # of layers in Neural Operator (e.g., 4)

  - **width** : # width of each layer in Neural Operator (e.g., 64)

- `pred_settingsM.yaml` 
  - **recov**: whether input data set contain repetitive stimuli with recovery stimulus (0: no recovery stimuli, 1: all input functions contain recovery stimuli, 2: some of input functions contain recovery stimuli)
  - **ranges_id**: id of ranges to be evaluated (refer to [Habituation_Sys].json in data_generation folder) (e.g., [1,2,3])



## Usage

### Dataset Generation

```
cd data_generation
```

```python
# Generate datasets
python main_build_datasets.py --json config_[Habituation_Sys].json --gjson config_global.json
# E.g.,) 
# python main_build_datasets.py --json config_TysonAL2D.json --gjson config_global.json
```



### Hyper Parameter Optimization 

```
cd OperatorLearning
```

```python
# HPO of Neural Operator (FNO/LNO) 
python main_hpo_NO.py train_settingsN.yaml

# HPO of DeepONet
python main_hpo_donet.py train_settingsN.yaml
```



### Operator Learning (Training)

```
cd OperatorLearning
```

```python
# Training without using results of HPO
python main_train.py train_settingsN.yaml 

# Training without using results of HPO (False is optional)
python main_train.py train_settingsN.yaml False

# Training without using results of HPO (True is required)
python main_train.py train_settingsN.yaml True

```



### Operator Learning (Evaluation)

```
cd OperatorLearning
```

```python
# Evaluation of trained model 
python main_pred.py pred_settingsM.yaml train_settingsN.yaml 

# Evaluation of trained model (False is optional)
python main_pred.py pred_settingsM.yaml train_settingsN.yaml False

# Evaluation of trained model given optimized hyper parameters (True is required)
python main_pred.py pred_settingsM.yaml train_settingsN.yaml True
```



## License

This project is licensed under the MIT License.



## Citation

To be updated soon. 

```bibtex
Mizuka Komatsu, Takatoshi Yasui, Takenao Ohkawa, Chis Budd, ``Data-driven modeling of habituation with its frequency-dependent hallmark based on Fourier Neural Operator'', to appear.
```

