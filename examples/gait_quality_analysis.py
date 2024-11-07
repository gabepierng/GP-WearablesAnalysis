import sys
import os
from itertools import islice
import logging
import datetime
import numpy as np
import pandas as pd
import scipy
import src.comparator_models as cm
from src.dataset_compiler_gq import DatasetCompiler
import src.hmmsm_model as hmmsm_model

run_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")

if not(os.path.isdir('../log_files')):
    os.mkdir('../log_files')

# log script running in log file to monitor training run
logging.basicConfig(filename=f'../log_files/{run_time}_GQA.log',
                    format = "%(asctime)s %(levelname)s %(message)s",
                    level = logging.INFO)

bucket_dir = 'gs://gaitbfb_propellab/'

# participant IDs from two datasets
participants = ['LLA1', 'LLA2', 'LLA3', 'LLA4', 'LLA5', 'LLA6', 'LLA7', 'LLA8', 'LLA9', 'LLA10', 'LLA11', 'LLA12', 'LLA13', 'LLA14', 'LLA15', 'LLA16']
bfb_participants = ['LLPU_P02', 'LLPU_P03', 'LLPU_P05', 'LLPU_P06', 'LLPU_P07', 'LLPU_P08', 'LLPU_P09', 'LLPU_P10', 'LLPU_P14', 'LLPU_P15']

participant_info = pd.read_csv(bucket_dir + 'Gait Quality Analysis/Data/Participant_Data/Raw Data/participant_info.csv')    # file containing main participant info

# experiment parameters
resize_len = 40
strides_to_concat = 10
num_states=2
train_iterations = 1000
train_tolerance = 1e-2
num_models_train = 1
num_models_train_control = 10
num_control_participants = 30

sensors_to_test = 'Lower R+L'

logging.info(f'Gait Quality Analysis testing {sensors_to_test} acc and gyro')
logging.info(f'Strides to concat: {strides_to_concat}   # HMM States: {num_states}   Max Iter.: {train_iterations}    Tol.: {train_tolerance}    Num_HMMs: {num_models_train}    Num_HMM_control: {num_models_train_control}')

b20, a20 = scipy.signal.butter(N=4, Wn = 0.8, btype = 'lowpass')  # Wn = 0.2 = 10 / Nyquist F = 50Hz

signals_of_interest = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

gvs_keys = ['pelvis_x', 'pelvis_y', 'pelvis_z',
               'r_hip_x', 'r_hip_y', 'r_hip_z',
               'l_hip_x', 'l_hip_y', 'l_hip_z',
               'r_knee_flex', 'l_knee_flex',
               'r_ankle_flex', 'l_ankle_flex',
               'r_foot_prog', 'l_foot_prog']

gait_model_results = {'Participant': [],
                        'GPS': [],
                        'HMM-SM': [],
                        'DTW': [],
                        'MDP': [],
                        'INI': [],
                        'MGS': []}

for key in gvs_keys:
    gait_model_results[key] = []

mgs_aspects = [
    np.arange(0,16),
    np.arange(16,21),
    np.arange(21,37),
    np.arange(37,45),
    np.arange(45,50),
    np.arange(50,55)
]


##### COMPILE XSENS DATASET FROM PARTICIPANT DATA #####
# print('Importing participant data...')
target_strides_per_control = 10

dataset = DatasetCompiler(participants, bfb_participants, participant_info, bucket_dir)
dataset.load_participant_data()
dataset.load_control_data()
dataset.sample_control_dataset(target_strides_per_control)
dataset.compile_dataset()

partitioned_strides = {}
trial_types = []
for i, (trial_type, gait_data_set) in enumerate(dataset.partitioned_gait_data.items()):
    trial_types.append(trial_type)

    if(sensors_to_test == 'Upper R+L'):
        partitioned_strides[trial_type] = np.concatenate([gait_data_set.partitioned_movement_data['UpperR_IMU'][:,:,0:], 
                                                        gait_data_set.partitioned_movement_data['UpperL_IMU'][:,:,0:],
                                                        ], axis=2)
    elif(sensors_to_test == 'Lower R+L'):
        partitioned_strides[trial_type] = np.concatenate([gait_data_set.partitioned_movement_data['LowerR_IMU'][:,:,0:], 
                                                        gait_data_set.partitioned_movement_data['LowerL_IMU'][:,:,0:],
                                                        ], axis=2)
    else:
        partitioned_strides[trial_type] = gait_data_set.partitioned_movement_data['Pelvis_IMU'][:,:,0:]
                            
np.random.shuffle(partitioned_strides[trial_type])

control_strides = partitioned_strides['control']
train_som = cm.train_minisom(np.reshape(control_strides, (control_strides.shape[0] * control_strides.shape[1], -1)), 10)

logging.info('Trained MDP control data...')

summarized_control_data_INI, eigen_comps_INI = cm.calculate_eigencomponents_INI(dataset.partitioned_gait_data['control'])
logging.info('Calculated INI control eigenvalues and eigenvectors')

participant_data = [dataset.partitioned_gait_data[i] for i in dataset.partitioned_gait_data if not (i == 'control')]
compiled_mgs_params, reduced_param_set = cm.determine_mgs_reduced_param_set(dataset.partitioned_gait_data['control'], dataset.control_strides_per_part, participant_data)

concat_strides = {}

for trial_type in trial_types:
    concat_strides[trial_type] = []
    for i in range(partitioned_strides[trial_type].shape[0] - strides_to_concat):
        temp = []
        for j in range(strides_to_concat):
            temp.append(partitioned_strides[trial_type][i + j])

        concat_strides[trial_type].append(np.concatenate(temp, axis=0))

    concat_strides[trial_type] = np.array(concat_strides[trial_type])
    concat_strides[trial_type] = scipy.signal.filtfilt(b20, a20, concat_strides[trial_type], axis=1)

###### TRAIN HMMs ON PARTICIPANT DATA ######
hmm_models = {}

for trial_type in trial_types:
    # print('Training %s models...' % (trial_type))
    hmm_models[trial_type] = []
    if(trial_type == 'control'):
        n_train = num_models_train_control
    else:
        n_train = num_models_train

    for j in range(n_train):
        isForward = False
        k = 0

        while not(isForward):
            # print('Train Attempt ', k+1, end="\r", flush=True)
            if(j > -1):
                np.random.shuffle(concat_strides[trial_type])
            strides_sequence_flattened = concat_strides[trial_type].reshape((concat_strides[trial_type].shape[0] * concat_strides[trial_type].shape[1], -1))
            sequence_length = resize_len * strides_to_concat

            hmm_model = hmmsm_model.initiate_hmm(n_components = num_states, n_iter = train_iterations, tolerance = train_tolerance)
            hmmsm_model.train_hmm(hmm_model, strides_sequence_flattened, sequence_length, len(concat_strides[trial_type]))

            isForward = hmmsm_model.check_forward(hmm_model)

        hmm_models[trial_type].append(hmm_model)

    logging.info(f'Trained HMM models for participant {trial_type}')

###### COMPUTE RESULTS FOR HMM-SM AND OTHER GAIT MODELS ######
for i, (trial_type, participant_data) in enumerate(islice(dataset.partitioned_gait_data.items(), 1, None), start=1):

    hmmsm = hmmsm_model.compute_hmmsm(hmm_models['control'], hmm_models[trial_types[i]], num_states)
    dtw = cm.tslearn_dtw_analysis(control_strides, partitioned_strides[trial_type])
    mdp = cm.calculate_mean_MDP(partitioned_strides[trial_type], control_strides, train_som)


    # ini = cm.calculate_INI(summarized_control_data_INI, eigen_comps_INI, dataset.partitioned_gait_data[trial_types[i]])
    ini = cm.calculate_INI(summarized_control_data_INI, eigen_comps_INI, participant_data)

    partial_mgs = cm.calculate_mgs(compiled_mgs_params[:num_control_participants], compiled_mgs_params[i + num_control_participants - 1], reduced_param_set)
    mgs = np.mean(partial_mgs)

    # gvs, gps = cm.calculate_gait_profile_score(dataset.partitioned_gait_data[trial_types[i]].partitioned_movement_data, 
    #                                            dataset.partitioned_gait_data['control'].partitioned_movement_data)
    gvs, gps = cm.calculate_gait_profile_score(participant_data.partitioned_movement_data, 
                                               dataset.partitioned_gait_data['control'].partitioned_movement_data)

    gait_model_results['Participant'].append(trial_type)
    gait_model_results['HMM-SM'].append(hmmsm)
    gait_model_results['DTW'].append(dtw)
    gait_model_results['MDP'].append(mdp)
    gait_model_results['INI'].append(ini)
    gait_model_results['MGS'].append(mgs)
    gait_model_results['GPS'].append(gps)
    for index, key in enumerate(gvs_keys):
        gait_model_results[key].append(gvs[index])

hmmsm_corr, hmmsm_p_val = scipy.stats.spearmanr(gait_model_results['GPS'], gait_model_results['HMM-SM'])
dtw_corr, dtw_p_val = scipy.stats.spearmanr(gait_model_results['GPS'], gait_model_results['DTW'])
mdp_corr, mdp_p_val = scipy.stats.spearmanr(gait_model_results['GPS'], gait_model_results['MDP'])
ini_corr, ini_p_val = scipy.stats.spearmanr(gait_model_results['GPS'], gait_model_results['INI'])
mgs_corr, mgs_p_val = scipy.stats.spearmanr(gait_model_results['GPS'], gait_model_results['MGS'])

logging.info(f'HMM-SM results ~~~ corr.: {round(hmmsm_corr, 4)}\t\tp-val: {round(hmmsm_p_val, 4)}')
logging.info(f'DTW results ~~~ corr.: {round(dtw_corr, 4)}\t\tp-val: {round(dtw_p_val, 4)}')
logging.info(f'MDP results ~~~ corr.: {round(mdp_corr, 4)}\t\tp-val: {round(mdp_p_val, 4)}')
logging.info(f'INI results ~~~ corr.: {round(ini_corr, 4)}\t\tp-val: {round(ini_p_val, 4)}')
logging.info(f'MGS results ~~~ corr.: {round(mgs_corr, 4)}\t\tp-val: {round(mgs_p_val, 4)}')