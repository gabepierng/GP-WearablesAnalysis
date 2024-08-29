from google.cloud import storage
import scipy
from scipy import signal
from scipy.signal import find_peaks
import numpy as np
import os
import sys
import scipy.interpolate as interp
sys.path.append(os.path.join(sys.path[0], '..', 'src'))
import excel_reader_gcp as excel_reader
import gait_metrics as gait_metrics
from gait_metrics import *
import datetime
import logging   
import csv
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import excel_reader_gcp_GN as excel_reader_GN
import copy
import re

b20, a20 = scipy.signal.butter(N=4, Wn = 0.8, btype = 'lowpass')  # Wn = 0.8 = 40 / Nyquist F = 50Hz
run_time = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")

def add_row_to_csv(csv_path, sensor_config, gait_param, algorithm, participant_num, level, parameter):
    header = ['csv_path', 'sensor_config', 'trial_type', 'algorithm', 'participant_num', 'Level', 'Similarity/Deviation']
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writerow(header)
        writer.writerow([csv_path, sensor_config, gait_param, algorithm, participant_num, level, parameter])
        
def reshape_vector(vectors_orig, new_size, num_axes=3):
    x_new = np.linspace(0, 100, new_size)
    trial_reshaped = []
    for stride in vectors_orig:
        x_orig = np.linspace(0, 100, len(stride))
        func_cubic = [interp.interp1d(x_orig, stride[:, i], kind='cubic') for i in range(num_axes)]
        vec_cubic = np.array([func_cubic[i](x_new) for i in range(num_axes)]).transpose()
        trial_reshaped.append(vec_cubic)
    return np.array(trial_reshaped)

XsensGaitParser =  excel_reader_GN.XsensGaitDataParser()
storage_client = storage.Client()
part_strides = {}
part_gait_params = {}
part_kinematic_params = {}
control_strides = {}
control_gait_params = {}
control_kinematic_params = {}

bucket_dir = 'gs://gaitbfb_propellab/'
def compile_gait_data(store_gait_cycles, store_gait_params, store_kin_params, filenames, trial_type_filter, print_filenames=False, look_at_all_files = True, desired_filetypes=None):   

    XsensGaitParser = excel_reader_GN.XsensGaitDataParser()  
    for i, file in enumerate(sorted(filenames)):
        trial_type = re.search(trial_type_filter, file).group(1)
        if(look_at_all_files or any(filetype in file for filetype in desired_filetypes)):
            XsensGaitParser.process_mvn_trial_data(os.path.join(bucket_dir, file))
            partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
            gait_params = XsensGaitParser.get_gait_param_info()

            if trial_type in store_gait_cycles:
                for body_part in store_gait_cycles[trial_type]:
                    for i, side in enumerate(store_gait_cycles[trial_type][body_part]):
                        # for each part (pelvis, l_hip, r_knee, etc.), append strides to appropriate list
                        store_gait_cycles[trial_type][body_part][i] = store_gait_cycles[trial_type][body_part][i] + partitioned_mvn_data[body_part][i]

                store_gait_params[trial_type].append(gait_params['spatio_temp'])

                for joint in store_kin_params[trial_type]:
                    for i, side in enumerate(store_kin_params[trial_type][joint]):
                        store_kin_params[trial_type][joint][i] = np.append(store_kin_params[trial_type][joint][i], gait_params['kinematics'][joint][i], axis=0)

            else:
                store_gait_cycles[trial_type] = partitioned_mvn_data
                store_gait_params[trial_type] = [gait_params['spatio_temp']]
                store_kin_params[trial_type] = gait_params['kinematics']


bucket_name = 'gaitbfb_propellab/'
base_directory = bucket_name + 'Wearable Biofeedback System (REB-0448)/Data/Raw Data'
bucket_name = 'gaitbfb_propellab'
prefix = 'control_dir'
control_dir = 'Gait Quality Analysis/Data/Participant_Data/Processed Data/AbleBodied_Control/CSV'
blobs = storage_client.list_blobs(bucket_name, prefix = control_dir)
control_files = []
for blob in blobs:
    if('.csv' in blob.name):
        control_files.append(blob.name)

compile_gait_data(control_strides, control_gait_params, control_kinematic_params, control_files, 'CSV/(.*?)-00')        

aggregate_control_data = {}
strides_per_control = 10
for i, indiv in enumerate(control_strides.keys()):
    indices = np.arange(len(control_strides[indiv]['gyro_data'][0]))
    np.random.shuffle(indices)
    #control_strides_per_part.append(min(strides_per_control, len(indices)))
    
    if(i == 0):
        aggregate_control_data = control_strides[indiv]
        
        for signal_type in control_strides[indiv]:
            for j, side in enumerate(control_strides[indiv][signal_type]):
                aggregate_control_data[signal_type][j] = [control_strides[indiv][signal_type][j][indices[k]] for k in range(min(strides_per_control, len(indices))) ]             
    else:
        # randomly sample 10 gait cycles from each able-bodied in control, or all gait cycles if less than 10
        for signal_type in control_strides[indiv]:
            for j, side in enumerate(control_strides[indiv][signal_type]):
                aggregate_control_data[signal_type][j] = aggregate_control_data[signal_type][j] + [control_strides[indiv][signal_type][j][indices[k]] 
                                                                                                for k in range(min(strides_per_control, len(indices))) ]

# reshape all the kinematic signals to the size specified for the GPS (51, e.g. 2% increments across the gait cycles from HS to HS)
# store in partitioned_awinda_control
partitioned_awinda_control = {}
partitioned_awinda_control['pelvis_orient'] = reshape_vector(aggregate_control_data['pelvis_orient'][0], new_size = 51)
partitioned_awinda_control['hip_angle'] = [reshape_vector(aggregate_control_data['hip_angle'][0], new_size = 51), reshape_vector(aggregate_control_data['hip_angle'][1], new_size = 51)]
partitioned_awinda_control['knee_angle'] = [reshape_vector(aggregate_control_data['knee_angle'][0], new_size = 51), reshape_vector(aggregate_control_data['knee_angle'][1], new_size = 51)]
partitioned_awinda_control['ankle_angle'] = [reshape_vector(aggregate_control_data['ankle_angle'][0], new_size = 51), reshape_vector(aggregate_control_data['ankle_angle'][1], new_size = 51)]


script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
run_time = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")
csv_filename = f"Eval_logresults_{run_time}.csv"
csv_path = os.path.join(logs_dir, csv_filename)
log_file = os.path.join(logs_dir, f"logresults_{run_time}.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

#Stuff for HMM-SM 
def calculate_state_correspondence_matrix(hmm_1, hmm_2, n_states):
    def calculate_stationary_distribution(hmm):
        eigenvals, eigenvectors = np.linalg.eig(hmm.model.transmat_.T)
        stationary = np.array(eigenvectors[:, np.where(np.abs(eigenvals - 1.) < 1e-8)[0][0]])
        stationary = stationary / np.sum(stationary)
        return np.expand_dims(stationary.real, axis=-1)

    # KL-Divergence = method for determining the difference between two probability distributions
    def calculate_KL_div(hmm_model_1, hmm_model_2, state_model_1, state_model_2):
        means_1 = np.expand_dims(hmm_model_1.means_[state_model_1], axis=-1)
        means_2 = np.expand_dims(hmm_model_2.means_[state_model_2], axis=-1)

        covars_1 = hmm_model_1.covars_[state_model_1]
        covars_2 = hmm_model_2.covars_[state_model_2]

        term_1 = (means_2 - means_1).T @ np.linalg.inv(covars_2) @ (means_2 - means_1)
        term_2 = np.trace(np.linalg.inv(covars_2) @ covars_1)
        term_3 = np.log(np.linalg.det(covars_1) / np.linalg.det(covars_2))
        term_4 = len(covars_1)

        kl_divergence = 0.5 * (term_1 + term_2 - term_3 - term_4)

        return kl_divergence

    kl_state_comparisons = np.zeros((n_states, n_states))
    pi_1 = calculate_stationary_distribution(hmm_1)
    pi_2 = calculate_stationary_distribution(hmm_2)
    total_expected_similarity = 0

    for i in range(n_states):
        for j in range(n_states):
            kl_state_comparisons[i,j] = 0.5 * (calculate_KL_div(hmm_1.model, hmm_2.model, i, j) + calculate_KL_div(hmm_2.model, hmm_1.model, i, j))
            total_expected_similarity = total_expected_similarity + (pi_1[i] * pi_2[j] * kl_state_comparisons[i,j])

    k = 1

    # alternative methods of calculating similarity based on KL-Divergence
    # s_e = np.exp(-k * kl_state_comparisons)
    s_e = 1 / kl_state_comparisons

    # pi_1.T @ pi_2 should produce a N x N matrix (pi_1i * pi_2j)
    q_matrix = ((pi_1 @ pi_2.T) * s_e) / total_expected_similarity

    return q_matrix

def calculate_gini_index(q_matrix, n_states):
    def calc_gini(vector):
        vector = np.sort(vector)
        l1_norm = np.linalg.norm(vector, 1)
        a = 0
        for i in range(1, n_states+1):
            a = a + (vector[i-1] / l1_norm) * ((n_states - i + 0.5) / (n_states - 1))

        vec_sparsity = (n_states / (n_states - 1)) - (2 * a)

        return vec_sparsity

    # calculate mean sparsity of each row vector (r) and column vector (c) in Q matrix
    r = (1 / n_states) * np.sum([calc_gini(row) for row in q_matrix])
    c = (1 / n_states) * np.sum([calc_gini(column) for column in q_matrix.T])

    gini_index = 0.5 * (r + c)
    return gini_index

"""Functions for reshaping/assembling raw sensor data together - option to normalize the data (standardizes)"""
def normalize_signal(signal):
    mean = np.mean(signal, axis=1, keepdims=True)
    std_dev = np.std(signal, axis=1, ddof=1, keepdims=True)
    normalized_signal = (signal - mean) / std_dev
    return normalized_signal

#uses dictionaries to extract the relevant raw sensor data, reshapes the data, then concatenates gyro and accelerometer signals together 
def organize_signals(sensor_mappings, gyro_signal, accel_signal,normalize=True): #Default = Normalization on
    combined_signals = {}
    for location, sensor in sensor_mappings.items():
        reshaped_gyro = reshape_vector(gyro_signal[sensor], 40, 3)
        reshaped_accel = reshape_vector(accel_signal[sensor], 40, 3)
        
        """ NORMALIZATION ADDED HERE"""
        if normalize:
            normalized_gyro = normalize_signal(reshaped_gyro) #Normalizing the reshaped signals
            normalized_accel = normalize_signal(reshaped_accel) #Normalizing the reshaped signals
            combined_signals[location] = np.concatenate((normalized_gyro, normalized_accel), axis=-1) #Concatenates to gyro x,y,z and accel x,y,z

        else:
            combined_signals[location] = np.concatenate((reshaped_gyro, reshaped_accel), axis=-1) #Concatenates to gyro x,y,z and accel x,y,z, without normalization 
    
    return combined_signals

""" Data grouping/splitting pipeline:"""
""" Split the data into desired number of groups, and extract the first "target mean" from the first group. Subsequent target means are calculated as X% (percent grading) away from previous groups
    ex. if 3% is selected, means of groups will increment by 3% - the direction of incrementing (up/down) depends on whether reverse is selected 
    Iterate through indices in the gait parameter data (in this case, STSR) and append to a grouping if they are within a threshold from the target mean. 
    Sort both the gait parameter and corresponding gait cycles based on the chosen indices. 
"""

"""
num_groups (float) - This sets how many groups the code will initially try to split the data by in order to create "target means". This number can be toggled with depending on the data distribution.
gait_parameter (list) - List of gait parameter values corresponding to the gait cycles (ex. STSR, GPS) - parameter you want to SPLIT BY
gait_cycles (dict) - Dictionary with the corresponding sensor location key for the raw sensor data. For each sensor location, stores a list of arrays, where each array corresponds to the raw sensor data for a gait cycle (ex. 40x6)
grading_value (float) - Sets what increment between groups that you want to set.
add_param (list): Additional parameter that you don't want to split the data by, but can be used to just check and see what the distribution of some other secondary gait parameter.

"""

def finding_groupings(num_groups, gait_parameter, gait_cycles, grading_value, add_param, reverse=True):
    
    if reverse:
        grading_value = -grading_value
        values_sorted = sorted(gait_parameter, reverse=True)
        sorted_indices = np.argsort(gait_parameter)[::-1]  # Sort indices in descending order of stance time symmetry/other sorting parameter
    else:
        sorted_indices = np.argsort(gait_parameter)
        values_sorted = sorted(gait_parameter, reverse=False)

    n = len(gait_parameter)
    group_sizes = [n // num_groups + (1 if i < n % num_groups else 0) for i in range(num_groups)]
    target_means = [np.mean(values_sorted[:group_sizes[0]])]

    # Initializes the groups and the remaining values to be picked from
    grouped_gait_cycles = {}
    groups = [[] for _ in range(num_groups)]
    add_params = [[] for _ in range(num_groups)]
    grouped_gait_cycles = {key: [[] for _ in range(num_groups)] for key in gait_cycles}
    remaining_indices = sorted_indices[:]

    for i in range(1, num_groups):
        target_means.append(target_means[0] + grading_value * i)

    for i in range(num_groups):
        target_mean = target_means[i]
        filtered_indices = [idx for idx in remaining_indices if abs(gait_parameter[idx] - target_mean) < grading_value / 2]
        selected_indices = filtered_indices[:]
        groups[i].extend(gait_parameter[idx] for idx in selected_indices)
        add_params[i].extend(add_param[idx] for idx in selected_indices)
        
        for key in gait_cycles: #Appends the correct gait cycles to the new list
            grouped_gait_cycles[key][i].extend(gait_cycles[key][idx] for idx in selected_indices)
        
        remaining_indices = [idx for idx in remaining_indices if idx not in selected_indices]

    return groups, grouped_gait_cycles, grading_value, add_params            

""" Random Sampling Gait Cycles:"""

""" random_sampling: Randomly sample 50 gait cycles from group 1. This serves as the "first_mean" that will be compared to in adaptive subsampling.
    adaptive_subsample: Randomly sample 50 gait cycles from a given group. If it is within X% * i +/- tolerance (i is the index of the group (ex. group 2 = 1)), then that group is accepted.
    Otherwise, the maximum or minimum is removed and another value still available is added (depending on if the current percent difference is too high or too low)
    Handles if the means are decreasing or increasing (if percent_diff is negative or positive)
    Returns the indices of the groups, and these are used to update the new groups in random_sampling.
"""

def random_sampling(groups, grouped_gait_cycles, add_params, grading_value, tolerance, sample_size=50):
    
    def adaptive_subsample(group, first_mean, i, sample_size=50, max_iterations=10000):
        available_indices = list(range(len(group)))  # Make a list that spans all the indices
        sample_indices = np.random.choice(available_indices, size=sample_size, replace=False)
        available_indices = [idx for idx in available_indices if idx not in sample_indices]  # Remove initial sample values from available values

        for _ in range(max_iterations):
            current_mean = np.mean([group[idx] for idx in sample_indices])
            percent_diff = current_mean - first_mean 
            target_diff = grading_value * i

            if len(available_indices) == 0:
                raise ValueError("No candidates available to adjust the mean")

            if (target_diff - tolerance) <= abs(percent_diff) <= (target_diff + tolerance):
                return sample_indices

            if abs(percent_diff) < (target_diff - tolerance):
                direction = 'lower' if percent_diff < 0 else 'upper'
            else:
                direction = 'upper' if percent_diff > 0 else 'lower'

            if direction == 'lower':
                candidates = [idx for idx in available_indices if group[idx] <= np.percentile(group, 50)]
            else:
                candidates = [idx for idx in available_indices if group[idx] >= np.percentile(group, 50)]

            if not candidates:
                candidates = available_indices
                
            new_idx = np.random.choice(candidates)
            available_indices.remove(new_idx)
            
            if direction == 'lower':
                sample_indices = np.append(sample_indices, new_idx)
                sample_indices = np.delete(sample_indices, np.argmax([group[idx] for idx in sample_indices]))
            else:
                sample_indices = np.append(sample_indices, new_idx)
                sample_indices = np.delete(sample_indices, np.argmin([group[idx] for idx in sample_indices]))
                
    
    indices_first_group = np.random.choice(range(len(groups[0])), size=sample_size*2, replace=False)
    random.shuffle(indices_first_group)
    baseline_1_indices, baseline_2_indices = indices_first_group[:50], indices_first_group[50:]

    subsampled_values = {
        'baseline1': [groups[0][j] for j in baseline_1_indices],
        'baseline2': [groups[0][j] for j in baseline_2_indices]
    }

    add_params_subsampled = {
        'baseline1': [add_params[0][j] for j in baseline_1_indices],
        'baseline2': [add_params[0][j] for j in baseline_2_indices]
    }

    gaitcycles_subsampled = {
        key: {
            'baseline1': [grouped_gait_cycles[key][0][j] for j in baseline_1_indices],
            'baseline2': [grouped_gait_cycles[key][0][j] for j in baseline_2_indices]
        }
        for key in grouped_gait_cycles
    }

    groups_subsampled_list = [subsampled_values['baseline1'], subsampled_values['baseline2']]
    add_params_subsampled_list = [add_params_subsampled['baseline1'], add_params_subsampled['baseline2']]
    gaitcycles_subsampled_list = {key: [gaitcycles_subsampled[key]['baseline1'], gaitcycles_subsampled[key]['baseline2']] for key in grouped_gait_cycles}

    group1_mean = np.mean(groups_subsampled_list[0])  # first mean used as the target for all subsequent groups

    for i in range(1, 3): #This ensures there are 4 groups (BL, L0, L1, L2)
        sample_indices = adaptive_subsample(np.array(groups[i]), group1_mean, i)
        groups_subsampled_list.append([groups[i][j] for j in sample_indices])
        add_params_subsampled_list.append([add_params[i][j] for j in sample_indices])
        
        for key in grouped_gait_cycles:
            gaitcycles_subsampled_list[key].append([grouped_gait_cycles[key][i][j] for j in sample_indices])
    
   
    return groups_subsampled_list, gaitcycles_subsampled_list, add_params_subsampled_list

""" Group splitting and sampling are called. Checks to see which direction the grouping should be done in, and only appends the groups that have at least 70 points
"""

def check_group_configurations(gait_split_parameter, raw_sensor_data, num_groups, add_param, grading_value, tolerance):
    
    def filter_groups(groups, grouped_gait_cycles, add_params):
        filtered_groups = []
        filtered_add_param = []
        filtered_gait_groups = {key: [] for key in grouped_gait_cycles}
        
        for i in range(len(groups)):
            if len(groups[i]) >= 70:
                filtered_groups.append(groups[i])
                filtered_add_param.append(add_params[i])
                for key in grouped_gait_cycles:
                    filtered_gait_groups[key].append(grouped_gait_cycles[key][i])
        
        return filtered_groups, filtered_add_param, filtered_gait_groups

    groups, grouped_gait_cycles, grading, add_params = finding_groupings(
        num_groups, gait_split_parameter, raw_sensor_data, grading_value, add_param, reverse=False
    )
    
    filtered_groups, filtered_add_param, filtered_gait_groups = filter_groups(groups, grouped_gait_cycles, add_params)

    if len(filtered_groups) < 3:
        print("Trying the reverse")
        groups, grouped_gait_cycles, grading, add_params = finding_groupings(
            num_groups, gait_split_parameter, raw_sensor_data, grading_value, add_param, reverse=True
        )
        filtered_groups, filtered_add_param, filtered_gait_groups = filter_groups(groups, grouped_gait_cycles, add_params)

        if len(filtered_groups) < 3:
            raise ValueError("Insufficient group sizes available for this participant")
    
    groups, gaitcycles, add_parameter = random_sampling(filtered_groups, filtered_gait_groups, filtered_add_param, grading_value, tolerance)
    return groups, gaitcycles, add_parameter

def get_participant_info(participant_id):
    participant = participant_info[participant_info['Participant_ID'] == participant_id]
    if not participant.empty:
        height = participant.iloc[0]['Height']
        side = int(participant.iloc[0]['Side'])
        return height, side
    else:
        return None, None

##################################################################################################################################################################################

"""Main section of code for processing participants"""
#Dictionary to map the sensor locations to their IDs.
sensor_mappings = {
    'pelvis': 1,
    'UpperR': 2,
    'LowerR': 3,
    'UpperL': 5,
    'LowerL': 6
}

XsensGaitParser =  excel_reader.XsensGaitDataParser()
storage_client = storage.Client()
bucket_name = 'gaitbfb_propellab/'
base_directory = bucket_name + 'Wearable Biofeedback System (REB-0448)/Data/Raw Data'
bucket_name = 'gaitbfb_propellab'
blobs = storage_client.list_blobs(bucket_name, prefix = base_directory)
prefix_from_bucket = 'Wearable Biofeedback System (REB-0448)/Data/Raw Data/' 

participant_list = ['LLPU_P01','LLPU_P02','LLPU_P03','LLPU_P04','LLPU_P05','LLPU_P06','LLPU_P08','LLPU_P09','LLPU_P10','LLPU_P12','LLPU_P15']
participant_info = pd.read_excel("Q:\\main_propellab\\Users\\Ng, Gabe\\Summer Student 2024\\LLA Participant Investigations\\LLPU_Height_ProstheticSide.xlsx") #Used to determine heights and prosthetic sides as needed.


for participant in participant_list:
    
    print(f"Processing participant {participant}")
    participant_id = participant
    height,side = get_participant_info(participant_id)
    height_m = height/100 
    
    if side == 1:
        prosth_side = 0 #Right side
        side_label = "Right"
        print("Prosthetic on the right side")
    else:
        prosth_side = 1 #Left side
        side_label = "Left"
        print("Prosthetic side on the left side")
    
    if height is None or side is None:
        print(f"Participant info not found for ID: {participant_id}")
        continue
    
    directory = prefix_from_bucket + participant + '/Excel_Data_Trimmed'
    blobs = storage_client.list_blobs(bucket_or_name=bucket_name, prefix=directory.replace("\\", "/"))
    part_strides = {}
    part_gait_params = {}
    part_kinematic_params = {}
    part_sensor_data = {}
    trial_type = 'GPS' #Use this to set whatever parameter you're looking to partition by 

    logging.info(f"Processing participant {participant}")
    if blobs:
        for blob in blobs:
            if blob.name.endswith('.csv'):
                try:
                    XsensGaitParser.process_mvn_trial_data(f"gs://{bucket_name}/{blob.name}")
                    partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
                    gait_params = XsensGaitParser.get_gait_param_info()
                    normalize = True
                    combined_signals = organize_signals(sensor_mappings, partitioned_mvn_data['gyro_data'], partitioned_mvn_data['acc_data'],normalize=normalize)
                    pelvis_data = combined_signals['pelvis']
                    upper_data = np.concatenate((combined_signals['UpperR'], combined_signals['UpperL']), axis=2)  # Concatenate by last axis
                    lower_data = np.concatenate((combined_signals['LowerR'], combined_signals['LowerL']), axis=2)  # Concatenate by last axis
                    
                    if trial_type in part_strides:
                        for body_part in part_strides[trial_type]:
                            for i, side in enumerate(part_strides[trial_type][body_part]):
                                # for each part (pelvis, l_hip, r_knee, etc.), append strides to appropriate list
                                part_strides[trial_type][body_part][i] = part_strides[trial_type][body_part][i] + partitioned_mvn_data[body_part][i]
                               
                        part_gait_params[trial_type].append(gait_params['spatio_temp'])
                        part_sensor_data['pelvis'] = np.append(part_sensor_data['pelvis'], pelvis_data,axis=0)
                        part_sensor_data['Upper'] = np.append(part_sensor_data['Upper'], upper_data,axis=0)
                        part_sensor_data['Lower'] = np.append(part_sensor_data['Lower'], lower_data,axis=0)
                    

                        for joint in part_kinematic_params[trial_type]:
                            for i, side in enumerate(part_kinematic_params[trial_type][joint]):
                                part_kinematic_params[trial_type][joint][i] = np.append(part_kinematic_params[trial_type][joint][i], gait_params['kinematics'][joint][i], axis=0) 

                    else:
                        part_strides[trial_type] = partitioned_mvn_data
                        part_gait_params[trial_type] = [gait_params['spatio_temp']]
                        part_kinematic_params[trial_type] = gait_params['kinematics']
                        part_sensor_data['pelvis'] = pelvis_data
                        part_sensor_data['Upper'] =  upper_data
                        part_sensor_data['Lower'] = lower_data
                    
                    file_name = os.path.basename(blob.name)

                except IndexError as e: #Exception based on an Index Error encountered in excel_reader_gcp.py **
                    continue                              
    
    if trial_type in part_gait_params:
        
        if prosth_side == 1: #Prosthetic on the left, reverses the calculation
            stance_time_symmetry = [1/item for sublist in [i[11] for i in part_gait_params[trial_type]] for item in sublist] # STSR = prosth/non_prosh (the convention is R/L, so this corrects for the calculation if the prosthetic is on the L)
        else:
            stance_time_symmetry = [item for sublist in [i[11] for i in part_gait_params[trial_type]] for item in sublist]
        
        partitioned_awinda_gait = {}
        partitioned_awinda_gait['pelvis_orient'] = reshape_vector(part_strides[trial_type]['pelvis_orient'][0], new_size = 51)
        partitioned_awinda_gait['hip_angle'] = [reshape_vector(part_strides[trial_type]['hip_angle'][0], new_size = 51), reshape_vector(part_strides[trial_type]['hip_angle'][1], new_size = 51)]
        partitioned_awinda_gait['knee_angle'] = [reshape_vector(part_strides[trial_type]['knee_angle'][0], new_size = 51), reshape_vector(part_strides[trial_type]['knee_angle'][1], new_size = 51)]
        partitioned_awinda_gait['ankle_angle'] = [reshape_vector(part_strides[trial_type]['ankle_angle'][0], new_size = 51), reshape_vector(part_strides[trial_type]['ankle_angle'][1], new_size = 51)]

        # Extract and reshape individual signals
        individual_signals = []
        gait_scores_list = []
        gait_variability = []

        for i in range(partitioned_awinda_gait['pelvis_orient'].shape[0]):
            signal_dict = {
                'pelvis_orient': partitioned_awinda_gait['pelvis_orient'][i].reshape(1, 51, 3),
                'hip_angle': [partitioned_awinda_gait['hip_angle'][0][i].reshape(1, 51, 3), partitioned_awinda_gait['hip_angle'][1][i].reshape(1, 51, 3)],
                'knee_angle': [partitioned_awinda_gait['knee_angle'][0][i].reshape(1, 51, 3), partitioned_awinda_gait['knee_angle'][1][i].reshape(1, 51, 3)],
                'ankle_angle': [partitioned_awinda_gait['ankle_angle'][0][i].reshape(1, 51, 3), partitioned_awinda_gait['ankle_angle'][1][i].reshape(1, 51, 3)]
            }
            individual_signals.append(signal_dict)


        for signal_val in individual_signals:
            gaitprofilescore, gaitvariabilityscore = calc_gait_profile_score(signal_val, partitioned_awinda_control)
            gait_scores_list.append(gaitprofilescore) #GPS
            gait_variability.append(gaitvariabilityscore) #GVS -- used this for additional analyses to determine how individual components of the GPS are different between different groups
        
        """ Use this line if wanting to partition by STSR """
        
        if trial_type == 'STSR':
            if participant == 'LLPU_P08':
                ordered_groups, ordered_gaitcycles, ordered_addparam = check_group_configurations(stance_time_symmetry, part_sensor_data,13, gait_scores_list, 0.03, 0.005)
            else:
                ordered_groups, ordered_gaitcycles, ordered_addparam = check_group_configurations(stance_time_symmetry, part_sensor_data,4, gait_scores_list, 0.03, 0.005)
        if trial_type == "GPS":
            ordered_groups, ordered_gaitcycles, ordered_addparam = check_group_configurations(gait_scores_list, part_sensor_data, 3, gait_scores_list, 0.4, 0.05)
        
        pre_gvs_dict = {}
        post_gvs_dict = {}
        
        #Old code when looking at GVS comparisons 

        # L0_params = ordered_addparam[1] #second group
        # L2_params = ordered_addparam[-1] #last group

        # for gait_variability_scores in L0_params:
        #     for signal, scores in gait_variability_scores.items():
        #         if signal not in pre_gvs_dict:
        #             pre_gvs_dict[signal] = []
        #         # Ensure scores is iterable (e.g., convert single values to a list)
        #         if isinstance(scores, (list, np.ndarray)):
        #             pre_gvs_dict[signal].extend(scores)
        #         else:
        #             pre_gvs_dict[signal].append(scores)
        #     # Add participant column
        #     participant_list = [participant_id] * len(L0)
        #     pre_gvs_dict.setdefault('Participant', participant_list.copy())

        # for gait_variability_scores in L2_params:
        #     for signal, scores in gait_variability_scores.items():
        #         if signal not in post_gvs_dict:
        #             post_gvs_dict[signal] = []
        #         # Ensure scores is iterable (e.g., convert single values to a list)
        #         if isinstance(scores, (list, np.ndarray)):
        #             post_gvs_dict[signal].extend(scores)
        #         else:
        #             post_gvs_dict[signal].append(scores)
        #     # Add participant column
        #     post_gvs_dict.setdefault('Participant', participant_list.copy())

        # # Convert dictionaries to DataFrames
        # pre_gvs_df = pd.DataFrame(pre_gvs_dict)
        # post_gvs_df = pd.DataFrame(post_gvs_dict)

        # # Add a column to distinguish between pre and post conditions
        # pre_gvs_df['Condition'] = 'Pre'
        # post_gvs_df['Condition'] = 'Post'

        # # Combine Pre and Post DataFrames
        # combined_gvs_df = pd.concat([pre_gvs_df, post_gvs_df], ignore_index=True)

        # # Reorder columns to place 'Participant' as the leftmost column
        # columns_order = ['Participant'] + [col for col in combined_gvs_df.columns if col != 'Participant']
        # combined_gvs_df = combined_gvs_df[columns_order]

        # file_path = r'Q:\main_propellab\Users\Ng, Gabe\Summer Student 2024\Manuscript\RAS_PrePostGVS.csv'
        # if not os.path.isfile(file_path):
        #     combined_gvs_df.to_csv(file_path, index=False, header=True)
        # else:
        #     combined_gvs_df.to_csv(file_path, index=False, header=False, mode='a')
        
        logging.info(f"{trial_type} Trial")
        ordered_group_means = [round(np.mean(group), 3) for group in ordered_groups]
        ordered_group_min = [round(min(group), 3) for group in ordered_groups]
        ordered_group_max = [round(max(group), 3) for group in ordered_groups]
        group_lengths = [len(group) for group in ordered_groups]

        for k, group in enumerate(ordered_groups):
            print(f"Group {k+1}: {len(group)} (Mean: {np.mean(group) if group else 'N/A'})")
        
        # Pelvis, upper, lower
        combined_sensor_configs = [ordered_gaitcycles['pelvis'], ordered_gaitcycles['Upper'], ordered_gaitcycles['Lower']]   
        arrangements = ['pelvis','upper','lower']  
        for sensor_idx, raw_sensor in enumerate(combined_sensor_configs): #Iterates through each sensor configuration (pelvis, upper, lower)
            
            """ DTW Implementation"""
            print(f"Sensor arrangement: {arrangements[sensor_idx]}")
            dtw_mean_distances = []
            #Computing the within group distance for baseline
            # dtw_within = tslearn_dtw_analysis(set1 = raw_sensor[0], set2=None) # type: ignore
            # dtw_mean_distances.append(dtw_within)
            # add_row_to_csv(csv_path, arrangements[sensor_idx],'GPS', 'DTW',participant, ordered_group_means[0], dtw_within)
            
            for j in range(1, len(raw_sensor)):
                dtw_between = tslearn_dtw_analysis(set1 = raw_sensor[0], set2 = raw_sensor[j]) # type: ignore
                dtw_mean_distances.append(dtw_between)
                #add_row_to_csv(csv_path, arrangements[sensor_idx],'STSR', 'DTW',participant, ordered_group_means[j], dtw_between)
                
            print(dtw_mean_distances) 
            
            """ SOM Implementation"""
            # Shuffle and split the list
            #train_arrays, test_arrays = train_test_split(raw_sensor[0], test_size=0.2, random_state=42)
            random.shuffle(raw_sensor[0])
            train_data = np.concatenate(raw_sensor[0],axis=0) #Train the SOM on BL data 
            print("Training Data Shape:", train_data.shape)
            print("Training the SOM on baseline data")
            MDP_mean_deviations = []
            trained_SOM = train_minisom(train_data, learning_rate=0.1, topology='hexagonal', normalize=True) # type: ignore
           
            for j in range(1, len(raw_sensor)):
                #Shuffle and split the list (only looking at the test data here)
                random.shuffle(raw_sensor[j])
                test_data = np.concatenate(raw_sensor[j],axis=0)
                MDP_scores = calculate_MDP(test_data, train_data, trained_SOM, normalize=True) # type: ignore
                MDP_mean_deviations.append(np.mean(MDP_scores))
                #add_row_to_csv(csv_path, arrangements[sensor_idx],'STSR','MDP',participant, ordered_group_means[j], np.mean(MDP_scores))
            
            print(f"MDP mean deviations: {MDP_mean_deviations}")
            
            """HMM Implementation"""
            strides_train_flat = {}
            strides_test_flat = {}
            strides_train = {}
            strides_test = {}
            resize_len = 40
            strides_to_concat = 10
            num_states= 3 #Changed from 5 to 2 #Test 3 as well 
            train_iterations = 300 
            train_tolerance = 1e-2
            num_models_train = 10
            concat_strides = {}
   
            for idx, group in enumerate(raw_sensor):
                concat_strides[idx] = []
                group = np.array(group)
                for i in range(group.shape[0] - strides_to_concat):
                    temp = []
                    for j in range(strides_to_concat):
                        temp.append(group[i + j])
                    concat_strides[idx].append(np.concatenate(temp, axis=0))

                concat_strides[idx] = np.array(concat_strides[idx])
                concat_strides[idx] = signal.filtfilt(b20, a20, concat_strides[idx], axis=1)
                
            hmm_models = {}  
                
            for idx, group in enumerate(raw_sensor):
                
                num_models_training = num_models_train #Re initialize the number of models for training 
                hmm_models[idx] = []
                
                for j in range(num_models_training):
                    train_forward_model = True
                    k = 0
                    
                    while(train_forward_model):
                        print('Train Attempt ', k+1, end="\r", flush=True)
                        
                        if(j > -1):
                            np.random.shuffle(concat_strides[idx])
                            
                        # flatten sequence for hmmlearn train function
                        strides_sequence_flattened = concat_strides[idx].reshape((concat_strides[idx].shape[0] * concat_strides[idx].shape[1], -1))
                        
                        # technically is no training/testing data, but this preserves a few gait cycles to compare the hidden-state sequence predictions of the HMMs
                        len_train = int(0.95 * len(concat_strides[idx]))
                        strides_train[idx] = concat_strides[idx][:len_train]
                        strides_test[idx] = concat_strides[idx][len_train:]
                        sequence_length = resize_len * strides_to_concat
                        strides_train_flat[idx] = strides_sequence_flattened[:sequence_length * len_train]
                        strides_test_flat[idx] = strides_sequence_flattened[sequence_length * len_train:]

                        hmm_model = HMMTrainer(n_components = num_states, n_iter = train_iterations, tolerance = train_tolerance)
                        hmm_model.train(strides_train_flat[idx], sequence_length, len_train)


                        # double checks for left-to-right architecture in transition matrix
                        valid_rows = 0
                        a_mat = hmm_model.model.transmat_
                        for i, row in enumerate(a_mat):
                            temp = np.argpartition(np.roll(row, -i), -2)[-2:]
                            if((np.array(temp) == np.array([0,1])).all() or (np.array(temp) == np.array([1,0])).all()):
                                valid_rows = valid_rows + 1

                        # correct_second_state = [i for i in range(num_states - 1)]
                        # correct_second_state.append(0)        
                        # for i, row in enumerate(hmm_model.model.transmat_):
                        #     max_state = np.argmax(row)
                        #     if(max_state == i):
                        #         temp = [j for j in row if not (j == row[max_state])]
                        #         if(np.argmax(temp) == correct_second_state[i]):
                        #             valid_rows = valid_rows + 1

                        #if model is left-to-right, consider model trained, train next model (until num_models_train reached)
                        
                        if(valid_rows == num_states):
                            train_forward_model = False
                        k = k + 1
                    hmm_models[idx].append(hmm_model)

            print('done')
            
            test_predict = strides_test[0][1] #2nd element of idx = 0
            min_predict = np.min(test_predict[:,1])
            max_predict = np.max(test_predict[:,1])
            
            
            def align_states(trained_hmm_model, roll_amount=0):
                new_hmm = copy.deepcopy(trained_hmm_model)
                array_order = np.roll(np.arange(num_states), roll_amount)

                new_hmm.model.transmat_ = new_hmm.model.transmat_[array_order,: ]
                for i, row in enumerate(new_hmm.model.transmat_):
                    new_hmm.model.transmat_[i] = np.roll(new_hmm.model.transmat_[i], roll_amount)

                new_hmm.model.means_ = new_hmm.model.means_[array_order, :]
                new_hmm.model.covars_ = new_hmm.model.covars_[array_order, :]
                new_hmm.model.startprob_ = new_hmm.model.startprob_[array_order]
                return new_hmm 

            pred_vals = np.ones(num_states)
            
            for state in range(num_states):
                pred_vals[state] = min_predict + ((state * (max_predict - min_predict)) / (num_states - 1))
        
            roll_amounts = {}
            match_trials = {}

            for idx, group in enumerate(raw_sensor):
                roll_amounts[idx] = [0 for i in range(num_models_train)]
                match_trials[idx] = 0

            shift_all = 0
        
            def find_best_alignment(hmm_1, hmm_2, test_stride, n_states):
                min_distance = 9999999
                best_roll = 0

                for j in range(n_states):
                    new_hmm = align_states(hmm_2, j)
                    prediction_1 = hmm_1.model.predict(test_stride)
                    prediction_2 = new_hmm.model.predict(test_stride)

                    distance = np.sum((prediction_1 - prediction_2) ** 2)
                    if (distance < min_distance):
                        min_distance = distance
                        best_roll = j

                return best_roll

            predictions = {}
            hmm_models_aligned_states = {}
            
            for idx, group in enumerate(raw_sensor):
                
                predictions[idx] = []
                hmm_models_aligned_states[idx] = []
                match_trials[idx] = find_best_alignment(hmm_models[0][0], hmm_models[idx][0], test_predict, num_states)
                roll_amounts[idx] = [0] * len(hmm_models[idx]) #Can now handle different lengths of trained models 
                
                for j in range(len(hmm_models[idx])):
                    roll_amounts[idx][j] = find_best_alignment(hmm_models[idx][0], hmm_models[idx][j], test_predict, num_states) + match_trials[idx]
                    # roll_amounts[idx][j] = 0
                    hmm_models_aligned_states[idx].append(align_states(hmm_models[idx][j], roll_amounts[idx][j] + shift_all))
                    predictions[idx].append(hmm_models_aligned_states[idx][-1].model.predict(test_predict))
                        
            
                #bunch of stuff for visualizing the hidden-state sequence predictions

                #print(roll_amounts[trial_types[1]][0])
                # fig, ax = plt.subplots()
                # fig.set_size_inches(12,8)
                # plt.plot(test_predict[:,0])

                # plt.plot([pred_vals[j] for j in predictions[0]][0], 'k')
                # plt.plot([pred_vals[j] for j in predictions[0]][4], 'r')

                # plt.plot([pred_vals[j] for j in predictions_post[4]], 'k')
                # plt.plot([pred_vals[j] for j in predictions_post[2]], 'r')

                # def trial_avg_and_CI(signal_set):
                #     conf_int_mult = 1.00    # confidence interval multiplier for 1 std

                #     avg_signal = np.mean(signal_set, axis=0)
                #     std_signal = np.std(signal_set, axis=0)
                #     upper_bound = avg_signal + (conf_int_mult * std_signal)
                #     lower_bound = avg_signal - (conf_int_mult * std_signal)

                #     return avg_signal, upper_bound, lower_bound

                # def confidence_plot(plot_signals, fig_ax, trial_num):
                #     plot_signals = trial_avg_and_CI(plot_signals)
                #     x = np.arange(len(plot_signals[0]))
                #     fig_ax.plot(x, plot_signals[0], color=plot_colors[trial_num])
                #     fig_ax.fill_between(x, plot_signals[1], plot_signals[2], color=plot_colors[trial_num], alpha=0.2)

                # textstr = '\n'.join((
                #     r'$tolerance=%.5f$' % (train_tolerance, ),
                #     r'$iterations=%d$' % (train_iterations, ),
                #     r'$states=%d$' % (num_states, )))
                # props = dict(boxstyle='round', facecolor='wheat', alpha=1)
                # ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                #         verticalalignment='top', bbox=props)

                #plt.show()

            symmranges = []
            mean_difs = []
                    
            if num_models_train == 1:
                for j, group in enumerate(raw_sensor):
                    sum_dif = 0
                    count = 0
                    if (j == 0):
                        m = 1 #only want the other trained model
                    else:
                        m = 0
                    x = calculate_state_correspondence_matrix(hmm_models_aligned_states[0][0], hmm_models_aligned_states[j][m], num_states)
                    sum_dif = sum_dif + calculate_gini_index(x, num_states)
                    # log HMM-SM similarity
                    print('%s - %s  :  %.5f' % (ordered_group_means[0], ordered_group_means[j], sum_dif))
                    symmrange = '%s - %s' % (ordered_group_means[0], ordered_group_means[j])
                    #add_row_to_csv(csv_path, arrangements[sensor_idx],'GPS','HMM-SM',participant, symmrange, sum_dif)
                    
            else: #If averaging a larger set (multiple HMMs trained)
                # i and j iterate over the trial types.
                # compare all permutations between HMMs in trial_types[i] and trial_types[j] to compute a mean HMM-SM similarity
                # between the two symmetry ranges. If i and j are the same (e.g., comparing within a symmetry range), don't compare
                # HMM to itself
                for j, group in enumerate(raw_sensor):
                    sum_dif = 0
                    count = 0
                    for k in range(num_models_train):
                        if(j == 0): #Always comparing to the first group, so if j (other group is same as baseline, make sure it is testing against )
                            indices = [a for a in range(num_models_train) if (not a == k)]
                        else:
                            indices = np.arange(num_models_train)
                        for m in indices:
                            x = calculate_state_correspondence_matrix(hmm_models_aligned_states[0][k], hmm_models_aligned_states[j][m], num_states)
                            sum_dif = sum_dif + calculate_gini_index(x, num_states)
                            count = count+1

                    # log average HMM-SM similarity
                    mean_dif = sum_dif / count
                    print('%s - %s  :  %.5f' % (ordered_group_means[0], ordered_group_means[j], mean_dif))
                    symmrange = '%s - %s' % (round(ordered_group_means[0],3), round(ordered_group_means[j],3))
                    symmranges.append(symmrange)
                    mean_difs.append(mean_dif)
                    #add_row_to_csv(csv_path, arrangements[sensor_idx],'STSR','HMM-SM',participant, symmrange, mean_dif)
                    
                
                
                     