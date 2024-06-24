from google.cloud import storage
from scipy.signal import find_peaks
import numpy as np
import os
import sys
import scipy.interpolate as interp
sys.path.append(os.path.join(sys.path[0], '..', 'src'))
import excel_reader_gcp as excel_reader
import datetime
import logging   

sensor_mappings = {
    'pelvis': 1,
    'UpperR': 2,
    'LowerR': 3,
    'UpperL': 5,
    'LowerL': 6
}

XsensGaitParser =  excel_reader.XsensGaitDataParser()
storage_client = storage.Client()

def reshape_vector(vectors_orig, new_size, num_axes=3):
    x_new = np.linspace(0, 100, new_size)
    trial_reshaped = []
    for stride in vectors_orig:
        x_orig = np.linspace(0, 100, len(stride))
        func_cubic = [interp.interp1d(x_orig, stride[:, i], kind='cubic') for i in range(num_axes)]
        vec_cubic = np.array([func_cubic[i](x_new) for i in range(num_axes)]).transpose()
        trial_reshaped.append(vec_cubic)
    return np.array(trial_reshaped)

def organize_signals(sensor_mappings, gyro_signal, accel_signal):
    combined_signals = {}
    for location, sensor in sensor_mappings.items():
        reshaped_gyro = reshape_vector(gyro_signal[sensor], 40, 3)
        reshaped_accel = reshape_vector(accel_signal[sensor], 40, 3)
        combined = np.concatenate((reshaped_gyro, reshaped_accel), axis=2) #Concatenates to gyro x,y,z and accel x,y,z
        combined_signals[location] = combined
    return combined_signals

def finding_groupings(num_groups, gait_parameter, gait_cycles, percent_grading, reverse=True):
    if reverse:
        percent_grading = -percent_grading
        values_sorted = sorted(gait_parameter, reverse=True)
        sorted_indices = np.argsort(gait_parameter)[::-1]  # Sort indices in descending order of stance time symmetry
    else:
        sorted_indices = np.argsort(gait_parameter)
        values_sorted = sorted(gait_parameter, reverse=False)

    n = len(gait_parameter)
    group_sizes = [n // num_groups + (1 if i < n % num_groups else 0) for i in range(num_groups)]
    target_means = [np.mean(values_sorted[:group_sizes[0]])]

    # Initializes the groups and the remaining values to be picked from
    groups = [[] for _ in range(num_groups)]
    grouped_gait_cycles = [[] for _ in range(num_groups)]
    remaining_indices = sorted_indices[:]

    for i in range(1, num_groups):
        target_means.append(target_means[0] * (1 + percent_grading * i))

    for i in range(num_groups):
        target_mean = target_means[i]
        filtered_indices = [idx for idx in remaining_indices if abs(gait_parameter[idx] - target_mean) < target_mean * abs(percent_grading) / 2]
        selected_indices = filtered_indices[:]
        groups[i].extend(gait_parameter[idx] for idx in selected_indices)
        grouped_gait_cycles[i].extend(gait_cycles[idx] for idx in selected_indices)
        remaining_indices = [idx for idx in remaining_indices if idx not in selected_indices]

    return groups, grouped_gait_cycles, percent_grading            

def random_sampling(groups, grouped_gait_cycles, sample_size=50):
    def adaptive_subsample(group, first_mean, i, percent_grading=0.03, tolerance=0.002, sample_size=50, max_iterations=10000):
        available_indices = list(range(len(group)))  # Make a list that spans all the indices
        sample_indices = np.random.choice(available_indices, size=sample_size, replace=False)
        
        for idx in sample_indices:
            available_indices.remove(idx)  # Remove initial sample values from available values

        for _ in range(max_iterations):
            current_mean = np.mean([group[idx] for idx in sample_indices])
            percent_diff = (current_mean - first_mean) / first_mean
            target_diff = percent_grading * i

            if (target_diff - tolerance) <= abs(percent_diff) <= (target_diff + tolerance):
                return sample_indices

            if len(available_indices) == 0:
                raise ValueError("No candidates available to adjust the mean")

            if abs(percent_diff) < (target_diff - tolerance):
                if percent_diff < 0:
                    # Choose a new sample from the lower half
                    new_idx = np.random.choice([idx for idx in available_indices if group[idx] <= np.percentile(group, 50)])
                    sample_indices = np.append(sample_indices, new_idx)
                    available_indices.remove(new_idx)
                    sample_indices = np.delete(sample_indices, np.argmax([group[idx] for idx in sample_indices]))
                else:
                    # Choose a new sample from the upper half
                    new_idx = np.random.choice([idx for idx in available_indices if group[idx] >= np.percentile(group, 50)])
                    sample_indices = np.append(sample_indices, new_idx)
                    available_indices.remove(new_idx)
                    sample_indices = np.delete(sample_indices, np.argmin([group[idx] for idx in sample_indices]))
            else:
                if percent_diff > 0:
                    # Choose a new sample from the lower half
                    new_idx = np.random.choice([idx for idx in available_indices if group[idx] <= np.percentile(group, 50)])
                    sample_indices = np.append(sample_indices, new_idx)
                    available_indices.remove(new_idx)
                    sample_indices = np.delete(sample_indices, np.argmax([group[idx] for idx in sample_indices]))
                else:
                    # Choose a new sample from the upper half
                    new_idx = np.random.choice([idx for idx in available_indices if group[idx] >= np.percentile(group, 50)])
                    sample_indices = np.append(sample_indices, new_idx)
                    available_indices.remove(new_idx)
                    sample_indices = np.delete(sample_indices, np.argmin([group[idx] for idx in sample_indices]))

        raise ValueError("Could not find suitable subsample within the maximum number of iterations")

    indices_first_group = list(range(len(groups[0])))  
    sample_indices_first_group = np.random.choice(indices_first_group, size=sample_size, replace=False)
    group1_mean = np.mean([groups[0][idx] for idx in sample_indices_first_group]) # first mean used as the target for all subsequent groups
    
    subsampled_values = [groups[0][j] for j in sample_indices_first_group]
    subsampled_gait_cycles = [grouped_gait_cycles[0][j] for j in sample_indices_first_group]
    
    groups_subsampled_list = []
    gaitcycles_subsampled_list = []
    groups_subsampled_list.append(subsampled_values)
    gaitcycles_subsampled_list.append(subsampled_gait_cycles)
    
    for i in range(1, len(groups)):
        sample_indices = adaptive_subsample(np.array(groups[i]), group1_mean, i)
        subsampled_values = [groups[i][j] for j in sample_indices]
        subsampled_gait_cycles = [grouped_gait_cycles[i][j] for j in sample_indices]
        
        groups_subsampled_list.append(subsampled_values)
        gaitcycles_subsampled_list.append(subsampled_gait_cycles)
    
    return groups_subsampled_list, gaitcycles_subsampled_list

def check_group_configurations(gait_split_parameter, raw_sensor_data):
    
    groups, grouped_gait_cycles, grading = finding_groupings(4, gait_split_parameter, raw_sensor_data, 0.03, reverse=False)
    
    filtered_groups = []
    filtered_gait_groups = []
    
    for i in range(len(groups)):
        if len(groups[i]) > 70:
            filtered_groups.append(groups[i])
            filtered_gait_groups.append(grouped_gait_cycles[i])
    
    if len(filtered_groups) < 3:
        groups, grouped_gait_cycles, grading = finding_groupings(4, gait_split_parameter, raw_sensor_data, 0.03, reverse=True)  # Try the other direction if requirements are not fulfilled
        filtered_groups = []
        filtered_gait_groups = []
        
        for i in range(len(groups)):
            if len(groups[i]) > 70:
                filtered_groups.append(groups[i])
                filtered_gait_groups.append(grouped_gait_cycles[i])

        if len(filtered_groups) < 3:
            raise ValueError("Insufficient group sizes available for this participant")
    
    groups, gaitcycles = random_sampling(filtered_groups, filtered_gait_groups)
    
    return groups, gaitcycles

    
# Specify the bucket name and base directory within the bucket
bucket_name = 'gaitbfb_propellab/'
base_directory = bucket_name + 'Wearable Biofeedback System (REB-0448)/Data/Raw Data'
bucket_name = 'gaitbfb_propellab'
blobs = storage_client.list_blobs(bucket_name, prefix = base_directory)
prefix_from_bucket = 'Wearable Biofeedback System (REB-0448)/Data/Raw Data/' 
participant_list = ['LLPU_P01','LLPU_P02','LLPU_P03','LLPU_P04','LLPU_P05','LLPU_P06','LLPU_P08','LLPU_P09','LLPU_P10','LLPU_P12','LLPU_P14','LLPU_P15']

run_time = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")     
logging.basicConfig(filename=f'C:\Personal_Repo/{run_time}_participant.log',
                    format = "%(asctime)s %(levelname)s %(message)s",
                    level = logging.INFO)


for participant in participant_list:
    directory = prefix_from_bucket + participant + '/Excel_Data_Trimmed'
    blobs = storage_client.list_blobs(bucket_or_name=bucket_name, prefix=directory.replace("\\", "/"))
    
    part_strides = {}
    part_gait_params = {}
    part_kinematic_params = {}
    part_sensor_data = []
    trial_type = 'LLPU'
    arrangement = 'upper' #Alternatively, can use upper (UpperL + UpperR) or lower (LowerL + LowerR) (changes how many signals are added)
    logging.info(f"Processing participant {participant}")

    if blobs:
        for blob in blobs:
            if blob.name.endswith('.csv'):
                try:
                    XsensGaitParser.process_mvn_trial_data(f"gs://{bucket_name}/{blob.name}")
                    #print(f"Processing data from gs://{bucket_name}/{blob.name}")
                    partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
                    gait_params = XsensGaitParser.get_gait_param_info()
                    
                    gyro_signal = partitioned_mvn_data['gyro_data']
                    accel_signal = partitioned_mvn_data['acc_data']
                    
                    combined_signals = organize_signals(sensor_mappings, partitioned_mvn_data['gyro_data'], partitioned_mvn_data['acc_data'])
                    
                    pelvis_data = combined_signals['pelvis']
                    upper_data = np.concatenate((combined_signals['UpperR'], combined_signals['LowerR']), axis=2)  # Concatenate by last axis
                    lower_data = np.concatenate((combined_signals['UpperL'], combined_signals['LowerL']), axis=2)  # Concatenate by last axis
                    
                    if arrangement == 'pelvis':
                        part_sensor_data.append(pelvis_data)
                    elif arrangement == 'upper':
                        part_sensor_data.append(upper_data)
                    elif arrangement == 'lower':
                        part_sensor_data.append(lower_data)
                    else:
                        print("Not a valid sensor arrangement")
                        break
                    
                    partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
                    knee_angle = partitioned_mvn_data['knee_angle']
                    knee_angle_R = knee_angle[0]
                    trial_type = 'LLPU'
                    
                    if trial_type in part_strides:
                        for body_part in part_strides[trial_type]:
                            for i, side in enumerate(part_strides[trial_type][body_part]):
                                # for each part (pelvis, l_hip, r_knee, etc.), append strides to appropriate list
                                part_strides[trial_type][body_part][i] = part_strides[trial_type][body_part][i] + partitioned_mvn_data[body_part][i]

                        part_gait_params[trial_type].append(gait_params['spatio_temp'])

                        for joint in part_kinematic_params[trial_type]:
                            for i, side in enumerate(part_kinematic_params[trial_type][joint]):
                                part_kinematic_params[trial_type][joint][i] = np.append(part_kinematic_params[trial_type][joint][i], gait_params['kinematics'][joint][i], axis=0) 

                    else:
                        part_strides[trial_type] = partitioned_mvn_data
                        part_gait_params[trial_type] = [gait_params['spatio_temp']]
                        part_kinematic_params[trial_type] = gait_params['kinematics']

                except IndexError as e: #Exception based on an Index Error encountered in excel_reader_gcp.py **
                    #print(f"File skipped: gs://{bucket_name}/{blob.name} due to error: {e}")
                    continue
        
    if trial_type in part_gait_params:
        stance_time_symmetry = [item for sublist in [i[12] for i in part_gait_params[trial_type]] for item in sublist]
        print(len(stance_time_symmetry))      

        sorted_stance_time_symmetry = sorted(stance_time_symmetry,reverse=True)
        flattened_raw_sensor = []
        for sublist in part_sensor_data:
            for item in sublist:
                flattened_raw_sensor.append(item) #Flatten to individual gait cycles 


    groups, gaitcycles = check_group_configurations(stance_time_symmetry, flattened_raw_sensor)

    for i, group in enumerate(groups):
        print(f"Group {i+1}: {len(group)} (Mean: {np.mean(group) if group else 'N/A'})")
        percentdiff = (np.mean(group)-np.mean(groups[0]))/np.mean(groups[0])*100
        print(f"Percent diff between groups = {round(percentdiff,3)}")

    
    
    
                   
        