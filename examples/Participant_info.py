from google.cloud import storage
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import re
import os
import json
import scipy
import seaborn as sns
import copy
import sys

from scipy import signal
import scipy.interpolate as interp
sys.path.append(os.path.join(sys.path[0], '..', 'src'))
import excel_reader_gcp as excel_reader
from hmmlearn import hmm
import logging
import datetime
import csv



XsensGaitParser =  excel_reader.XsensGaitDataParser()
"""
folder_path = "Q:\\gaitbfb_propellab\\Wearable Biofeedback System (REB-0448)\\Data\\Raw Data\\LLPU_P01\\Excel_Data\\"


part_strides = {}
part_gait_params = {}
part_kinematic_params = {}
trial_type = 'LLPU'
folder_name = os.path.basename(os.path.dirname(folder_path.rstrip('\\')))

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        xsens_path_file = os.path.join(folder_path, filename)
        print(f"Processing file: {xsens_path_file}")

        
        XsensGaitParser.process_mvn_trial_data(xsens_path_file)
        partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
        gait_params = XsensGaitParser.get_gait_param_info()

        
        if trial_type in part_strides:
            for body_part in part_strides[trial_type]:
                for i, side in enumerate(part_strides[trial_type][body_part]):
                    part_strides[trial_type][body_part][i] += partitioned_mvn_data[body_part][i]

            part_gait_params[trial_type].append(gait_params['spatio_temp'])

            for joint in part_kinematic_params[trial_type]:
                for i, side in enumerate(part_kinematic_params[trial_type][joint]):
                    part_kinematic_params[trial_type][joint][i] = np.append(
                        part_kinematic_params[trial_type][joint][i],
                        gait_params['kinematics'][joint][i], axis=0
                    )
        else:
            part_strides[trial_type] = partitioned_mvn_data
            part_gait_params[trial_type] = [gait_params['spatio_temp']]
            part_kinematic_params[trial_type] = gait_params['kinematics']

stance_time_symmetry = [item for sublist in [i[12] for i in part_gait_params[trial_type]] for item in sublist]

speed = [item for sublist in [i[16] for i in part_gait_params[trial_type]] for item in sublist]
cadence = [item for sublist in [i[17] for i in part_gait_params[trial_type]] for item in sublist]

print('Mean Speed: %.3f ± %.3f' % (np.mean(speed), np.std(speed)))
print('Mean Cadence: %.3f ± %.3f' % (np.mean(cadence), np.std(cadence)))
print('Num strides: ', len(stance_time_symmetry))
print('Mean symmetry: %.3f ± %.3f' % (np.mean(stance_time_symmetry), np.std(stance_time_symmetry)))
print('Upper 95% of the data: ', sorted(stance_time_symmetry)[round(0.95 * len(stance_time_symmetry))])
print('Lower 95% of the data: ', sorted(stance_time_symmetry)[round(0.05 * len(stance_time_symmetry))])
print()

plt.figure(figsize=(10, 6))
plt.hist(stance_time_symmetry, bins=15, edgecolor='black', alpha=0.7)
plt.title(f'Stance Time Symmetry Distribution - {folder_name}')
plt.xlabel('Stance Time Symmetry')
plt.ylabel('Frequency')
plt.show()
"""
#Base directory
base_directory = "Q:\\gaitbfb_propellab\\Wearable Biofeedback System (REB-0448)\\Data\\Raw Data"

# Iterate through each subfolder in the base directory that starts with "LLPU"
for subfolder_name in os.listdir(base_directory):
    if subfolder_name.startswith("LLPU_P10"):
        llpu_folder_path = os.path.join(base_directory, subfolder_name)

        # Look for the "Excel_Data_Trimmed" folder within the LLPU folders
        excel_data_path = os.path.join(llpu_folder_path, "Excel_Data_Trimmed")
        if os.path.isdir(excel_data_path):
            print(f"Processing folder: {excel_data_path}")

            
            part_strides = {}
            part_gait_params = {}
            part_kinematic_params = {}
            trial_type = 'LLPU'

            # Looping through each file in the "Excel_Data_Trimmed" folder
            for filename in os.listdir(excel_data_path):
                if filename.endswith('.csv'):
                    xsens_path_file = os.path.join(excel_data_path, filename)
                    print(f"Processing file: {xsens_path_file}")

                    # Process the file
                    XsensGaitParser.process_mvn_trial_data(xsens_path_file)
                    partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
                    gait_params = XsensGaitParser.get_gait_param_info()

                    # Aggregate data
                    if trial_type in part_strides:
                        for body_part in part_strides[trial_type]:
                            for i, side in enumerate(part_strides[trial_type][body_part]):
                                part_strides[trial_type][body_part][i] += partitioned_mvn_data[body_part][i]

                        part_gait_params[trial_type].append(gait_params['spatio_temp'])

                        for joint in part_kinematic_params[trial_type]:
                            for i, side in enumerate(part_kinematic_params[trial_type][joint]):
                                part_kinematic_params[trial_type][joint][i] = np.append(
                                    part_kinematic_params[trial_type][joint][i],
                                    gait_params['kinematics'][joint][i], axis=0
                                )
                    else:
                        part_strides[trial_type] = partitioned_mvn_data
                        part_gait_params[trial_type] = [gait_params['spatio_temp']]
                        part_kinematic_params[trial_type] = gait_params['kinematics']

            # Compute symmetry values
            if trial_type in part_gait_params:
                stance_time_symmetry = [item for sublist in [i[12] for i in part_gait_params[trial_type]] for item in
                                        sublist]

                # Generate histogram
                plt.figure(figsize=(10, 6))
                plt.hist(stance_time_symmetry, bins=30, edgecolor='black', alpha=0.7)
                plt.title(f'Stance Time Symmetry Distribution - {subfolder_name}')
                plt.xlabel('Stance Time Symmetry')
                plt.ylabel('Frequency')
                plt.show()

                # Compute and print additional stats
                speed = [item for sublist in [i[16] for i in part_gait_params[trial_type]] for item in sublist]
                cadence = [item for sublist in [i[17] for i in part_gait_params[trial_type]] for item in sublist]

                print(f'Folder: {excel_data_path}')
                print('Mean Speed: %.3f ± %.3f' % (np.mean(speed), np.std(speed)))
                print('Mean Cadence: %.3f ± %.3f' % (np.mean(cadence), np.std(cadence)))
                print('Num strides: ', len(stance_time_symmetry))
                print('Mean symmetry: %.3f ± %.3f' % (np.mean(stance_time_symmetry), np.std(stance_time_symmetry)))
                print('Upper 95% of the data: ', sorted(stance_time_symmetry)[round(0.95 * len(stance_time_symmetry))])
                print('Lower 95% of the data: ', sorted(stance_time_symmetry)[round(0.05 * len(stance_time_symmetry))])
                print()
            else:
                print(f"No data found for trial type '{trial_type}' in folder {excel_data_path}")
        else:
            print(f"Folder 'Excel_Data_Trimmed' not found in {llpu_folder_path}")