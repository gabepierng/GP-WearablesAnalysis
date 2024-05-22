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

base_directory = "Q:\\gaitbfb_propellab\\Wearable Biofeedback System (REB-0448)\\Data\\Raw Data"
run_time = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")
logging.basicConfig(filename=f'../log_files/{run_time}_ParticipantInfo.log',
                    format = "%(asctime)s %(levelname)s %(message)s",
                    level = logging.INFO)

# Iterate through each subfolder in the base directory that starts with "LLPU"
for subfolder_name in os.listdir(base_directory):
    if subfolder_name.startswith("LLPU_P11"):
        llpu_folder_path = os.path.join(base_directory, subfolder_name)

        # Look for the "Excel_Data_Trimmed" folder within each of the LLPU folders
        excel_data_path = os.path.join(llpu_folder_path, "Excel_Data_Trimmed")
        if os.path.isdir(excel_data_path):
            print(f"Processing folder: {excel_data_path}")
            
            part_strides = {}
            part_gait_params = {}
            part_kinematic_params = {}
            trial_type = 'LLPU'

            
            for filename in os.listdir(excel_data_path):
                if filename.endswith('.csv'):
                    xsens_path_file = os.path.join(excel_data_path, filename)
                    print(f"Processing file: {xsens_path_file}")
                    try:
                        
                        XsensGaitParser.process_mvn_trial_data(xsens_path_file)
                        partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
                        gait_params = XsensGaitParser.get_gait_param_info()

                        # Aggregate all of the data for each participant 
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
                    
                    except IndexError as e: #Exception based on an Index Error encountered in the excel_reader_gcp.py code -- still need to debug this **
                        print(f"Skipping file {xsens_path_file} due to error: {e}")
                        logging.error(f"Skipping file {xsens_path_file} due to error: {e}")
                        continue
                    
            # Compute symmetry values
            if trial_type in part_gait_params:
                stance_time_symmetry = [item for sublist in [i[12] for i in part_gait_params[trial_type]] for item in
                                        sublist]

                # Generate histogram and scatter plot side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Histogram
                bincount = int(np.sqrt(len(stance_time_symmetry))) #Bin count selected based on the "square root rule (for now)"
                ax1.hist(stance_time_symmetry, bins=bincount, edgecolor='black', alpha=0.7)
                ax1.set_title(f'Histogram of Stance Time Symmetry - {subfolder_name}')
                ax1.set_xlabel('Stance Time Symmetry')
                ax1.set_ylabel('Frequency')
                ax1.grid(True)

                # Scatter plot
                y = (np.zeros(len(stance_time_symmetry))) + np.random.normal(0, 0.02, len(stance_time_symmetry))
                ax2.scatter(stance_time_symmetry, y, s=6, color='black')
                ax2.set_title(f'Scatterplot of Stance Time Symmetry - {subfolder_name}')
                ax2.set_xlabel('Stance Time Symmetry')
                ax2.grid(True)

                plt.tight_layout()
                plt.show()
                fig.savefig(f'{subfolder_name}.jpg')

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
                logging.info(f'Folder: {excel_data_path}')
                logging.info(f'NumStrides: {len(stance_time_symmetry)}')
                logging.info(f'MeanSymmetry: {round(np.mean(stance_time_symmetry),3)}')
                logging.info(f'StdevSymmetry: {round(np.std(stance_time_symmetry),3)}')
                logging.info(f'Upper95%Symmetry: {sorted(stance_time_symmetry)[round(0.95 * len(stance_time_symmetry),3)]}')
                logging.info(f'Lower95%Symmetry: {sorted(stance_time_symmetry)[round(0.05 * len(stance_time_symmetry),3)]}')
                logging.info(f'MINSymmetry: {round(np.min(stance_time_symmetry),3)}')
                logging.info(f'MAXSymmetry: {round(np.max(stance_time_symmetry),3)}')
                logging.info('--------------')
                
            else:
                print(f"No data found for trial type '{trial_type}' in folder {excel_data_path}")
        else:
            print(f"Folder 'Excel_Data_Trimmed' not found in {llpu_folder_path}")
            
            