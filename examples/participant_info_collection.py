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

base_directory = "Q:\gaitbfb_propellab\Wearable Biofeedback System (REB-0448)\Data\Raw Data"

def calculate_magnitude(values):
    return np.linalg.norm(values)
                        
def reshape_vector(vectors_orig, new_size, num_axes=3):
    x_new = np.linspace(0, 100, new_size)
    trial_reshaped = []
    for stride in vectors_orig:
        x_orig = np.linspace(0, 100, len(stride))
        func_cubic = [interp.interp1d(x_orig, stride[:, i], kind='cubic') for i in range(num_axes)]
        vec_cubic = np.array([func_cubic[i](x_new) for i in range(num_axes)]).transpose()
        trial_reshaped.append(vec_cubic)
    return np.array(trial_reshaped)

# # Check if the CSV file exists, if not, create it and write headers
# if not os.path.exists(csv_file_path):
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Folder', 'Num Strides', 'Mean Symmetry', 'Stdev Symmetry', 
#                          'Upper 95% Symmetry', 'Lower 95% Symmetry', 'Min Symmetry', 'Max Symmetry'])


# Iterate through each subfolder in the base directory that starts with "AB"

participant_info = pd.read_excel('Q:\\main_propellab\\Users\\Ng, Gabe\\Summer Student 2024\\LLPU_DataSummaries\\LLPU_Height_ProstheticSide.xlsx')

def get_participant_info(participant_id):
    participant = participant_info[participant_info['Participant_ID'] == participant_id]
    if not participant.empty:
        height = participant.iloc[0]['Height']
        side = int(participant.iloc[0]['Side'])
        return height, side
    else:
        return None, None

for subfolder_name in os.listdir(base_directory):
    if subfolder_name.startswith("LLPU"):
        folder_path = os.path.join(base_directory, subfolder_name)
        participant_id = subfolder_name
        height,side = get_participant_info(participant_id)
        print(type(side))
        height_m = height/100 
        
        if side == 1:
            prosth_side = 0 #Right side
            intact_side = 1
            side_label = "Right"
            intact_label = "Left"
            print("Prosthetic on the right side")
        else:
            prosth_side = 1 #Left side
            intact_side = 0
            side_label = "Left"
            intact_label = "Right"
            print("Prosthetic side on the left side")
        
        if height is None or side is None:
            print(f"Participant info not found for ID: {participant_id}")
            continue
        
        excel_data_path = os.path.join(folder_path, "Excel_Data_Trimmed")
        if os.path.isdir(excel_data_path):
            knee_roms_list = []
            hip_roms_list = []
            ankle_roms_list = []
            step_lengths_list_intact = []
            step_lengths_list_prosth = []
            print(f"Processing folder: {excel_data_path}")
            part_strides = {}
            part_gait_params = {}
            part_kinematic_params = {}
            trial_type = 'LLPU'
            strides = 0
            for filename in os.listdir(excel_data_path):
                #if filename.endswith('.csv') and filename.startswith('Baseline'):
                if filename.endswith('.csv'):
                    xsens_path_file = os.path.join(excel_data_path, filename)
                    print(f"Processing file: {xsens_path_file}")
                    try:
                        XsensGaitParser.process_mvn_trial_data(xsens_path_file)
                        partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
                        gait_params = XsensGaitParser.get_gait_param_info()
                        
                        # step_lengths = gait_params['spatio_temp'][10]
                        
                        
                        # step_lengths_intact = step_lengths[intact_side]/height_m 
                        # step_lengths_list_intact.append(step_lengths_intact)
                        # step_lengths_prosth = step_lengths[prosth_side]/height_m 
                        # step_lengths_list_prosth.append(step_lengths_prosth)
                        
                        # knee_roms = gait_params['spatio_temp'][13]
                        # # Isolate the sagittal plane angle (flexion/extension)
                        # knee_ROMs_full = [
                        # [sublist[:][0][-1], sublist[:][1][-1]]
                        # for sublist in knee_roms]
                        
                        # knee_ROMs_sideofinterest = [
                        # sublist[:][intact_side]
                        # for sublist in knee_ROMs_full] 
                    
                        # knee_roms_list.append(knee_ROMs_sideofinterest)   
                        
                        # hip_roms = gait_params['spatio_temp'][14]
                        # # Isolate the sagittal plane angle (flexion/extension)
                        # hip_ROMs_full = [
                        # [sublist[:][0][-1], sublist[:][1][-1]]
                        # for sublist in hip_roms]
                        
                        # hip_ROMs_sideofinterest = [
                        # sublist[:][intact_side]
                        # for sublist in hip_ROMs_full] 
                        
                        # hip_roms_list.append(hip_ROMs_sideofinterest) 
                        
                        # ankle_roms = gait_params['spatio_temp'][15]
                        
                        # ankle_ROMs_full = [
                        # [sublist[:][0][-1], sublist[:][1][-1]]
                        # for sublist in ankle_roms]
                        
                        # ankle_ROMs_sideofinterest = [
                        # sublist[:][prosth_side]
                        # for sublist in ankle_ROMs_full] 
                        
                        # ankle_roms_list.append(ankle_ROMs_sideofinterest) 
                
                        
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
                        #logging.error(f"Skipping file {xsens_path_file} due to error: {e}")
                        continue
                    
            # Compute symmetry values
            if trial_type in part_gait_params:
                if prosth_side == 1:
                    stance_time_symmetry = [1/item for sublist in [i[11] for i in part_gait_params[trial_type]] for item in
                                            sublist]
                else:
                    stance_time_symmetry = [item for sublist in [i[11] for i in part_gait_params[trial_type]] for item in
                                            sublist]
                    
                knee_roms = [item for sublist in knee_roms_list for item in sublist]
                step_lengths_intact = [item for sublist in step_lengths_list_intact for item in sublist]
                step_lengths_prosth = [item for sublist in step_lengths_list_prosth for item in sublist]
                hip_roms = [item for sublist in hip_roms_list for item in sublist]
                ankle_roms = [item for sublist in ankle_roms_list for item in sublist]
                
                
                #Evaluate normality of datasets:
                from scipy import stats
                
                # Generate histogram and scatter plot side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Histogram
                bincount = int(np.sqrt(len(stance_time_symmetry))) #Bin count selected based on the "square root rule (for now)"
                ax1.hist(stance_time_symmetry, bins=bincount, edgecolor='black', alpha=0.7)
                ax1.set_title(f'STSR - {subfolder_name}')
                # ax1.set_xlabel('Sagittal Ankle ROM (deg)')
                # ax1.set_ylabel('Frequency')
                ax1.grid(True)

                # Scatter plot
                y = (np.zeros(len(stance_time_symmetry))) + np.random.normal(0, 0.02, len(stance_time_symmetry))
                ax2.scatter(stance_time_symmetry, y, s=6, color='black')
                # ax2.set_title(f'Ankle Hip ROM (Prosthetic Side) - {subfolder_name}')
                # ax2.set_xlabel('Ankle Hip ROM (deg)')
                ax2.grid(True)

                plt.savefig(f"C:\\Users\\ekuep\\Desktop\\STSR_{participant_id}.jpg")
                plt.tight_layout()
                plt.show()
                # plt.figure()
                # y = (np.zeros(len(step_lengths_intact))) + np.random.normal(0, 0.02, len(step_lengths_intact))
                # plt.scatter(step_lengths_intact, y, label = 'Intact limb')
                # plt.scatter(step_lengths_prosth, y, label = 'Prosthetic limb')
                # plt.title(f"Sagittal Knee ROM - {participant_id}")
                # plt.legend()
                #plt.savefig(f'Q:\\main_propellab\\Users\\Ng, Gabe\\Summer Student 2024\\LLPU_DataSummaries\\ankleROM\\Sagittal_ProstheticSide\\{subfolder_name}_prosthetic_ankleROM.jpg')

                # Compute and print additional stats
                # speed = [item for sublist in [i[16] for i in part_gait_params[trial_type]] for item in sublist]
                # cadence = [item for sublist in [i[17] for i in part_gait_params[trial_type]] for item in sublist]

                
                # print('Mean Speed: %.3f ± %.3f' % (np.mean(speed), np.std(speed)))
                # print('Mean Cadence: %.3f ± %.3f' % (np.mean(cadence), np.std(cadence)))
                # print('Num strides: ', len(step_lengths))
                # print('Mean step length R: %.3f ± %.3f' % (np.mean(step_lengths), np.std(step_lengths)))
                # print('Upper 95% of the data: ', sorted(step_lengths)[round(0.95 * len(step_lengths))])
                # print('Lower 95% of the data: ', sorted(step_lengths)[round(0.05 * len(step_lengths))])
                # print() 
                # logging.info(f'Folder: {excel_data_path}')
                # logging.info(f'NumStrides: {len(stance_time_symmetry)}')
                # logging.info(f'MeanSymmetry: {round(np.mean(stance_time_symmetry),3)}')
                # logging.info(f'StdevSymmetry: {round(np.std(stance_time_symmetry),3)}')
                # logging.info(f'Upper95%Symmetry: {sorted(stance_time_symmetry)[round(0.95 * len(stance_time_symmetry),3)]}')
                # logging.info(f'Lower95%Symmetry: {sorted(stance_time_symmetry)[round(0.05 * len(stance_time_symmetry),3)]}')
                # logging.info(f'MINSymmetry: {round(np.min(stance_time_symmetry),3)}')
                # logging.info(f'MAXSymmetry: {round(np.max(stance_time_symmetry),3)}')
                # logging.info('--------------')
                
              
            else:
                print(f"No data found for trial type '{trial_type}' in folder {excel_data_path}")
        else:
            print(f"Folder 'Excel_Data_Trimmed' not found")
            
            