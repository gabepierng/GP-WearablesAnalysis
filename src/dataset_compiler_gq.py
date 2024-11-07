import src.excel_reader_gcp as excel_reader
from google.cloud import storage
import numpy as np
import scipy
import re
import os

class DatasetCompiler():
    
    '''
    Organizes relevant Awinda info including participant (trial type), partitioned gait pattern data for each participant,
    and gait index-specific gait parameters
    '''
    class ParticipantData():
        def __init__(self, trial_type, partitioned_movement_data, ini_gait_params = None, mgs_temporal_params = None):
            self.trial_type = trial_type
            self.partitioned_movement_data = partitioned_movement_data
            self.ini_gait_params = ini_gait_params
            self.mgs_temporal_params = mgs_temporal_params
    
    def __init__(self, participants, bfb_participants, participant_info, bucket_dir):
        self.client = storage.Client()
        self.bucket_dir = bucket_dir
        
        self.participants = participants
        self.bfb_participants = bfb_participants
        self.participant_info = participant_info

        self.part_strides = {}
        self.part_gait_params = {}
        self.part_kinematic_params = {}
        self.control_strides = {}
        self.control_gait_params = {}
        self.control_kinematic_params = {}

        self.aggregate_control_data = {}
        self.control_strides_per_part = []

        self.partitioned_gait_data = {}

        self.params_INI = {
                    'stride time':[],
                    'stride length':[],
                    'swing phase':[],
                    'MAV':[],
                    'MAH':[],
                    'MHD':[],
                    'MAB':[],
                    'MAD':[],
                    'ShROM':[]
                    }

        self.params_temporal_mgs = {
                    'stance time':[],
                    'swing time':[],
                    'double support':[],
                    'step time':[],
                    'stride time':[]
                    }

        # indices of gait_params['spatio_temp'] (returned from excel_reader_gcp.process_mvn_trial_data())
        self.mgs_to_gait_params_map = {
                    'stance time':9,
                    'swing time':10,
                    'double support':11,
                    'step time':12,
                    'stride time':0
                    }

        self.ini_height_norm = {
                    'stride time':False,
                    'stride length':True,
                    'swing phase':False,
                    'MAV':True,
                    'MAH':True,
                    'MHD':True,
                    'MAB':True,
                    'MAD':True,
                    'ShROM':False
                    }

    '''
    re-interpolate gait cycles to common length for comparison using different models
    Inputs:
        vectors_orig: non-resized vectors, of shape N x T x D (N = number of gait cycles, T = length of time series, D = dimension)
        new_size: new length to interpolate signals to

    Output:
        new vector of dimensions N x new_size x D
    '''
    def reshape_vector(self, vectors_orig, new_size = 50):
        num_axes = vectors_orig[0].shape[-1]
        x_new = np.linspace(0, 100, new_size)
        trial_reshaped = []

        for stride in vectors_orig:
            x_orig = np.linspace(0, 100, len(stride))
            func_cubic = [scipy.interpolate.interp1d(x_orig, stride[:,i], kind='cubic') for i in range(num_axes)]
            vec_cubic = np.array([func_cubic[i](x_new) for i in range(num_axes)]).transpose()
            trial_reshaped.append(vec_cubic)

        return np.array(trial_reshaped)

    '''
    Stack accelerometer and gyroscope signals
    '''
    def combine_acc_and_gyro(self, numpy_acc, numpy_gyro):
        return np.concatenate((numpy_acc, numpy_gyro), axis=-1)

    '''
    Read .csv files with raw Xsens data. Feeds in files to excel_reader which parses those into individual gait cycles then calculates
    and returns a mix of spatiotemporal and kinematic gait parameters as well as the partitioned gait cycles
    Inputs:
        store_gait_cycles, store_gait_params, & store_kin_params: dictionaries where gait cycles and corresponding gait parameters will be stored
        filenames: files to read containing the raw Xsens recording data
        trial_type_filter: substring for input to Re (RegEx) for automatically pulling trial_type (e.g., participant) from filename
        print_filenames: Boolean whether to print filenames to command line as your parse
        look_at_all_files: Boolean to override desired_filetypes, if you don't want to apply any filters (e.g., only files with Baseline data, etc.)
        desired_filetypes: optional list with strings, if you only want to look at subset of files (e.g., files with Baseline in filename, etc.)

    Output:
        None, compile_gait_data stores processed data in store_gait_cycles, store_gait_params, & store_kin_params
    '''

    def compile_gait_data(self, store_gait_cycles, store_gait_params, store_kin_params, filenames, trial_type_filter, 
                            print_filenames=False, look_at_all_files = True, desired_filetypes=None):   
 
        XsensGaitParser = excel_reader.XsensGaitDataParser()  
        for i, file in enumerate(sorted(filenames)):
            trial_type = re.search(trial_type_filter, file).group(1)

            if(look_at_all_files or any(filetype in file for filetype in desired_filetypes)):
                XsensGaitParser.process_mvn_trial_data(os.path.join(self.bucket_dir, file), print_filenames)
                partitioned_mvn_data = XsensGaitParser.get_partitioned_mvn_data()
                gait_params = XsensGaitParser.get_gait_param_info()

                if trial_type in store_gait_cycles:
                    for body_part in store_gait_cycles[trial_type]:
                        for j, side in enumerate(store_gait_cycles[trial_type][body_part]):
                            # for each body segment (e.g., pelvis, l_hip, r_knee, etc.), append strides to appropriate list
                            store_gait_cycles[trial_type][body_part][j] = store_gait_cycles[trial_type][body_part][j] + partitioned_mvn_data[body_part][j]

                    for j, param in enumerate(store_gait_params[trial_type]):
                        store_gait_params[trial_type][j] = np.concatenate( (store_gait_params[trial_type][j], gait_params['spatio_temp'][j]), axis=-1)

                    for joint in store_kin_params[trial_type]:
                        for j, side in enumerate(store_kin_params[trial_type][joint]):
                            store_kin_params[trial_type][joint][j] = np.append(store_kin_params[trial_type][joint][j], gait_params['kinematics'][joint][j], axis=0)

                else:
                    store_gait_cycles[trial_type] = partitioned_mvn_data
                    store_gait_params[trial_type] = gait_params['spatio_temp']
                    store_kin_params[trial_type] = gait_params['kinematics']

    def load_participant_data(self):
        # dictionaries to store participant gait information

        for participant in self.participants:
            xsens_files = []
            if('AB' in participant):
                participant_dir =  os.path.join('Gait Quality Analysis/Data/Participant_Data/Raw Data/AB_BaselineWalking/', participant, 'CSV')
            else:
                participant_dir =  os.path.join('Gait Quality Analysis/Data/Participant_Data/Raw Data/LLA_xsens/PT Protocol Data/', participant, 'CSV')

            for blob in self.client.list_blobs('gaitbfb_propellab', prefix=participant_dir):
                if ('.csv' in blob.name):
                    xsens_files.append(blob.name)

            if('AB' in participant):
                self.compile_gait_data(self.part_strides, self.part_gait_params, self.part_kinematic_params, xsens_files, 'CSV/(.*?)_Baseline', False)
            else:
                self.compile_gait_data(self.part_strides, self.part_gait_params, self.part_kinematic_params, xsens_files, 'PT Protocol Data/(.*?)/CSV', False, False, ['Baseline', 'Pre'])


        for participant in self.bfb_participants:
            bfb_xsens_files = []
            participant_dir =  os.path.join('Wearable Biofeedback System (REB-0448)/Data/Raw Data/', participant, 'Excel_Data_Trimmed')

            for blob in self.client.list_blobs('gaitbfb_propellab', prefix=participant_dir):
                if ('.csv' in blob.name):
                    bfb_xsens_files.append(blob.name)
                    
            # compile_gait_data(part_strides, part_gait_params, part_kinematic_params, xsens_files, 'ExcelData/(.*?)-00')
            self.compile_gait_data(self.part_strides, self.part_gait_params, self.part_kinematic_params, bfb_xsens_files, 'Raw Data/(.*?)/Excel_Data_Trimmed', False, False, ['Baseline', 'Pre'])

    def load_control_data(self):
        control_files = []

        # directory with control data for gait quality scores
        control_dir = 'Gait Quality Analysis/Data/Participant_Data/Processed Data/AbleBodied_Control/CSV'
        for blob in self.client.list_blobs('gaitbfb_propellab', prefix=control_dir):
            if('.csv' in blob.name):
                control_files.append(blob.name)
                
        self.compile_gait_data(self.control_strides, self.control_gait_params, self.control_kinematic_params, control_files, 'CSV/(.*?)-00')

        
    def sample_control_dataset(self, strides_per_control):
        
        for i, indiv in enumerate(self.control_strides.keys()):
            indiv_height = self.participant_info.loc[self.participant_info['Participant'] == indiv]['Height (m)'].item()
            for j, param in enumerate(self.params_INI):
                if(self.ini_height_norm[param]):
                    self.control_gait_params[indiv][j] = self.control_gait_params[indiv][j] / indiv_height  
                    
            indices = np.arange(len(self.control_strides[indiv]['gyro_data'][0]))
            np.random.shuffle(indices)
            self.control_strides_per_part.append(min(strides_per_control, len(indices)))
            
            if(i == 0):
                self.aggregate_control_data = self.control_strides[indiv]
                
                for signal_type in self.control_strides[indiv]:
                    for j, side in enumerate(self.control_strides[indiv][signal_type]):
                        self.aggregate_control_data[signal_type][j] = [self.control_strides[indiv][signal_type][j][indices[k]] for k in range(min(strides_per_control, len(indices))) ]
                            
                for j, param in enumerate(self.params_INI.keys()):
                    for k in range(2):
                        self.params_INI[param].append([])
                        self.params_INI[param][k] = [self.control_gait_params[indiv][j][k][indices[a]] for a in range(min(strides_per_control, len(indices))) ]
                        
                for param in self.params_temporal_mgs.keys():
                    for k in range(2):
                        self.params_temporal_mgs[param].append([])
                        self.params_temporal_mgs[param][k] = [self.control_gait_params[indiv][self.mgs_to_gait_params_map[param]][k][indices[a]] 
                                                                                            for a in range(min(strides_per_control, len(indices))) ]
            else:
                # choose 10 gait cycles randomly from each able-bodied participant, or all gait cycles if less than 10
                for signal_type in self.control_strides[indiv]:
                    for j, side in enumerate(self.control_strides[indiv][signal_type]):
                        self.aggregate_control_data[signal_type][j] = self.aggregate_control_data[signal_type][j] + [self.control_strides[indiv][signal_type][j][indices[k]] 
                                                                                                        for k in range(min(strides_per_control, len(indices))) ]
                for j, param in enumerate(self.params_INI.keys()):
                    for k in range(2):
                        self.params_INI[param][k] = self.params_INI[param][k] + [self.control_gait_params[indiv][j][k][indices[a]] 
                                                                                    for a in range(min(strides_per_control, len(indices))) ]
                        
                for param in self.params_temporal_mgs.keys():
                    for k in range(2):
                        self.params_temporal_mgs[param][k] = self.params_temporal_mgs[param][k] + [self.control_gait_params[indiv][self.mgs_to_gait_params_map[param]][k][indices[a]] 
                                                                        for a in range(min(strides_per_control, len(indices))) ]

        for param in self.params_INI:
            self.params_INI[param] = np.array(self.params_INI[param])
            
        for param in self.params_temporal_mgs:
            self.params_temporal_mgs[param] = np.array(self.params_temporal_mgs[param])

    def compile_dataset(self):

        '''
        Compile inertial sensor and kinematic data as well as INI and MGS parameters into single object for each participant and collective control data
        Inputs:
            gait_cycle_data: dictionary of gait data (kinematics and inertial sensor data), 
                    each entry of shape N x T x D (N = number of gait cycles, T = length of time series, D = dimension)

        Output:
            ParticipantData object with formatted inertial sensor data, kinematic data (for computing Gait Profile Score), and corresponding gait parameters for
            the INI and MGS calculation
        '''
        def form_participant(gait_cycle_data):
            imu_resize_len = 40
            kin_resize_len = 51
            acc_scale = 0.02

            compiled_participant_signals = {}
            compiled_participant_signals['Pelvis_IMU'] = self.combine_acc_and_gyro(acc_scale*self.reshape_vector(gait_cycle_data['acc_data'][1], imu_resize_len), 
                                                                                        self.reshape_vector(gait_cycle_data['gyro_data'][1], imu_resize_len))
            compiled_participant_signals['UpperR_IMU'] = self.combine_acc_and_gyro(acc_scale*self.reshape_vector(gait_cycle_data['acc_data'][2], imu_resize_len), 
                                                                                        self.reshape_vector(gait_cycle_data['gyro_data'][2], imu_resize_len))
            compiled_participant_signals['UpperL_IMU'] = self.combine_acc_and_gyro(acc_scale*self.reshape_vector(gait_cycle_data['acc_data'][5], imu_resize_len), 
                                                                                        self.reshape_vector(gait_cycle_data['gyro_data'][5], imu_resize_len))
            compiled_participant_signals['LowerR_IMU'] = self.combine_acc_and_gyro(acc_scale*self.reshape_vector(gait_cycle_data['acc_data'][3], imu_resize_len), 
                                                                                        self.reshape_vector(gait_cycle_data['gyro_data'][3], imu_resize_len))
            compiled_participant_signals['LowerL_IMU'] = self.combine_acc_and_gyro(acc_scale*self.reshape_vector(gait_cycle_data['acc_data'][6], imu_resize_len), 
                                                                                        self.reshape_vector(gait_cycle_data['gyro_data'][6], imu_resize_len))

            compiled_participant_signals['pelvis_orient'] = self.reshape_vector(gait_cycle_data['pelvis_orient'][0], new_size = kin_resize_len)
            compiled_participant_signals['hip_angle'] = [self.reshape_vector(gait_cycle_data['hip_angle'][0], new_size = kin_resize_len), 
                                                         self.reshape_vector(gait_cycle_data['hip_angle'][1], new_size = kin_resize_len)]
            compiled_participant_signals['knee_angle'] = [self.reshape_vector(gait_cycle_data['knee_angle'][0], new_size = kin_resize_len), 
                                                          self.reshape_vector(gait_cycle_data['knee_angle'][1], new_size = kin_resize_len)]
            compiled_participant_signals['ankle_angle'] = [self.reshape_vector(gait_cycle_data['ankle_angle'][0], new_size = kin_resize_len), 
                                                           self.reshape_vector(gait_cycle_data['ankle_angle'][1], new_size = kin_resize_len)]

            return compiled_participant_signals
        
        partitioned_signals_awinda = form_participant(self.aggregate_control_data)
        self.partitioned_gait_data['control'] = self.ParticipantData('control', dict(partitioned_signals_awinda), dict(self.params_INI), dict(self.params_temporal_mgs))

        for trial_type in self.part_strides.keys():
            partitioned_signals_awinda = form_participant(self.part_strides[trial_type])

            indiv_height = self.participant_info.loc[self.participant_info['Participant'] == trial_type]['Height (m)'].item()
            for i, param in enumerate(self.params_INI):
                self.params_INI[param] = self.part_gait_params[trial_type][i]
                if(self.ini_height_norm[param]):
                    self.params_INI[param] = self.params_INI[param] / indiv_height
                    
            for param in self.params_temporal_mgs:
                self.params_temporal_mgs[param] = self.part_gait_params[trial_type][self.mgs_to_gait_params_map[param]]
                # param_data = self.params_temporal_mgs[param][0,:]
            
            self.partitioned_gait_data[trial_type] = self.ParticipantData(trial_type, dict(partitioned_signals_awinda), dict(self.params_INI), dict(self.params_temporal_mgs))