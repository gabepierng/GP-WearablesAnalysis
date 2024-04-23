import cloudstorage as gcs
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

from scipy import signal
import scipy.interpolate as interp
import excel_reader_gcp as excel_reader
from hmmlearn import hmm
import logging
import datetime

use_manual_filenames = False

# create log file to store model experiment results
run_time = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")
logging.basicConfig(filename=f'./log_files/{run_time}_HMM_Sweep.log',
                    format = "%(asctime)s %(levelname)s %(message)s",
                    level = logging.INFO)

# establish filepath to desired participant data
# participant_info: csv file matching Awinda and DOT data files
ab_or_llpu = 'AB'
bucket_dir = 'gs://gaitbfb_propellab/'
if(ab_or_llpu == 'AB'):
    participant_info = bucket_dir + 'Wearable Biofeedback System (REB-0448)/Data/Raw Data/Data_Info_AB_vG_5Sensors.csv'
else:
    participant_info = bucket_dir + 'Wearable Biofeedback System (REB-0448)/Data/Raw Data/Data_Info_LLPU.csv'

df = pd.read_csv(participant_info)

part_strides = {}
part_gait_params = {}
part_kinematic_params = {}
temp_len = 0

# store xsens DOT data for each trial.
# each element of the list contains the DOT data for that trial for each sensor location collected
partitioned_signals_dot = []
# target_SRs = []
participant_of_interest = 28
look_at_all_files = True

signals_of_interest = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

'''
For time aligning two time series signals, specifically the Xsens DOT (60Hz) and Awinda (100Hz)
Inputs:
    packetCount: packet numbers from Xsens DOT
    original_signal: signal to convert, as a 1 x T array (T = number of packets collected)
    old_freq: frequency of original signal
    new_freq: frequency to interpolate to

Return:
    y_new: original accelerometer or gyroscope signal, interpolated to new frequency

'''
def convert_dot_freq(packetCount, original_signal, old_freq=60, new_freq=100):
    x_o = np.array(packetCount)
    x_new = np.linspace(x_o[0], x_o[-1], int(x_o[-1] * (new_freq / old_freq)) )
    func_cubic = interp.interp1d(x_o, original_signal, kind='cubic')
    y_new = func_cubic(x_new)
    return y_new

'''
calculate Q matrix as established in Sahraeian et al. (https://ieeexplore.ieee.org/document/5654664)
Inputs:
    hmm_1 and hmm_2: HMM models to be compared. Type = GaussianHMM Object from hmmlearn library
    n_states: number of states for the HMMs

Return:
    Q matrix: N x N matrix, N = n_states. Each element of Q, Q[i, j] = similarity of B_1[i] and B_2[j]
        for HMMs 1 and 2 to be compared.

'''
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


'''
calculate normalized Gini Index as described in Sahraeian et al. (https://ieeexplore.ieee.org/document/5654664)
Inputs:
    q_matrix: N x N similarity matrix
    n_states: number of states for the HMMs

Return:
    gini_index: sparsity of q_matrix, normalized between 0 (low sparsity/similarity) and 1 (high sparsity/similarity)
'''
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

'''
Resample gait cycle vectors at effectively lower sampling rate, to reduce the size of the time series being used to train the HMMs
and normalize the gait cycles to the same time length
Inputs:
    vectors_orig: list of time series signals to normalize to same length. Each element of the list is expected to be an
        T_o x A numpy array, where T_o = number of packets in original sample, and A = number of axes included (e.g. 6 for tri-axial gyro and acc)
    new_size: new length for signals to be interpolated to
    num_axes: A, expected axes of input signal

Return:
    np.array(trial_reshaped): N x T_n x A matrix. N = number of gait cycles, T_n = new length of time series, A = axes for signal

'''
def reshape_vector(vectors_orig, new_size, num_axes = 3):
    x_new = np.linspace(0, 100, new_size)
    trial_reshaped = []

    for stride in vectors_orig:
        # print(stride.shape)
        x_orig = np.linspace(0, 100, len(stride))
        func_cubic = [interp.interp1d(x_orig, stride[:,i], kind='cubic') for i in range(6)]
        vec_cubic = np.array([func_cubic[i](x_new) for i in range(num_axes)]).transpose()
        trial_reshaped.append(vec_cubic)

    return np.array(trial_reshaped)


'''
Initialize HMM class and models
Functions:
    __init__: initialize HMM object
    train: train HMM model
    get_score: return likelihood of sample time series being produced by HMM model

    more information about hmmlearn and parameters found at https://hmmlearn.readthedocs.io/en/latest/index.html

'''
class HMMTrainer(object):
    '''
    Inputs:
        model_name: type of probability distributions to represent emissions matrices
        n_components: number of states to initialize HMM
        cov_type: how to populate covariance matrix of emission probabilities. Options 'full', 'diag', 'spherical', 'tied'
        n_iter: max iterations to train HMM
        tolerance: cutoff for likelihood gait during training (i.e., model fitting stops once likelihood gait < tolerance)
    '''
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='full', n_iter=50, tolerance = 0.005):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []
        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter, tol=tolerance, params='stmc', init_params='mc', verbose=False)
            self.model.transmat_ = np.zeros((n_components, n_components))
            for i in range(n_components):
                # initialize transition matrix to encourage a left-to-right model. Found that with this library, setting the 
                # non-elements to 0 decreased consistency of the training when visualizing hidden-state sequence predictions. 
                # Setting them to very small but non-zero values promotes l-to-r learning and improved HMM training consistency in testing
                self.model.transmat_[i, 0:2] = np.random.dirichlet(np.ones(2)/1.5, size=1)
                self.model.transmat_[i, 2:] = np.random.dirichlet(np.ones(self.n_components - 2), size=1)[0] / 1e80
                self.model.transmat_[i] = np.roll(self.model.transmat_[i], i)


        else:
            raise TypeError('Invalid model type') 

    # resize_len = length of training sequences
    def train(self, X, resize_len, num_sequences):
        np.seterr(all='ignore')

        self.model.fit(X, lengths=resize_len * np.ones(num_sequences, dtype=int))
        # x_concat = np.concatenate(X).reshape(-1,1)
        # self.models.append(self.model.fit(X))
        # self.models.append(self.model.fit(x_concat, lengths = [X.shape[1] for i in range(X.shape[0])]))
        # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)
        # return self.model.score(input_data.reshape(-1,1))

# ranges of stance-time symmetry ratio used for the able-bodied participants, for 3-symmetry-level paradigm
sym_range_parts = {
    14: [[1.01, 1.05], [0.94, 0.98], [0.87, 0.91]],
    15: [[1.07, 1.13], [1.00, 1.04], [0.90, 0.98]],
    16: [[1.02, 1.07], [0.97, 1.01], [0.88, 0.95]],
    19: [[1.02, 1.07], [0.97, 1.01], [0.90, 0.95]],
    22: [[0.98, 1.02], [0.90, 0.95], [0.83, 0.87]],
    23: [[0.99, 1.04], [0.91, 0.96], [0.83, 0.88]],
    24: [[0.98, 1.02], [0.94, 0.97], [0.87, 0.93]],
    25: [[0.99, 1.03], [0.95, 0.98], [0.89, 0.94]],
    26: [[1.01, 1.04], [0.97, 1.01], [0.92, 0.96]],
    27: [[0.99, 1.05], [0.955, 0.98], [0.88, 0.95]],
    28: [[0.95, 0.99], [0.91, 0.94], [0.85, 0.90]]
}

# sensor configurations to test. Single sensors, and combining the upper leg signals and lower leg signals together
sensor_combos = [['Pelvis'],
                ['LowerR', 'LowerL'],
                ['UpperR', 'UpperL'],
                ['UpperR'],
                ['UpperL'],
                ['LowerR'],
                ['LowerL']
                ]

# parameters to use for the HMM model training. Number of gait cycles to concatenate (along time axis), HMM states, iterations, and training tolerance
strides_to_concat = 4
num_states=5
train_iterations = 300
train_tolerance = 1e-2
logging.info(f'Strides to concat: {strides_to_concat}   # HMM States: {num_states}   Max Iter.: {train_iterations}    Tol.: {train_tolerance}')

# Overall pipeline
# For each participant, load their Xsens MVN and DOT data. Partition DOT gait data from each sensor into gait cycles, and extract relevant stance-time
# symmetry for those gait cycles. Split gait cycles and STSR into levels determined by STSR ranges, concatenate gait cycles, and train HMMs on the 
# gait cycles data within each symmetry level. Compute mean HMM-similarity measure (HMM-SM) between symmetry levels and within symmetry level

# for participant_of_interest in sym_range_parts:
for participant_of_interest in list(reversed(list(sym_range_parts.keys()))):
    logging.info(f'Training participant {str(participant_of_interest)}...')
# for participant_of_interest in [14]:

    # create google cloud storage client, for reading data stored in the google cloud server
    storage_client = storage.Client()
    for index, row in df.iterrows():
        xsens_file = row['Xsens Filename'][:-4] + 'csv'
        prefix_from_bucket = 'Wearable Biofeedback System (REB-0448)/Data/Raw Data/' + row['Participant']
        trial_num = row['App Filename']

        participant_num = int(re.search(ab_or_llpu + '_P(.*)', row['Participant']).group(1))

        if(participant_num == participant_of_interest):

            trial_type = re.search('(.*)-0', row['Xsens Filename']).group(1)

            if(trial_type == 'Training'):
                dot_trial_type = 'COL'
            else:
                dot_trial_type = trial_type


            # search for .json file with information of each trial such as sensor placement
            for blob in storage_client.list_blobs('gaitbfb_propellab', prefix=os.path.join(prefix_from_bucket, 'App_Data/')):
                file = blob.name
                if( ((str(trial_num) + '.json') in file) and (dot_trial_type in file) and not ('Audio' in file) and not ('._' in file)):
                    bfb_json_file = file

            bucket = storage_client.get_bucket('gaitbfb_propellab')
            json_blob = bucket.blob(bfb_json_file)
            x = json.loads(json_blob.download_as_string(client=None))

            # initialSR = x['description']['initialSR']

            # gets unique sensor locations from logged data folder, i.e., ul, ur, p, etc.
            dot_sensor_data = list(set([f.name.strip('.csv').split('_')[-1] for f in storage_client.list_blobs('gaitbfb_propellab', 
                                                                                        prefix=os.path.join(prefix_from_bucket, 'App_Data_Logged/Renamed/'))
                                                                                       if not f.name[-1] == '/']))
            dot_sensor_data = dict.fromkeys(dot_sensor_data)
            temp_dot = {}

            # get filenames for each DOT sensor location corresponding to this trial
            for location in dot_sensor_data.keys():
                for blob in storage_client.list_blobs('gaitbfb_propellab', prefix=os.path.join(prefix_from_bucket, 'App_Data_Logged/Renamed/')):
                    filename = blob.name 
                    if( ('_' + str(trial_num) + '_' + location) in filename):
                        temp_dot[filename] = None

            # get xsens file number for this trial (e.g., 1, 2, etc.)
            xsens_trial_num = int(re.search('-0(.*).mvnx', row['Xsens Filename']).group(1))

            for location, filename in zip(dot_sensor_data.keys(), temp_dot.keys()):
                # link each DOT file to pandas dataframe with DOT accelerometer and gyroscope data
                temp_dot[filename] = pd.read_csv(os.path.join(bucket_dir, filename))

                # due to data pipeline, some csv files have "sep=," at beginning which needs to be removed/skipped for pd.read_csv to read columns correctly
                # this checks to see whether this is the case, and need to reread the csv file.
                if(len(temp_dot[filename].columns) == 2 ):
                    temp_dot[filename] = pd.read_csv(os.path.join(bucket_dir, filename), skiprows=1)

                # dot_sensor_data = list of xsens DOT data from each location, with this data interpolated to 100 Hz to match the MVN data (for time aligning)
                dot_sensor_data[location] = np.array([convert_dot_freq(temp_dot[filename]['PacketCounter'], temp_dot[filename][sig]) for sig in signals_of_interest]).T

            # load Xsens MVN full data
            xsens_path_file = os.path.join(bucket_dir, prefix_from_bucket, 'Excel_Data/', xsens_file)
            # lower_body_strides, gait_params, partitioned_dot_signal = excel_reader.process_trial_data(0, xsens_path_file, xsens_dot[0][-1], stride_events)

            '''
            excel_reader: reads MVN data (in .csv format), extracts gait cycles and suite of spatiotemporal, kinematic, and signal parameters as well as
                time-aligning to Xsens DOT data
            Outputs:
                lower_body_strides: MVN Data partitioned into gait cycles for each limb recorded during trial
                gait_params: gait parameters calculated from the MVN data
                partitioned_signals: Xsens DOT data for the trial, partitioned into gait cycles using MVN foot contact, following time-alignment with MVN data
            '''
            partitioned_mvn_data, gait_params, gait_events = excel_reader.process_mvn_trial_data(0, xsens_path_file, dot_sensor_data)

            partitioned_signals_dot.append(excel_reader.time_align_mvn_and_dot(dot_sensor_data, partitioned_mvn_data[]))

            # reorganizes so that mvn data from all trials (e.g. partitioned mvn gait cycles, gait parameters) stored in single arrays.
            # trial type is set static for here, since all trials are grouped together. Can be set to variable parameter or read from filenames (if named under proper convention)
            # such as baseline, post-intervention, etc. if multiple trial conditions or types that want to group the MVN data by.
            trial_type = 'AB'
            # target_SRs.append(initialSR)
            if trial_type in part_strides:
                for body_part in part_strides[trial_type]:
                    for i, side in enumerate(part_strides[trial_type][body_part]):
                        # for each part (pelvis, l_hip, r_knee, etc.), append strides to appropriate list
                        part_strides[trial_type][body_part][i] = part_strides[trial_type][body_part][i] + lower_body_strides[body_part][i]

                part_gait_params[trial_type].append(gait_params['spatio_temp'])
                # print(part_gait_params['AB'])

                for joint in part_kinematic_params[trial_type]:
                    for i, side in enumerate(part_kinematic_params[trial_type][joint]):
                        part_kinematic_params[trial_type][joint][i] = np.append(part_kinematic_params[trial_type][joint][i], gait_params['kinematics'][joint][i], axis=0)

            else:
                part_strides[trial_type] = lower_body_strides
                part_gait_params[trial_type] = [gait_params['spatio_temp']]
                part_kinematic_params[trial_type] = gait_params['kinematics']

    resize_len = 40

    # butterworth filter, to filter some high frequency noise from the data
    b20, a20 = scipy.signal.butter(N=4, Wn = 0.8, btype = 'lowpass')  # Wn = 0.8 = 40 / Nyquist F = 50Hz

    # get sensor locations again for dot sensors. Downsample each to resize_len, to reduce time series length
    sensor_locs = partitioned_signals_dot[0].keys()
    for i in range(len(partitioned_signals_dot)):
        for sensor_location in sensor_locs:
            partitioned_signals_dot[i][sensor_location] = reshape_vector(partitioned_signals_dot[i][sensor_location], resize_len, 6)


    partitioned_signals_dot_grouped = {}
    overflow_check = []
    
    # restructure partitioned DOT data from each trial into single data object/variable
    # rarely, DOT data will have large spike in data (could be caused by sensor blip or accidental contact with arms during gait cycle, not sure). 
    # overlow_check used to find gait cycles with this issue to remove from analysis
    for sensor_location in sensor_locs:
        partitioned_signals_dot_grouped[sensor_location] = np.concatenate([x[sensor_location] for x in partitioned_signals_dot], axis=0)

        for i, stride in enumerate(partitioned_signals_dot_grouped[sensor_location]):
            if (np.max(stride) > 1000):
                overflow_check.append(i)

    for sensor_location in sensor_locs:
        partitioned_signals_dot_grouped[sensor_location] = np.delete(partitioned_signals_dot_grouped[sensor_location], overflow_check, axis=0)

    # get all stance time symmetry values from MVN data (in order matching the partitioned dot signals), and delete corresponding overflow STSR elements
    stance_time_symmetry = [item for sublist in [i[12] for i in part_gait_params['AB']] for item in sublist]
    stance_time_symmetry = np.delete(stance_time_symmetry, overflow_check)

    # visualize STSR values per participant
    # plt.figure(figsize = (10, 4))
    # y = (np.zeros(len(stance_time_symmetry))) + np.random.normal(0,  0.02, len(stance_time_symmetry))
    # plt.scatter(stance_time_symmetry, y, s = 6, color='black')

    # if(ab_or_llpu == 'AB'):
    #     plt.xlim([0.7, 1.10])
    # else:
    #     plt.xlim([0.8, 1.2])

    sym_ranges = sym_range_parts[participant_of_interest]
    
    sym_range_strides = {}              # store gait cycles for each of STSR ranges
    symmetry_split_into_ranges = []     # corresponding STSR values
    sym_strides_to_add = [ [] for _ in range(len(sym_ranges)) ]

    # determine indices for gait cycle/STSR values within each of the STSR ranges
    for i in range(len(stance_time_symmetry)):
        for j in range(len(sym_ranges)):
            if((stance_time_symmetry[i] > sym_ranges[j][0]) and (stance_time_symmetry[i] < sym_ranges[j][1])):
                sym_strides_to_add[j].append(i)
                break

    # move STSR values, trial types named by mean STSR for each group
    # sym_range_strides = dict of [trial types][DOT sensor locations], elements being numpy arrays
    trial_types = []
    for i in range(len(sym_ranges)):
        symmetry_split_into_ranges.append( np.array([stance_time_symmetry[k] for k in sym_strides_to_add[i]]) )
        trial_types.append('Avg. Sym - ' + str( np.round( np.mean(symmetry_split_into_ranges[-1]), 3) ) )
        sym_range_strides[trial_types[-1]] = {}
        for sensor in sensor_locs:
            sym_range_strides[trial_types[-1]][sensor] = partitioned_signals_dot_grouped[sensor][sym_strides_to_add[i]]

    
    for sensor_array_to_test in sensor_combos:
        if(len(sensor_array_to_test) == 1):
            logging.info(f'Sensors used: {sensor_array_to_test[0]}...')
        else:
            logging.info(f'Sensors used: {sensor_array_to_test[0]} and {sensor_array_to_test[1]}...')
        partitioned_strides = {}

        # if testing sensor combo, concatenate sensors by last axis
        for i, trial_type in enumerate(trial_types):
            if(len(sensor_array_to_test) == 1):
                partitioned_strides[trial_type] = sym_range_strides[trial_type][sensor_array_to_test[0]][:,:,0:]
            else:
                partitioned_strides[trial_type] = np.concatenate([sym_range_strides[trial_type][sensor][:,:,0:] for sensor in sensor_array_to_test], axis=2)

            np.random.shuffle(partitioned_strides[trial_type])
            # print('Number of %s strides: %d' % (trial_type, len(partitioned_strides[trial_type])))

        # concatenate multiple strides together. Done in a "sliding window" over gait cycles, to still have reasonable amount of training samples
        concat_strides = {}

        for trial_type in trial_types:
            concat_strides[trial_type] = []
            for i in range(partitioned_strides[trial_type].shape[0] - strides_to_concat):
                temp = []
                for j in range(strides_to_concat):
                    temp.append(partitioned_strides[trial_type][i + j])

                concat_strides[trial_type].append(np.concatenate(temp, axis=0))

            concat_strides[trial_type] = np.array(concat_strides[trial_type])
            concat_strides[trial_type] = signal.filtfilt(b20, a20, concat_strides[trial_type], axis=1)


        strides_train_flat = {}
        strides_test_flat = {}
        strides_train = {}
        strides_test = {}

        hmm_models = {}
        num_models_train = 10

        # for each symmetry range, train num_models_train HMMs on respective training data, stored in hmm_models dict
        for trial_type in trial_types:
            # print('Training %s models...' % (trial_type))
            hmm_models[trial_type] = []

            for j in range(num_models_train):
                train_forward_model = True
                k = 0

                while(train_forward_model):
                    # print('Train Attempt ', k+1, end="\r", flush=True)
                    if(j > -1):
                        np.random.shuffle(concat_strides[trial_type])

                    # flatten sequence for hmmlearn train function
                    strides_sequence_flattened = concat_strides[trial_type].reshape((concat_strides[trial_type].shape[0] * concat_strides[trial_type].shape[1], -1))

                    # technically is no training/testing data, but this preserves a few gait cycles to compare the hidden-state sequence predictions of the HMMs
                    len_train = int(0.95 * len(concat_strides[trial_type]))
                    strides_train[trial_type] = concat_strides[trial_type][:len_train]
                    strides_test[trial_type] = concat_strides[trial_type][len_train:]
                    sequence_length = resize_len * strides_to_concat
                    strides_train_flat[trial_type] = strides_sequence_flattened[:sequence_length * len_train]
                    strides_test_flat[trial_type] = strides_sequence_flattened[sequence_length * len_train:]

                    hmm_model = HMMTrainer(n_components = num_states, n_iter = train_iterations, tolerance = train_tolerance)
                    hmm_model.train(strides_train_flat[trial_type], sequence_length, len_train)

                    
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

                    # if model is left-to-right, consider model trained, train next model (until num_models_train reached)
                    if(valid_rows == num_states):
                        train_forward_model = False
                    k = k + 1

                # print()
                hmm_models[trial_type].append(hmm_model)

        # print('done')
        

        test_predict = strides_test[trial_types[0]][1]
        min_predict = np.min(test_predict[:,1])
        max_predict = np.max(test_predict[:,1])


        # technically shouldn't be necessary for the HMM similarity measure, but useful for comparing hidden state
        # sequence predictions and plotting to align the HMM states and emissions matrices
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
        for i in range(num_states):
            pred_vals[i] = min_predict + ((i * (max_predict - min_predict)) / (num_states - 1))

        roll_amounts = {}
        match_trials = {}

        for trial in trial_types:
            roll_amounts[trial] = [0 for i in range(num_models_train)]
            match_trials[trial] = 0

        shift_all = 0

        '''
        HMMs may have similar transition matrices, but associate the states with different parts of the gait cycle.
        For example, for HMM-1, states 1,2,3 might correspond to 2,3,1 in HMM-2. To align them for hidden-state
        sequence comparison, need to rotate the transition matrices and emissions matrices. Function compares hidden
        state sequence predictions on test input, finds alignment with minimum error between two.

        Inputs:
            hmm_1, hmm_2: HMM models to align
            test_stride: time-series to generate hidden-state sequence predictions for comparing alignment
            n_states: number of states for HMM models

        Returns roll amount as input to np.roll to rotate elements of HMM matrices to align HMM-1 and HMM-2
        '''
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
        for i, trial_type in enumerate(trial_types):
            predictions[trial_type] = []
            hmm_models_aligned_states[trial_type] = []
            match_trials[trial_type] = find_best_alignment(hmm_models[trial_types[0]][0], hmm_models[trial_type][0], test_predict, num_states)

            for j in range(num_models_train):
                roll_amounts[trial_type][j] = find_best_alignment(hmm_models[trial_type][0], hmm_models[trial_type][j], test_predict, num_states) + match_trials[trial_type]
                # roll_amounts[trial_type][j] = 0
                hmm_models_aligned_states[trial_type].append(align_states(hmm_models[trial_type][j], roll_amounts[trial_type][j] + shift_all))
                predictions[trial_type].append(hmm_models_aligned_states[trial_type][-1].model.predict(test_predict))
    

        '''
        bunch of stuff for visualizing the hidden-state sequence predictions

        print(roll_amounts[trial_types[1]][0])
        fig, ax = plt.subplots()
        fig.set_size_inches(12,8)
        plt.plot(test_predict[:,0])

        plt.plot([pred_vals[j] for j in predictions[trial_types[0]][0]], 'k')
        plt.plot([pred_vals[j] for j in predictions[trial_types[0]][3]], 'r')

        # plt.plot([pred_vals[j] for j in predictions_post[4]], 'k')
        # plt.plot([pred_vals[j] for j in predictions_post[2]], 'r')

        def trial_avg_and_CI(signal_set):
            conf_int_mult = 1.00    # confidence interval multiplier for 1 std

            avg_signal = np.mean(signal_set, axis=0)
            std_signal = np.std(signal_set, axis=0)
            upper_bound = avg_signal + (conf_int_mult * std_signal)
            lower_bound = avg_signal - (conf_int_mult * std_signal)

            return avg_signal, upper_bound, lower_bound

        def confidence_plot(plot_signals, fig_ax, trial_num):
            plot_signals = trial_avg_and_CI(plot_signals)
            x = np.arange(len(plot_signals[0]))
            fig_ax.plot(x, plot_signals[0], color=plot_colors[trial_num])
            fig_ax.fill_between(x, plot_signals[1], plot_signals[2], color=plot_colors[trial_num], alpha=0.2)

        textstr = '\n'.join((
            r'$tolerance=%.5f$' % (train_tolerance, ),
            r'$iterations=%d$' % (train_iterations, ),
            r'$states=%d$' % (num_states, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        plt.show()

        '''

        q_matrix = np.zeros((num_states, num_states))
        div_scores = np.zeros((num_states, num_states))

        # i and j iterate over the trial types.
        # compare all permutations between HMMs in trial_types[i] and trial_types[j] to compute a mean HMM-SM similarity
        # between the two symmetry ranges. If i and j are the same (e.g., comparing within a symmetry range), don't compare
        # HMM to itself
        for i in range(len(trial_types)):
            # print()
            for j in range(len(trial_types)):
                sum_dif = 0
                count = 0
                for k in range(num_models_train):
                    if(i == j):
                        indices = [a for a in range(num_models_train) if (not a == k)]
                    else:
                        indices = np.arange(num_models_train)
                    for m in indices:
                        x = calculate_state_correspondence_matrix(hmm_models_aligned_states[trial_types[i]][k], hmm_models_aligned_states[trial_types[j]][m], num_states)
                        sum_dif = sum_dif + calculate_gini_index(x, num_states)
                        count = count+1

                # log average HMM-SM similarity
                mean_dif = sum_dif / count
                logging.info(f'{trial_types[i]} - {trial_types[j]}   :   {str(np.round(mean_dif, 5))}')
                # print('%s - %s  :  %.5f' % (trial_types_print[i], trial_types_print[j], mean_dif))

        logging.info('')
