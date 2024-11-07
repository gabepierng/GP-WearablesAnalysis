import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import math
from EntropyHub import SampEn
from tslearn.metrics import dtw_path
from minisom import MiniSom

# =============================================================== #
# =============== ADDITIONAL GAIT QUALITY METHODS =============== #
# =============================================================== #
'''
Calculates the Gait Profile Score (GPS) as outlined by Baker et al. (https://pubmed.ncbi.nlm.nih.gov/19632117/)
Inputs:
    part_kinematics: kinematics array for individual participant. Pelvic orientation all 3 axes (unilateral) and right/left
        hip angles (all 3 planes), knee flexion, ankle flexion, and foot progression = 15 signals
        Current calculation excludes foot progression
    control_kinematics: equivalent dictionary for control dataset
    
    both part_kinmatics and control_kinematics should be reshaped to be 2% increments of the gait cycle. This leads to 51
    data points (eg. HS to subsequent HS)
'''
def calculate_gait_profile_score(part_kinematics, control_kinematics):
    # organize kinematics into single array
    def create_kin_profile(kinematics_dict):
        kinematics_arr = []
        kinematics_arr.append(np.mean(kinematics_dict['pelvis_orient'], axis=0).transpose())  # pelvis orientation (x, y, z)
        kinematics_arr.append(np.mean(kinematics_dict['hip_angle'][0], axis = 0).transpose()) # hip abduction/adduction, rotation, flexion
        kinematics_arr.append(np.mean(kinematics_dict['hip_angle'][1], axis = 0).transpose())
        kinematics_arr.append(np.expand_dims(np.mean(kinematics_dict['knee_angle'][0][:,:,2], axis = 0), axis=0))  # knee flexion
        kinematics_arr.append(np.expand_dims(np.mean(kinematics_dict['knee_angle'][1][:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(kinematics_dict['ankle_angle'][0][:,:,2], axis = 0), axis=0)) # ankle flexion
        kinematics_arr.append(np.expand_dims(np.mean(kinematics_dict['ankle_angle'][1][:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(kinematics_dict['ankle_angle'][0][:,:,1], axis = 0), axis=0)) # foot progression (ankle rotation)
        kinematics_arr.append(np.expand_dims(np.mean(kinematics_dict['ankle_angle'][1][:,:,1], axis = 0), axis=0))
        
        kinematics_arr = np.concatenate(kinematics_arr, axis=0)
        return kinematics_arr
    
    
    # mean kinematics arrays. Should be shape [Number of signals, 51 time points]
    part_mean_kinematics = create_kin_profile(part_kinematics)
    control_mean_kinematics = create_kin_profile(control_kinematics)
    
    gait_variability_score = [np.linalg.norm(part_mean_kinematics[i] - control_mean_kinematics[i]) / np.sqrt(len(part_mean_kinematics[i]))
                                 for i in range(len(part_mean_kinematics))]
    
    gait_profile_score = np.linalg.norm(gait_variability_score) / np.sqrt(len(gait_variability_score))
    
    return gait_variability_score, gait_profile_score


""" 
TSLEARN DTW - documentation: https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw_path.html#tslearn.metrics.dtw_path
    Parameters: 
        set1, set2 (List): List containing (sz,D) arrays as elements, where D is the dimension of the data, sz is the length of the gait cycle 
    Returns: 
        Mean DTW distance between partitioned gait cycles - if set 1 and set 2 are inputs, the DTW distance between datasets is computed. If set 1 is an input, the within DTW distance is computed.
"""
   
def tslearn_dtw_analysis(set1, set2=None):
    def tslearndtw_distance(data1, data2):
        path, distance = dtw_path(data1.astype(np.double), data2.astype(np.double))
        return distance
    
    dtw_dist = [] 
    
    if set2 is None:
        for i in range(len(set1)):
            for j in range(i + 1, len(set1)):
                distance = tslearndtw_distance(set1[i], set1[j])
                dtw_dist.append(distance)
        return np.mean(dtw_dist)
    else:
        for gaitcycle1 in set1:
            for gaitcycle2 in set2:
                distance = tslearndtw_distance(gaitcycle1, gaitcycle2)
                dtw_dist.append(distance)
        return np.mean(dtw_dist)

"""
Training a Self Organizing Map (SOM)
    Parameters:
        gait_data (numpy array): The input data for training the SOM. Dimensions N x D, where D is the dimension of the data 
            and N is the number of gait cycles multiplied by the length of the time series (e.g., the number of gait cycles and
            the length of the time series must be flattened into single dimension before being intput to train_minisom)
        learning_rate (float): The initial learning rate for the SOM.
        topology (str): The topology of the SOM ('rectangular' or 'hexagonal').
    Returns:
        som (class 'minisom.MiniSom'): Trained SOM
"""

def train_minisom(gait_data, desired_steps, learning_rate=0.1, topology='hexagonal', normalize=True, visualize_u_map=False):
    
    if normalize:
        means = np.mean(gait_data, axis=0, keepdims=True)
        mean_centered_array = gait_data - means
        std_devs = np.std(mean_centered_array, axis=0, ddof=0, keepdims=True) #Along each column 
        gait_data = mean_centered_array / std_devs
        
    # Parameters for training/initializing: 
    dim = 5 * np.sqrt(gait_data.shape[0]) # Heuristic: # map units = 5*sqrt(n), where n is the number of training samples [1]
    # print("Desired dimensions: %d" % dim)
    cov  = np.cov(gait_data.T)
    w, v = la.eig(cov)
    ranked = sorted(abs(w))
    ratio = ranked[-1]/ranked[-2] # Ratio between side lengths of the map
    
    # Solving for the dimensions of the map - (1) x*y = dim, (2) x/y = ratio -- solve the system of equations
    y_sq = dim / ratio
    y = np.sqrt(y_sq)
    x = ratio*y
    x = round(x)
    y = round(y)
    msize = (x,y) # Map dimensions 
    # print("Map size: %d x %d" % (msize[0], msize[1]))
    # print("Number of map units: ", x*y)
    
    steps = 500 * (dim ** 2) # Number of iterations - previous heuristic: 500 * number of network units [2] 
    steps = math.ceil(10*(x*y)/gait_data.shape[0]) #Based on .trainlen = 10*m/n (m is # map units, n is the number of training samples)
    # steps = desired_steps
    sigma = max(msize) / 4 # Sigma used: Heuristic: max(msize)/4 [1] 
    
    #Initializing the SOM
    som = MiniSom(x=msize[0], y=msize[1], input_len=gait_data.shape[1], sigma=sigma, learning_rate=learning_rate, topology=topology) # Initializing the SOM 
    som.random_weights_init(gait_data) # Initializes the weights of the SOM picking random samples from data.
    som.train(gait_data, steps, use_epochs=True) 
    
    som._learning_rate = learning_rate/10 # Reduce the learning rate on the next iteration 
    som._sigma = max(sigma/4,1)           # Drops to a quarter of original sigma, unless it is less than 1
    steps2 = steps*4                      # Based on .trainlen = 40*m/n

    som.train(gait_data, steps2, use_epochs=True)
    
    # Option for visualizing the map
    if(visualize_u_map):
        u_matrix = som.distance_map().T
        plt.figure(figsize=(10,10))
        plt.pcolor(u_matrix, cmap= 'viridis' )
        plt.colorbar()
        plt.show()
    
    return som
    
    """    
    Resources
    [1]	    Juha. Vesanto and (Libella), SOM toolbox for Matlab 5. Helsinki University of Technology, 2000.
    [2]	    S. Lek and Y. S. Park, “Artificial Neural Networks,” Encyclopedia of Ecology, Five-Volume Set, vol. 1–5, pp. 237–245, Jan. 2008, doi: 10.1016/B978-008045405-4.00173-7.
    """

"""    
Calculates Mean MDP scores for the given data and trained SOM.
Parameters:
    data (numpy array): The input data for which Mean MDP scores are calculated, a single gait cycle (T x D), where T is the length of the 
        time-series, and D is the number of dimensions.
    som (MiniSom): The trained SOM.
    means (numpy array): The means used for normalization.
    std_devs (numpy array): The standard deviations used for normalization.
    normalize (bool): Whether the data was normalized.
Returns:
    deviation: Euclidian distance between BMU and dataset for each time point -- used to generate MDP
"""

def calculate_MDP(data, som_train_data, som, normalize=True):
    
    normalized_gait_data = None
    if normalize: # normalize the test data to scale of SOM trained on normalized gait data
        means = np.mean(som_train_data, axis=0, keepdims=True)
        mean_centered_array = data - means
        std_devs = np.std(mean_centered_array, axis=0, ddof=0, keepdims=True) #Along each column 
        normalized_test_data = mean_centered_array / std_devs
    
    bmu_vector = np.array([som._weights[som.winner(instance)] for instance in normalized_test_data])
    denormed_bmu_vector = (bmu_vector * std_devs) + means
    deviation = np.linalg.norm(data - denormed_bmu_vector, axis=1)  # Calculate Euclidean distances in the full dataset
    return deviation

def calculate_mean_MDP(test_mdp_dataset, control_gait_cycles, train_som):
    sum_mdp = 0
    for gait_cycle in test_mdp_dataset:
        mdp_score = calculate_MDP(gait_cycle, np.reshape(control_gait_cycles, (control_gait_cycles.shape[0] * control_gait_cycles.shape[1], -1)), train_som)
        sum_mdp = sum_mdp + np.mean(mdp_score)
        # mdp_over_gait_cycle.append(mdp_score)

    mean_mdp = sum_mdp / test_mdp_dataset.shape[0]
    return mean_mdp


def calculate_eigencomponents_INI(control_data):
    control_INI_params = []
    for param in control_data.ini_gait_params:
        control_INI_params.append(control_data.ini_gait_params[param][0])
    control_INI_params = np.transpose(np.array(control_INI_params))

    mean_control = np.mean(control_INI_params, axis = 0)
    std_control = np.std(control_INI_params, axis = 0)

    standardized_control = (control_INI_params - mean_control) / std_control
    cov_control = np.cov(standardized_control.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_control)

    return ((mean_control, std_control), (eig_vals, eig_vecs))

'''
Calculates the IMU-based Gait Normalcy Index (INI), proposed measure of gait normalcy based on 3 spatiotemporal and 
6 kinematic parameters which can be determined (theoretically) using inertial sensor data. https://pubmed.ncbi.nlm.nih.gov/32224469/ 
Inputs:
    control_mean: mean parameters of control group, numpy array of shape (P,), where P is the number of parameters (9 for the INI)
    control_std: standard deviation of parameters for control group
    eig_vals: eigenvalues of PCA transformed vectors
    eig_vecs: corresponding eigenvectors following PCA transformation
    data: non-transformed gait parameters to be used for calculating the INI

Output:
    gini_index: returns float in range[0,1]. 1 indicates maximum sparsity
'''       
def calculate_INI(summarized_control_data, eigen_comps_INI, participant_data):
    control_mean, control_std = summarized_control_data
    eig_vals, eig_vecs = eigen_comps_INI
    
    side_1 = np.transpose(np.array( [participant_data.ini_gait_params[param][0] for param in participant_data.ini_gait_params] ))
    side_2 = np.transpose(np.array( [participant_data.ini_gait_params[param][1] for param in participant_data.ini_gait_params] ))
    participant_bilateral_avg = np.mean([side_1, side_2], axis=0)

    standardized_data = (participant_bilateral_avg - control_mean) / control_std
    transformed_data = (np.dot(standardized_data, eig_vecs)) / np.sqrt(eig_vals)

    ini_score = np.mean(np.linalg.norm(transformed_data, axis=-1))
    
    return ini_score



def determine_mgs_reduced_param_set(control_data, control_strides_per_part, participant_data):

    '''
    The multifeature gait score (MGS) is a proposed measure of gait normalcy based on a large collection of temporal
    and signal-based features from the inertial sensors. https://pmc.ncbi.nlm.nih.gov/articles/PMC5648116/
    The MGS uses PCA and some established thresholds to select a set of the "most important" gait parameters
    Inputs:
        pelvis_signal: IMU signal from pelvis
        temporal_params: temporal parameters

    Output:
        mgs_params: reduced set of the parameters identified as important for the MGS
    '''       
    def calculate_MGS_params(pelvis_signal, temporal_params):
        norm_acc = np.expand_dims(np.linalg.norm(pelvis_signal[:,:,:3], axis=-1), axis=-1)
        norm_gyro = np.expand_dims(np.linalg.norm(pelvis_signal[:,:,3:], axis=-1), axis=-1)
        pelvis_mgs_sig = np.concatenate((pelvis_signal, norm_acc, norm_gyro), axis=-1)
        
        concatenated_mgs_signals = []
        flattened_signal = np.reshape(pelvis_mgs_sig, (pelvis_mgs_sig.shape[0] * pelvis_mgs_sig.shape[1], -1))
        x_std = np.std(flattened_signal, axis=0)
        
        n = 4   # number of gait cycles to concatenate for the MGS score
        num_cycles = len(pelvis_mgs_sig)
        for i in range(int(num_cycles / n)):
            temp = [pelvis_mgs_sig[ (i*n)+j, :, : ] for j in range(n)]
            temp = np.concatenate(temp)
            concatenated_mgs_signals.append(temp)
        
        concatenated_mgs_signals = np.array(concatenated_mgs_signals)
        
        amp_range = np.mean(np.max(pelvis_mgs_sig, axis=1) - np.min(pelvis_mgs_sig, axis=1), axis=0)
        amp_rms = np.mean(np.linalg.norm(pelvis_mgs_sig, axis=1) / np.sqrt(pelvis_mgs_sig.shape[1]), axis=0)
        
        mean_temp_params = np.mean( np.mean(temporal_params, axis=1), axis=-1)
        
        skew = np.mean(stats.skew(pelvis_mgs_sig, axis=1), axis=0)
        kurtosis = np.mean(stats.kurtosis(pelvis_mgs_sig, axis=1), axis=0)
        entropy = []
        for i in range(concatenated_mgs_signals.shape[0]):
            temp = []
            for j in range(concatenated_mgs_signals.shape[-1]):
                x = concatenated_mgs_signals[i,:,j]
                sig_ent, _, _ = SampEn(x / x_std[j], m=3, r=1.0)
                temp.append(sig_ent[-1])
            entropy.append(temp)
        entropy = np.mean(np.array(entropy), axis=0)
        symmetry = []
        for i, param in enumerate(temporal_params):
            # handle edge case of double-stance support time being 0 (one participant occasionally with assistive device) --> div by 0 --> inf symmetry
            symmetry.append([ param[0,j]/param[1,j] for j in range(param.shape[-1]) if not ((param[0,j] == 0) or (param[1,j] == 0)) ])

        # normalize symmetry to between 0 and 1 for participants (e.g., between participants, doesn't matter
        # if right or left side has the larger parameter values)
        symmetry = np.array([np.mean(symmetry[i]) for i in range(len(symmetry))])
        for i in range(len(symmetry)):
            if (symmetry[i] > 1):
                symmetry[i] = 1 / symmetry[i]

        regularity = np.mean([1 - (np.abs(param[0,:] - param[1,:]) / (np.abs(param[0,:]) + np.abs(param[1,:]))) for param in temporal_params], axis=1)
        mgs_params = np.concatenate((amp_range, amp_rms, mean_temp_params, skew, kurtosis, entropy, symmetry, regularity))
        
        return mgs_params

    mgs_aspects = [
        np.arange(0,16),
        np.arange(16,21),
        np.arange(21,37),
        np.arange(37,45),
        np.arange(45,50),
        np.arange(50,55)
    ]
    mgs_params = []
    # part_mgs_params = []
    # control_mgs_params = []

    index = 0
    for i in range(len(control_strides_per_part)):
        control_indiv_strides = control_data.partitioned_movement_data['Pelvis_IMU'][index:index+control_strides_per_part[i], :, :]
        control_indiv_params = np.array([control_data.mgs_temporal_params[param][:,index:index+control_strides_per_part[i]] 
                                                                                                for param in control_data.mgs_temporal_params])
        index = index + control_strides_per_part[i]
        # control_mgs_params.append(calculate_MGS_params(control_indiv_strides, control_indiv_params))
        mgs_params.append(calculate_MGS_params(control_indiv_strides, control_indiv_params))
        
    for i in range(len(participant_data)):
        mgs_temporal_params = np.array([participant_data[i].mgs_temporal_params[param] for param in control_data.mgs_temporal_params])
        # part_mgs_params.append(calculate_MGS_params(participant_data[i].partitioned_movement_data['Pelvis_IMU'], mgs_temporal_params))
        mgs_params.append(calculate_MGS_params(participant_data[i].partitioned_movement_data['Pelvis_IMU'], mgs_temporal_params))


    # control_mgs_params = np.array(control_mgs_params)  
    # part_mgs_params = np.array(part_mgs_params)  
    # mgs_params = np.array(np.concatenate([control_mgs_params, part_mgs_params], axis=0))
    mgs_params = np.array(mgs_params)

    mean_mgs = np.mean(mgs_params, axis = 0)
    std_mgs = np.std(mgs_params, axis = 0)

    standardized_mgs = (mgs_params - mean_mgs) / std_mgs
    cov_mgs = np.cov(standardized_mgs.T)

    eig_vals_mgs, eig_vecs_mgs = np.linalg.eig(cov_mgs)
    eig_vecs_mgs_real = []


    for i, val in enumerate(eig_vals_mgs):
        if not(val.real < 1.0):
            eig_vecs_mgs_real.append([comp.real for comp in eig_vecs_mgs[:,i]])

    eig_vecs_mgs_real = np.array(eig_vecs_mgs_real).T

    transformed_mgs_data = np.dot(standardized_mgs, eig_vecs_mgs_real)
    reduced_param_set = []

    for i in range(eig_vecs_mgs_real.shape[1]):
        for aspect in mgs_aspects:
            temp = []
            for param in aspect:
                res = stats.pearsonr(standardized_mgs[:,param], transformed_mgs_data[:,i])
                if( (np.abs(res[0]) > 0.4) and (res[1] < 0.05) ):
                    temp.append([param, np.abs(res[0])])
            temp = np.array(temp)
            if not(temp.size == 0):
                highest_aspect_param = temp[np.argmax(temp[:,1]), 0]
                if not (highest_aspect_param in reduced_param_set):
                    reduced_param_set.append(int(highest_aspect_param))

    return (mgs_params, reduced_param_set)

# calculate partial MGS scores based on the MGS parameter set
def calculate_mgs(control_mgs_params, indiv_full_gait_params, reduced_param_set):
    mgs_aspects = [
        np.arange(0,16),
        np.arange(16,21),
        np.arange(21,37),
        np.arange(37,45),
        np.arange(45,50),
        np.arange(50,55)
    ]
    partial_scores = []

    for aspect in mgs_aspects:
        s_partial = 0
        params = 0
        for i in aspect:
            if(i in reduced_param_set):
                z_score = (indiv_full_gait_params[i] - np.mean(control_mgs_params[:,i])) / np.std(control_mgs_params[:,i])
                norm_score = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
                params = params + 1
                s_partial = s_partial + norm_score

        s_partial = (10 / params) * s_partial
        partial_scores.append(s_partial)
        
    return list(partial_scores)