# functions for calculating different gait scores
import numpy as np
from tslearn.metrics import dtw_path
from minisom import MiniSom
import matplotlib.pyplot as plt

'''
Calculates the Gait Profile Score (GPS) as outlined by Baker et al. (https://pubmed.ncbi.nlm.nih.gov/19632117/)
Inputs:
    part_kinematics: kinematics dictionary for individual participant. Pelvic orientation all 3 axes (unilateral) and right/left
        hip angles (all 3 planes), knee flexion, ankle flexion, and foot progression = 15 signals
        Current calculation excludes foot progression
    control_kinematics: equivalent dictionary for control dataset
    
    both part_kinmatics and control_kinematics should be reshaped to be 2% increments of the gait cycle. This leads to 51
    data points (eg. HS to subsequent HS)


'''
def calc_gait_profile_score(part_kinematics, control_kinematics):
    # organize kinematics into single array
    def create_kin_profile(kinematics_dict):
        kinematics_arr = []
        kinematics_arr.append(np.mean(part_kinematics['pelvis_orient'], axis=0).tranpose())
        kinematics_arr.append(np.mean(part_kinematics['hip_angle'][0], axis = 0).tranpose())
        kinematics_arr.append(np.mean(part_kinematics['hip_angle'][1], axis = 0).tranpose())
        kinematics_arr.append(np.expand_dims(np.mean(part_kinematics['knee_angle'][0,:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(part_kinematics['knee_angle'][1,:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(part_kinematics['ankle_angle'][0,:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(part_kinematics['ankle_angle'][1,:,:,2], axis = 0), axis=0))
        
        kinematics_arr = np.concatenate(kinematics_arr, axis=0)
        return kinematics_arr
    
    
    # mean kinematics arrays. Should be shape [Number of signals, 51 time points]
    part_mean_kinematics = create_kin_profile(part_kinematics)
    control_mean_kinematics = create_kin_profile(control_kinematics)
    
    
    gait_variability_score = [np.linalg.norm(part_mean_kinematics[i] - control_mean_kinematics[i]) / np.sqrt(len(part_mean_kinematics[i]))
                                 for i in range(len(part_mean_kinematics))]
    
    gait_profile_score = np.linalg.norm(gait_variability_score) / np.sqrt(len(gait_variability_score))
    
    return gait_profile_score


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
    control_data (numpy array): The input data for training the SOM. Dimensions NxD, where D is the dimension of the data 
    learning_rate (float): The initial learning rate for the SOM.
    topology (str): The topology of the SOM ('rectangular' or 'hexagonal').
Returns:
    som: Trained SOM
"""

def train_minisom(control_data, learning_rate=0.1, topology='hexagonal'):
 
    # Parameters for training/initializing: 
    dim = int(np.sqrt(5 * np.sqrt(control_data.shape[0]))) #Heuristic: # map units = 5*sqrt(n) [1]
    som_dim = (dim, dim)
    steps = 500 * (dim ** 2) #Number of iterations - Heuristic: 500 * number of network units [2] 
    sigma = dim/4 #Sigma used: Heuristic: max(msize)/4 [1]
    
    som = MiniSom(x=som_dim[0], y=som_dim[1], input_len=control_data.shape[1], sigma=sigma, learning_rate=learning_rate, topology=topology) #Initializing the SOM 
    
    som.random_weights_init(control_data) #Initializes the weights of the SOM picking random samples from data.
    som.train_batch(control_data, steps) #Trains the SOM using all the vectors in data sequentially.
    
    """    
    Resources
    [1]	    Juha. Vesanto and (Libella), SOM toolbox for Matlab 5. Helsinki University of Technology, 2000.
    [2]	    S. Lek and Y. S. Park, “Artificial Neural Networks,” Encyclopedia of Ecology, Five-Volume Set, vol. 1–5, pp. 237–245, Jan. 2008, doi: 10.1016/B978-008045405-4.00173-7.
    
    """
    return som

"""    
Calculates Mean MDP scores for the given data and trained SOM.
Parameters:
    data (numpy array): The input data for which Mean MDP scores are calculated.
    som (MiniSom): The trained SOM.
Returns:
    deviation: Euclidian distance between BMU and dataset for each time point -- used to generate MDP
"""

def calculate_MDP(data, som):
    winners = np.array([som.winner(instance) for instance in data])  # Finds the best matching unit (BMU) for all time points in the data
    BMU = som._weights[winners[:, 0], winners[:, 1]]  # Collects the corresponding weight vectors for all BMUs
    deviation = np.linalg.norm(data - BMU, axis=1)  # Calculate Euclidean distances in the full dataset
    return deviation
