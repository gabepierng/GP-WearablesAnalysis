# functions for calculating different gait scores
import numpy as np
from tslearn.metrics import dtw_path
from minisom import MiniSom
import matplotlib.pyplot as plt
import scipy.linalg as la
import math

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
    print("Desired dimensions: %d" % dim)
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
    print("Map size: %d x %d" % (msize[0], msize[1]))
    print("Number of map units: ", x*y)
    
    #steps = 500 * (dim ** 2) # Number of iterations - previous heuristic: 500 * number of network units [2] 
    # steps = math.ceil(10*(x*y)/gait_data.shape[0]) #Based on .trainlen = 10*m/n (m is # map units, n is the number of training samples)
    steps = desired_steps
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

def calc_MDP(data, som_train_data, som, normalize=True):
    
    normalized_gait_data = None
    if normalize: # normalize the test data to scale of SOM trained on normalized gait data
        means = np.mean(som_train_data, axis=0, keepdims=True)
        mean_centered_array = data - means
        std_devs = np.std(mean_centered_array, axis=0, ddof=0, keepdims=True) #Along each column 
        normalized_test_data = mean_centered_array / std_devs
    
    bmu_vector = np.array([som._weights[som.winner(instance)] for instance in data])
    denormed_bmu_vector = (bmu_vector * std_devs) + means
    deviation = np.linalg.norm(data - denormed_bmu_vector, axis=1)  # Calculate Euclidean distances in the full dataset
    return deviation


