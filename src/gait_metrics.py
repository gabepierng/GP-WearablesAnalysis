# functions for calculating different gait scores
import numpy as np

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
    def create_kin_profile(kinematics_dict)
        kinematics_arr = []
        kinematics_arr.append(np.mean(part_kinematics['pelvis_orient'], axis=0).tranpose())
        kinematics_arr.append(np.mean(part_kinematics['hip_angle'][0], axis = 0).tranpose())
        kinematics_arr.append(np.mean(part_kinematics['hip_angle'][1], axis = 0).tranpose())
        kinematics_arr.append(np.expand_dims(np.mean(part_kinemaics['knee_angle'][0,:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(part_kinemaics['knee_angle'][1,:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(part_kinemaics['ankle_angle'][0,:,:,2], axis = 0), axis=0))
        kinematics_arr.append(np.expand_dims(np.mean(part_kinemaics['ankle_angle'][1,:,:,2], axis = 0), axis=0))
        
        kinematics_arr = np.concatenate(kinematics_arr, axis=0)
        return kinematics_arr
    
    
    # mean kinematics arrays. Should be shape [Number of signals, 51 time points]
    part_mean_kinematics = create_kin_profile(part_kinematics)
    control_mean_kinematics = creat_kin_profile(control_kinematics)
    
    
    gait_variability_score = [np.linalg.norm(part_mean_kinmatics[i] - control_mean_kinematics[i]) / np.sqrt(len(part_mean_kinematics[i]))
                                 for i in range(len(part_mean_kinematics))]
    
    gait_profile_score = np.linalg.norm(gait_variability_score) / np.sqrt(len(gait_variability_score))
    
    return gait_profile_score
    