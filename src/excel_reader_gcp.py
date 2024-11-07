# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:21:22 2019

@author: BIEL
"""

'''-------------------------------------------------------
The class has different attributes: 
                 - xml_file gives you the full xml fil as an untangle element
                 - mvn_version and mvn_build are info of the MVN software 
                 - available_info gives you a list of all the parameters that you have exported 
                   (position, orientation,...)
                 - all_index is an object that gives you the labels for the data,
                   depending if the data is from sensors, joints, segments or others;
                 - total_frames returns the number of frames captured by xsense,
                 - subject is a dictionary with the capture information
                 - frames is the most important one. It's a list of untangle elements
                   each one containg all info (position, orientation) of a frame
--------------------------------------------------------
 Data may be presented diferently depending of what information you extract
 For example: position gives a dictionary with 23 labels, each one containing
 a 1x3 list (corresponding to a xyz point). In the other hand, sensor orientations
 gives a dictionary with 17 labels (different ones), each one containing a 1x4 list.

 For more information, I recommend checking pages 95-96 in MVNX manual
'''

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import math
import time
import pandas as pd
import sys

import scipy.interpolate as interp
import scipy
import sklearn.metrics as metrics

# plot_colors = ['black', 'red', 'blue', 'green']
plot_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey']
sides = ['Right', 'Left']
joint_planes = ['Abd/Add', 'Int/Ext Rot', 'Flex/Ext']
axis_planes = ['x', 'y', 'z']
liveplot = False

def get_default_sensor_index():
    index=['t8','pelvis',
               'right_upper_leg','right_lower_leg','right_foot',
               'left_upper_leg','left_lower_leg','left_foot']
    return index

def centroid_poly(X, Y):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    N = len(X)
    # minimal sanity check
    if not (N == len(Y)): raise ValueError('X and Y must be same length.')
    elif N < 3: raise ValueError('At least 3 vertices must be passed.')
    sum_A, sum_Cx, sum_Cy = 0, 0, 0
    last_iteration = N-1
    # from 0 to N-1
    for i in range(N):
        if i != last_iteration:
            shoelace = X[i]*Y[i+1] - X[i+1]*Y[i]
            sum_A  += shoelace
            sum_Cx += (X[i] + X[i+1]) * shoelace
            sum_Cy += (Y[i] + Y[i+1]) * shoelace
        else:
            # N-1 case (last iteration): substitute i+1 -> 0
            shoelace = X[i]*Y[0] - X[0]*Y[i]
            sum_A  += shoelace
            sum_Cx += (X[i] + X[0]) * shoelace
            sum_Cy += (Y[i] + Y[0]) * shoelace
    A  = 0.5 * sum_A
    factor = 1 / (6*A)
    Cx = factor * sum_Cx
    Cy = factor * sum_Cy
    # returning abs of A is the only difference to
    # the algo from above link
    return Cx, Cy, abs(A)

'''
calculate mean of collection of strides from trial/trial_type and upper and lower confidence intervals
Input = 2d numpy array, each row is a stride
Output = 3 signals: mean, upper and lower bound
'''
def trial_avg_and_CI(signal_set):
    conf_int_mult = 1.00    # confidence interval multiplier for 1 std

    avg_signal = np.mean(signal_set, axis=0)
    std_signal = np.std(signal_set, axis=0)
    upper_bound = avg_signal + (conf_int_mult * std_signal)
    lower_bound = avg_signal - (conf_int_mult * std_signal)
    avg_variance = np.mean(std_signal)
    print('Mean standard deviation: %.4f' % avg_variance)

    return avg_signal, upper_bound, lower_bound

'''
visualize strides in plot
Input:  strides = list of strides to plot (if confidence plot, must be 2d reshaped numpy array of strides)
        fix_ax = axis of figure to plot. Typically ax1 or ax2 for plotting R/L signals
        signal_axis = which axis of signal to plot (0:X, 1:Y, 2:Z)
'''
def visualize_strides(strides, fig_ax, signal_axis, trial_num, confidence_plot = True):

    if(confidence_plot):
#         plot_signals = trial_avg_and_CI(strides[:,:,signal_axis])
        plot_signals = trial_avg_and_CI(strides[:,:])
        x = np.linspace(0, 100, strides.shape[1])
        fig_ax.plot(x, plot_signals[0], color=plot_colors[trial_num])
        fig_ax.fill_between(x, plot_signals[1], plot_signals[2], color=plot_colors[trial_num], alpha=0.2)

    else:
        plt.ylabel('Degrees (º)')

        for stride in strides:

            fig_ax.plot(np.linspace(0,100,len(stride)), stride[:,signal_axis], color = plot_colors[int(trial_num / 1)])
            # ax1.plot(stride, color = plot_colors[trial_val])
            if(liveplot):
                plt.draw()
                plt.pause(0.001)

    # plt.show()


'''
Gait parser object. Main function is to store info pertaining to Xsens file, as well as MVN gait information

init parameters:
    mvn_csv_filename: filename of .csv file containing desired mvnx data
'''
class XsensGaitDataParser:

    def __init__(self):
        self.mvn_filename = None
        # following variables initialized in process_mvn_trial_data
        self.gait_events = []             # list with right and left gait events, organized into matched gait cycles
        self.gait_params = {}             # dict. containing spatiotemporal, kinematics, and signal gait parameters
        self.mvnxData = None
        self.partitioned_mvn_data = {}
        self.partitioned_dot_data = {}

    def clear_gait_data(self):
        self.gait_events = []
        self.gait_params = {}
        self.mvnxData = None
        self.partitioned_mvn_data = {}
        self.partitioned_dot_data = {}


    '''
    Cleans heel_strikes and returns start-times/end-times for valid gait cycles as well as Toe Off events
    Input:  position_not_ends = frames within allowable position
            heel_contacts = raw identified heel contact information
            toe_contacts = raw identified toe contact information
            seq: sequence to search for when finding gait events

    output is
            gait_events = 3 x N matrix with start of gait cycles (HS), toe off, and end of cycle HS
    '''
    def identify_gait_events(self, position_not_ends, heel_contacts, toe_contacts, seq):
        # searches for sequences of [0,1] to indicate heel strikes within heel contact, returns index values
        # exclude gait events at either end of walkway (i.e. not steady state/possibly turning around)
        heel_strikes = np.array([i+1 for i in range(len(heel_contacts)) 
                            if (np.array_equal(heel_contacts[i:i+len(seq)], seq)) and (i in position_not_ends) and (toe_contacts[i-10] == 0)])

        toe_offs = np.array([i+1 for i in range(len(toe_contacts)) 
                            if (np.array_equal(toe_contacts[i:i+len(seq)], [1-j for j in seq])) and (i in position_not_ends)])

        
        # remove double detections in close succession, choose 1st for HS
        heel_strikes = np.delete(heel_strikes, 1 + np.where(np.diff(heel_strikes)<70)[0]) 

        # includes large diff of HSs between passes (i.e. last included HS of one pass to first included HS of next pass)
        # number will be firmly between the actual mean difference of gait cycles and the between-pass difference.
        # used when making gait_events array with gait cycles to ignore the between passes HS-HS
        mean_dif_heel_strikes = np.mean(np.diff(heel_strikes))
        # print(mean_dif_heel_strikes)

        gait_events = []

        # gait_events initially Nx3 matrix, each row is gait cycles HS-TO-subsequent HS
        for i in range(len(heel_strikes) - 1):
            if(heel_strikes[i+1] - heel_strikes[i] < (1.2 * mean_dif_heel_strikes)):

                # pick TO detected just before the terminal HS/end-cycle HS
                temp_array = toe_offs - heel_strikes[i+1]
                toe_off = toe_offs[np.where(temp_array < 0, temp_array, -np.inf).argmax()]
                if(toe_off > heel_strikes[i]):
                    gait_events.append([heel_strikes[i], toe_off, heel_strikes[i+1]])

        # gait events: 3 x N matrix. Rows = HS at beginning of cycle, TO, HS at end of cycle
        # N = number of valid HS-HS gait cycles detected
        gait_events = np.transpose(np.asarray(gait_events))

        return gait_events


    '''
    calculate desired spatiotemporal parameters
    Input:  gait_events = HS (gait cycle) and TO events, pelvis position vector, 
                            foot position vector, and frame rate MVN samples at for trial
    Output: Spatiotemporal parameters
    '''
    def calc_spatio_temp_params(self, gait_events, pelvis_position, foot_position, frame_rate, knee_angles, hip_angles, foot_velocity, lower_leg_orientation):
        # inputs are Nx3 vectors with 3D position coordinates
        def calc_distances(positions1, positions2):
            # can generally ignore z-axis, remove before doing calculations
            positions1 = np.delete(positions1, -1, axis=1)
            positions2 = np.delete(positions2, -1, axis=1)
            distances = np.sqrt(np.sum(np.power(positions1 - positions2, 2), axis=1))
            return distances

        '''
        calculate parameters related to step which require slightly different math since
        # of steps for L and R side not necessarily equal (total or for any given pass
        along the walk area)

        This has been amended in recent code where L and R gait cycles are matched
        '''
        def calc_step_params(stride_time):
            r_to_l_steps = []
            l_to_r_steps = []
            r_to_l_strides = []

            # looks at valid Heel_Strike_foot1 to Heel_Strike_foot2 for left to right and vice versa
            for i in range(r_gait_events.shape[1]):
                valid_step_index = [t for t,x in enumerate(l_gait_events[0,:]) if (x > r_gait_events[0,i] and x < r_gait_events[2,i])]
                valid_step = l_gait_events[0, valid_step_index]
                if (len(valid_step) > 0):
                    r_to_l_steps.append([r_gait_events[0,i], valid_step[0], r_gait_events[1,i]])
                    r_to_l_strides.append([r_gait_events[0,i], r_gait_events[1,i], r_gait_events[2,i], l_gait_events[0,valid_step_index][0], l_gait_events[1, valid_step_index][0], l_gait_events[2, valid_step_index][0]])
            r_to_l_strides = np.array(r_to_l_strides)
            for i in range(l_gait_events.shape[1]):
                valid_step = r_gait_events[0,(r_gait_events[0,:] > l_gait_events[0,i]) &
                    (r_gait_events[0,:] < l_gait_events[2,i])]
                if (len(valid_step) > 0):
                    l_to_r_steps.append([l_gait_events[0,i], valid_step[0], l_gait_events[1,i]])

            r_to_l_steps = np.array(r_to_l_steps).T
            # l_to_r_steps = np.array(l_to_r_steps).T
            # l_to_r_steps = np.array(l_to_r_steps).T
            
            r_to_l_strides = np.array(r_to_l_strides).T

            # double stance double_stance_support as percent of gait cycle
            r_dbl_stnc_sprt = (r_to_l_steps[2,:] - r_to_l_steps[1,:]) / (stride_time[0] * frame_rate)
            # l_dbl_stnc_sprt = np.mean(l_to_r_steps[2,:] - l_to_r_steps[1,:]) / (stride_time[1] * frame_rate)
            # double_stance_support = [r_dbl_stnc_sprt, l_dbl_stnc_sprt]
            double_stance_support = np.array([r_dbl_stnc_sprt, r_dbl_stnc_sprt])
            r_dbl_stnc_sprt = (r_to_l_steps[2,:] - r_to_l_steps[1,:]) / (stride_time[0] * frame_rate)
            # l_dbl_stnc_sprt = np.mean(l_to_r_steps[2,:] - l_to_r_steps[1,:]) / (stride_time[1] * frame_rate)
            double_stance_support = [r_dbl_stnc_sprt, r_dbl_stnc_sprt]
            # double_stance_support = r_dbl_stnc_sprt
            
            r_to_l_step_lengths = calc_distances(foot_position[0, r_to_l_steps[0,:]],
                                                        foot_position[1, r_to_l_steps[1,:]])

            step_lengths = np.array([calc_distances(foot_position[0, r_to_l_strides[2,:]],
                                            foot_position[1, r_to_l_strides[3,:]]),
                                   calc_distances(foot_position[0, r_to_l_strides[0,:]],
                                            foot_position[1, r_to_l_strides[3,:]])])
            
            
            step_length_SR = step_lengths[0] / step_lengths[1]
            
            stride_lengths = np.array([calc_distances(foot_position[0, r_to_l_strides[0,:]],
                                            foot_position[0, r_to_l_strides[2,:]]),
                                   calc_distances(foot_position[1, r_to_l_strides[3,:]],
                                            foot_position[1, r_to_l_strides[5,:]])])
            
            stride_length_SR = stride_lengths[0] / stride_lengths[1]
            
            stance_times = [[stride[1] - stride[0] for stride in r_to_l_strides.T], 
                            [stride[4] - stride[3] for stride in r_to_l_strides.T]]
            stance_time_ratio = [(stride[1]-stride[0]) / (stride[4]-stride[3]) for stride in r_to_l_strides.T]
            
            # not spatiotemporal, but added these in here to calculate kinematics matched to the relevant gait cycle. Should move eventually
            knee_ROMs = np.array([np.array([np.max(knee_angles[0][stride[0]:stride[2]], axis=0) - np.min(knee_angles[0][stride[0]:stride[2]], axis=0),
                            np.max(knee_angles[1][stride[3]:stride[5]], axis=0) - np.min(knee_angles[1][stride[3]:stride[5]], axis=0)]) for stride in r_to_l_strides.T])
            
            knee_ROM_SR = np.array([stride[0] / stride[1] for stride in knee_ROMs])
            
            hip_ROMs = np.array([np.array([np.max(hip_angles[0][stride[0]:stride[2]], axis=0) - np.min(hip_angles[0][stride[0]:stride[2]], axis=0),
                            np.max(hip_angles[1][stride[3]:stride[5]], axis=0) - np.min(hip_angles[1][stride[3]:stride[5]], axis=0)]) for stride in r_to_l_strides.T])
            

            # return [double_stance_support, step_length_avg, step_length_std, step_lengths, stride_length_SR, stance_time_ratio, knee_ROMs, knee_ROM_SR, hip_ROMs, stride_lengths]
            return [double_stance_support, step_lengths, stride_length_SR, stance_time_ratio, knee_ROMs, knee_ROM_SR, hip_ROMs, stride_lengths]

        r_gait_events = gait_events[0]
        l_gait_events = gait_events[1]

        
        gait_cycle_time_mean = [np.mean((r_gait_events[2,:] - r_gait_events[0,:]) / frame_rate),
                                   np.mean((l_gait_events[2,:] - l_gait_events[0,:]) / frame_rate)]
        # calculate non-step gait parameters (easier calculations)
        # frames from HS to TO same foot
        stance_times = np.array([(r_gait_events[1,:] - r_gait_events[0,:]) / frame_rate,
                        (l_gait_events[1,:] - l_gait_events[0,:]) / frame_rate])

        stance_time_avg = [np.mean(stance_times[0]) / gait_cycle_time_mean[0], 
                        np.mean(stance_times[1]) / gait_cycle_time_mean[1]]

        stance_time_std = [np.std(stance_times[0], ddof=1),
                            np.std(stance_times[1], ddof=1)]

        # frames from TO to next HS same foot
        swing_times = np.array([(r_gait_events[2,:] - r_gait_events[1,:]) / frame_rate,
                        (l_gait_events[2,:] - l_gait_events[1,:]) / frame_rate])
        
        swing_time_avg = [np.mean(swing_times[0]),
                        np.mean(swing_times[1])]

        swing_time_std = [np.std(swing_times[0], ddof=1),
                            np.std(swing_times[1], ddof=1)]
                    

        stance_time_ratio = stance_time_avg[0] / stance_time_avg[1]
        swing_time_ratio = swing_time_avg[0] / swing_time_avg[1]

        stride_times = np.array([(r_gait_events[2,:] - r_gait_events[0,:]) / frame_rate,
                        (l_gait_events[2,:] - l_gait_events[0,:]) / frame_rate])
        
        
        step_times = np.array([(r_gait_events[2,:] - l_gait_events[0,:]) / frame_rate,
                        (l_gait_events[0,:] - r_gait_events[0,:]) / frame_rate])

        stride_time_avg = [np.mean(stride_times[0]),
                            np.mean(stride_times[1])]

        stride_time_std = [np.std(stride_times[0], ddof=1),
                            np.std(stride_times[1], ddof=1)]

        # foot_positions [row 0] = right foot, [row 1] = left foot
        stride_lengths = [(calc_distances(foot_position[0, r_gait_events[0,:]],
                                                    foot_position[0, r_gait_events[2,:]])),
                        (calc_distances(foot_position[1, l_gait_events[0,:]],
                                                    foot_position[1, l_gait_events[2,:]]))]

        stride_length_avg = [np.mean(stride_lengths[0]),
                            np.mean(stride_lengths[1])]

        stride_length_std = [np.std(stride_lengths[0], ddof=1),
                            np.std(stride_lengths[1], ddof=1)]

        # could technically use any of r_stride_length, l_stride_length, or pelvis
        # position to calculate speed, using pelvis just because it has most consistent movement

        # calculate speed in m/s
        ind_dist_walked = calc_distances(pelvis_position[r_gait_events[0,:]],
                                            pelvis_position[r_gait_events[2,:]])
        avg_dist_walked = np.mean(ind_dist_walked)

        # dist_traversed = calc_distances(pelvis_position[r_gait_events[0,:]],
        #                                         pelvis_position[r_gait_events[2,:]])
        # walk_time = (r_gait_events[2,:] - r_gait_events[0,:]) / frame_rate

        ind_walk_time = (r_gait_events[2,:] - r_gait_events[0,:]) / frame_rate
        avg_walk_time = np.mean(ind_walk_time)

        # instantaneous_speed = dist_traversed / walk_time
        speed = avg_dist_walked / avg_walk_time
        ind_speed = np.divide(ind_dist_walked, ind_walk_time)

        # cadence in steps/min, doubled to get left and right steps
        cadence = (2 / avg_walk_time) * 60 # sec/min
        ind_cadence = (2 / ind_walk_time) * 60

        step_based_params = calc_step_params(stride_time_avg)
        # double_stance_support = step_based_params[0]
        step_lengths = step_based_params[1]
        stride_length_SR = step_based_params[2]
        stance_time_ratio_ind = step_based_params[3]
        knee_ROMs = step_based_params[4]
        knee_ROM_SR = step_based_params[5]
        hip_ROMs = step_based_params[6]
        stride_lengths = step_based_params[7]
        
        double_stance_support = np.array( [(r_gait_events[1,:] - l_gait_events[0,:]) / (stride_times[0] * frame_rate),
                                           (l_gait_events[1,:] - r_gait_events[2,:]) / (stride_times[1] * frame_rate)] )
        
        # IMU Gait Normalcy Index gait parameters (https://ieeexplore.ieee.org/document/9049129)
        swing_phase_percent = np.array([swing_times[0]/stride_times[0], swing_times[1]/stride_times[1]])
        max_forward_velocity = []
        max_vert_ank_disp = []
        step_length_at_max_vert = []
        max_ank_lat_displace = []
        max_ank_med_displace = []
        shank_sagit_rom = []
        
        for i in range(len(gait_events)):
            max_forward_velocity.append([])
            max_vert_ank_disp.append([])
            step_length_at_max_vert.append([])
            max_ank_lat_displace.append([])
            max_ank_med_displace.append([])
            shank_sagit_rom.append([])
            
            for j in range(gait_events[i].shape[1]):
                start_gait_cycle = gait_events[i][0,j]
                swing_phase_start = gait_events[i][1,j]
                end_gait_cycle = gait_events[i][2,j]
                max_forward_velocity[i].append( np.max(foot_velocity[i, start_gait_cycle:end_gait_cycle ,0]) )
                max_vert_ank_disp[i].append( np.max(foot_position[i, start_gait_cycle:end_gait_cycle ,2]) )
                time_of_max_vert_ank = np.argmax(foot_position[i, start_gait_cycle:end_gait_cycle ,2])
                step_length_at_max_vert[i].append(foot_position[i, time_of_max_vert_ank + start_gait_cycle, 0] - foot_position[i, start_gait_cycle, 0])
                # accounts for whether positive of negative y-direction desired, since direction of medical and lateral in reference to global frame
                # changes whether using right or left leg.
                if(i == 0):
                    time_max_ank_lat = np.argmin(foot_position[i, start_gait_cycle:end_gait_cycle, 1])
                    time_max_ank_med = np.argmax(foot_position[i, start_gait_cycle:end_gait_cycle, 1])
                    max_ank_lat_displace[i].append( foot_position[i, time_max_ank_lat + start_gait_cycle, 1] 
                                                   - pelvis_position[time_max_ank_lat + start_gait_cycle, 1] )
                    max_ank_med_displace[i].append( foot_position[i, time_max_ank_med + start_gait_cycle, 1] 
                                                   - pelvis_position[time_max_ank_med + start_gait_cycle, 1] )
                else:
                    time_max_ank_lat = np.argmax(foot_position[i, start_gait_cycle:end_gait_cycle ,1])
                    time_max_ank_med = np.argmin(foot_position[i, start_gait_cycle:end_gait_cycle ,1])
                    max_ank_lat_displace[i].append( foot_position[i, time_max_ank_lat + start_gait_cycle, 1] 
                                                   - pelvis_position[time_max_ank_lat + start_gait_cycle, 1] )
                    max_ank_med_displace[i].append( foot_position[i, time_max_ank_med + start_gait_cycle, 1] 
                                                   - pelvis_position[time_max_ank_med + start_gait_cycle, 1] )
                
                max_shank_orient = np.max( lower_leg_orientation[i, swing_phase_start:end_gait_cycle, 1] )
                min_shank_orient = np.min( lower_leg_orientation[i, swing_phase_start:end_gait_cycle, 1] )
                shank_sagit_rom[i].append(max_shank_orient - min_shank_orient)
        
        max_forward_velocity = np.array(max_forward_velocity)
        max_vert_ank_disp = np.array(max_vert_ank_disp)
        step_length_at_max_vert = np.array(step_length_at_max_vert)
        max_ank_lat_displace = np.array(max_ank_lat_displace)
        max_ank_med_displace = np.array(max_ank_med_displace)
        shank_sagit_rom = np.array(shank_sagit_rom)
        
        spatio_temp_params = [stride_times, stride_lengths, swing_phase_percent, max_forward_velocity, max_vert_ank_disp, step_length_at_max_vert,
                            max_ank_lat_displace, max_ank_med_displace, shank_sagit_rom, stance_times, swing_times, double_stance_support, step_times]

        return spatio_temp_params

    '''
    calculate desired kinematic parameters
    Input:  gait_events = contains 2 lists for r and l gait events
                            HS (gait cycle) and TO events
            pelvis_orientation = xyz orientation of pelvis
            hip, knee, ankle _angles = undivided angle xyz angle information

    Output: Kinematic parameters (in dictionary)
    '''
    def calc_kinematic_params(self, gait_events, pelvis_orientation, hip_angles, knee_angles, ankle_angles, sternum_orientation):

        # returns stance/swing time min/max angles for each side, xyz axes
        def calc_ROM(signal):
            stance_ROM = []
            swing_ROM = []
            gait_cycle_ROM = []
            for i, side in enumerate(signal):

                gait_cycle_maxes = np.array([np.max(side[gait_events[i][0,j]:gait_events[i][2,j]], axis=0) 
                                            for j in range(gait_events[i].shape[1])])
        
                gait_cycle_mins = np.array([np.min(side[gait_events[i][0,j]:gait_events[i][2,j]], axis=0) 
                                            for j in range(gait_events[i].shape[1])])
                gait_cycle_ROM.append(np.expand_dims(np.mean(gait_cycle_maxes, axis=0) - np.mean(gait_cycle_mins, axis=0), axis=0))

            return gait_cycle_ROM

        kinematic_params = {}
        kinematic_params['pelvis_ROM'] = calc_ROM(pelvis_orientation)
        kinematic_params['sternum_ROM'] = calc_ROM(sternum_orientation)
        kinematic_params['hip_ROM'] = calc_ROM(hip_angles)
        kinematic_params['knee_ROM'] = calc_ROM(knee_angles)
        kinematic_params['ankle_ROM'] = calc_ROM(ankle_angles)

    #     print(kinematic_params['knee_ROM'])

        return kinematic_params

    '''
    calculate signal-based parameters
    Input:  lower_body_strides = dictionary with pelvis Euler orientation, hip angles, 
                            knee angles, and ankle angles. All but 1st = right and left sides

    Output: Signal-based parameters
    '''
    def calc_signal_params(self, lower_body_strides):
        
        def calc_RMS(signal):
            RMS = []
            for side in signal:
                total_signal = np.concatenate(side)
                RMS.append(np.sqrt(np.mean(total_signal ** 2, axis=0)))
            return RMS

        def calc_variance(signal):
            variance = []
            for side in signal:
                total_signal = np.concatenate(side)
                variance.append((np.mean(total_signal**2, axis=0) - (np.mean(total_signal, axis=0)**2)))

            return variance

        signal_params = {}
        for segment in lower_body_strides:
            signal_params[segment] = [calc_RMS(lower_body_strides[segment]), 
                                        calc_variance(lower_body_strides[segment])]

        return signal_params

    '''
    divide signal into gait cycles based on gait events
    Input:  gait events for left and right, signal to be divided into gait cycles
    '''
    def divide_into_strides(self, gait_events, signal, side, signal_num):
        if(side == 'r'):
            strides = [np.array(signal[signal_num, gait_events[0][0,i]:gait_events[0][2,i]]) 
                                    for i in range(gait_events[0].shape[1])]
        else:
            strides = [np.array(signal[signal_num, gait_events[1][0,i]:gait_events[1][2,i]]) 
                                    for i in range(gait_events[1].shape[1])]

        return strides


    '''
    Get various kinematic, movement, and positional signals from the .csv file
    '''
    def get_pelvis_euler_orientation(self):
        return np.expand_dims(np.transpose([self.mvnxData['Pelvis Orient ' + plane] for plane in axis_planes]), axis=0)

    def get_sternum_euler_orientation(self):
        return np.expand_dims(np.transpose([self.mvnxData['Sternum Orient ' + plane] for plane in axis_planes]), axis=0)
    
    def get_lower_leg_orientation(self):
        return np.array([ np.transpose([self.mvnxData['Right Lower Leg Orient ' + plane] for plane in axis_planes]),
                np.transpose([self.mvnxData['Left Lower Leg Orient ' + plane] for plane in axis_planes]) ])
    
    def get_lower_leg_velocity(self):
        return np.array([ np.transpose([self.mvnxData['Right Lower Leg Velocity ' + plane] for plane in axis_planes]),
                np.transpose([self.mvnxData['Left Lower Leg Velocity ' + plane] for plane in axis_planes]) ])
    
    def get_foot_velocity(self):
        return np.array([ np.transpose([self.mvnxData['Right Foot Velocity ' + plane] for plane in axis_planes]),
                np.transpose([self.mvnxData['Left Foot Velocity ' + plane] for plane in axis_planes]) ])

    def get_pelvis_position(self):
        return np.transpose([self.mvnxData['Pelvis Position ' + plane] for plane in axis_planes])

    def get_foot_position(self):
        return np.array([ np.transpose([self.mvnxData['Right FootPosition ' + plane] for plane in axis_planes]),
                                np.transpose([self.mvnxData['Left FootPosition ' + plane] for plane in axis_planes]) ])

    def get_heel_contacts(self):
        return np.array([self.mvnxData[side + ' HeelContacts'] for side in sides])

    def get_toe_contacts(self):
        return np.array([self.mvnxData[side + ' ToeContacts'] for side in sides])

    def get_hip_angles(self):
        return np.array([ np.transpose([self.mvnxData['Right Hip ' + plane] for plane in joint_planes]),
                                    np.transpose([self.mvnxData['Left Hip ' + plane] for plane in joint_planes]) ])

    def get_knee_angles(self):
        return np.array([ np.transpose([self.mvnxData['Right Knee ' + plane] for plane in joint_planes]),
                                    np.transpose([self.mvnxData['Left Knee ' + plane] for plane in joint_planes]) ])

    def get_ankle_angles(self):
        return np.array([ np.transpose([self.mvnxData['Right Ankle ' + plane] for plane in joint_planes]),
                                    np.transpose([self.mvnxData['Left Ankle ' + plane] for plane in joint_planes]) ])

    def get_gyro_data(self, sensors):
        return np.array([np.transpose([self.mvnxData[signal] for signal in self.mvnxData.columns if (sensor + '_gyro') in signal]) for sensor in sensors])

    def get_acc_data(self, sensors):
        return np.array([np.transpose([self.mvnxData[signal] for signal in self.mvnxData.columns if (sensor + '_acc') in signal]) for sensor in sensors])

    def get_gait_param_info(self):
        if(self.gait_params):
            return self.gait_params
        else:
            raise ValueError('Gait parameters dictionary empty, call process_mvn_trial_data() first')

    def get_gait_events(self):
        if(self.gait_events):
            return self.gait_events
        else:
            raise ValueError('Gait events dictionary empty, call process_mvn_trial_data() first')

    def get_partitioned_mvn_data(self):
        if(self.partitioned_mvn_data):
            return self.partitioned_mvn_data
        else:
            raise ValueError('Partitioned dot data dictionary empty, call time_align_mvn_and_dot() first')

    '''
    If there is corresponding Xsens DOT data, this function splits the Xsens DOT data into gait cycles, using gait events identified
    by the MVN system data
    Input:  mvn_gyro_data: Gyroscope data from the MVN system
            xsens_dot_data: Xsens DOT data to be partitioned, expects a dictionary with structure {sensor-locations: N x 6 numpy array}
                                                                                                    N = number of packets
                            Each key corresponds to sensor location ('ur', 'p', etc.)
            gait_events: gait events from MVN contact information. 2 elements of list --> right and left gait events per gait cycle
    Output:
            partitioned_dot_data: 
    '''
    def time_align_mvn_and_dot(self, xsens_dot_data, is_dot_data_time_synced=True):

        if ((self.mvnxData is None) or (not self.gait_events)):
            raise ValueError('Need to read MVN data first, call process_mvn_trial_data()')

        sensors = get_default_sensor_index()
        mvn_gyro_data = self.get_gyro_data(sensors)
        # main gyroscope axes for 5 signal locations (based on CZ sensor placement/orientation), used for time-aligning with xsens mvn
        dot_main_axis = {'LowerL':-1, 'LowerR':-1, 'Pelvis':-2, 'UpperL':-1, 'UpperR':-1}
        # because of orientation, need to flip Pelvis and UpperR signals for time-aligning
        dot_flip_axis = {'LowerL':1, 'LowerR':1, 'Pelvis':1, 'UpperL':1, 'UpperR':-1}
        dot_xsens_sensor_loc = {'LowerL':6, 'LowerR':3, 'Pelvis':1, 'UpperL':5, 'UpperR':2}
        partitioned_dot_data = {}
        sensor_locs = {'ll':'LowerL', 'lr':'LowerR', 'p':'Pelvis', 'ul':'UpperL', 'ur':'UpperR'}
        b20, a20 = scipy.signal.butter(N=2, Wn = 0.8, btype = 'lowpass')
        
        # once get an initial time-alignment based off gait events, finetune by minimizing the RMSE
        # between DOT and MVN data. This fixes issues where slight differences in filtering or sensor placement
        # could lead to slightly different peaks, but overall shapes are the same.
        def finetune_dot_alignment(dot_time_signal, xsens_time_signal, initial_time_align, dot_peaks):
            dot_start = dot_peaks[0] 
            dot_end = dot_peaks[6]
            dot_compare = dot_time_signal[dot_start : dot_end]
            
            xsens_start = dot_start - initial_time_align
            xsens_end = dot_end - initial_time_align
            
            min_rmse = 99999999
            new_align = initial_time_align
            for i in range(-19, 20):
                xsens_compare = xsens_time_signal[xsens_start + i : xsens_end + i]
                rmse = metrics.root_mean_squared_error(xsens_compare, dot_compare)
                if (rmse < min_rmse):
                    # print('New offset: %d       rmse: %.4f' % (i, rmse))
                    min_rmse = rmse
                    new_align = initial_time_align - i     
                
            return new_align
                    
                    
        
        # for each DOT sensor, time align it with the MVN data and partition the gait cycles based on the MVN gait events
        for sensor_location in xsens_dot_data:
            # if DOT sensors were synced prior to trial, time align based off one of lower legs --> most distinct
            # shapes and peaks when time aligning. Otherwise, align each individually
            fixed_time_align_sensor = 'll'
            if(is_dot_data_time_synced):
                dot_sensor_location = sensor_locs[fixed_time_align_sensor]
            else:
                dot_sensor_location = sensor_locs[sensor_location]
            
            dot_main_axis_time_align = dot_main_axis[dot_sensor_location]               # based on dot sensor, get relevant axis for time-align with mvn (either Y or Z of gyro)
            mvn_corresponding_sensor = dot_xsens_sensor_loc[dot_sensor_location]        # based on dot sensor, get matching mvn sensor number (based on mvn lower-body sensor order)
            
            test_filt_1 = scipy.signal.filtfilt(b20, a20, mvn_gyro_data[mvn_corresponding_sensor, :, 1])
            if(is_dot_data_time_synced):
                test_filt_2 = scipy.signal.filtfilt(b20, a20, xsens_dot_data[fixed_time_align_sensor][:, dot_main_axis_time_align])
            else:
                test_filt_2 = scipy.signal.filtfilt(b20, a20, xsens_dot_data[sensor_location][:, dot_main_axis_time_align])
            # min_dot_signal = -1 * np.min(dot_flip_axis[dot_sensor_location] * xsens_dot_data[i][:, dot_main_axis_time_align])
            # min_mvn_signal = -1 * np.min(mvn_gyro_data[mvn_corresponding_sensor, :, 1])
            min_dot_signal = -1 * np.min(dot_flip_axis[dot_sensor_location] * test_filt_2)
            min_mvn_signal = -1 * np.min(test_filt_1)
            
            # dot_peaks, _ = find_peaks(dot_flip_axis[dot_sensor_location] * -xsens_dot_data[i][:, dot_main_axis_time_align], height = 0.3 * min_dot_signal, distance = 30)
            # mvn_peaks, _ = find_peaks(-mvn_gyro_data[mvn_corresponding_sensor, :, 1], height = 0.3 * min_mvn_signal, distance = 30)
            dot_peaks, _ = find_peaks(dot_flip_axis[dot_sensor_location] * -test_filt_2, height = 0.3 * min_dot_signal, distance = 30)
            mvn_peaks, _ = find_peaks(-test_filt_1, height = 0.3 * min_mvn_signal, distance = 30)
            
            time_align = dot_peaks[0] - mvn_peaks[0]
            time_align = finetune_dot_alignment(dot_flip_axis[dot_sensor_location] * -test_filt_2, -test_filt_1 * 70, time_align, dot_peaks)
            
            mvn_gait_side = 0
            if('erL' in sensor_locs[sensor_location]):       # if sensor on LowerL or UpperL, use mvn left-gait events for partitioning (1). Else use right side
                mvn_gait_side = 1
                
            time_aligned_dot_gait_events = self.gait_events[mvn_gait_side].T + time_align
            partitioned_dot_data[sensor_locs[sensor_location]] = [xsens_dot_data[sensor_location][cycle[0]:cycle[2],:] for cycle in time_aligned_dot_gait_events]

        return partitioned_dot_data

    def process_mvn_trial_data(self, mvn_csv_filename, print_filenames = False):
        self.clear_gait_data()
        # default sample rate for MVN = 100Hz
        self.mvn_filename = mvn_csv_filename

        frame_rate = 100

        # Read .csv file which has all the MVN data. MVNX files must be converted to .csv using the excel_from_mvnx.py script
        self.mvnxData = pd.read_csv(self.mvn_filename)
        if print_filenames:
            print('Parsed file \'%s\'...' % (mvn_csv_filename))
        sensors = get_default_sensor_index()        # get sensors from MVN data

        pelvis_euler_orientation = self.get_pelvis_euler_orientation()
        sternum_euler_orientation = self.get_sternum_euler_orientation()
        lower_leg_orientation = self.get_lower_leg_orientation()
        foot_velocity = self.get_foot_velocity()
        hip_angles = self.get_hip_angles()
        knee_angles = self.get_knee_angles()
        ankle_angles = self.get_ankle_angles()

        # plt.figure()    
        # plt.plot(pelvis_euler_orientation[0,:,2], color = 'black')
        # plt.show()
        # exit()

        # orientation for turns already corrected in mvnx-to-excel conversion. Best practice is if you care about accelerometer or gyroscope
        # data from the MVN system, trials are trimmed to contain single passes only with heading reset, for consistent orientation
        gyro_data = self.get_gyro_data(sensors)
        acc_data = self.get_acc_data(sensors)
        
        # get position information from all segments and pelvis to map 
        # walkway and eliminate any transition/turning data present in samples  
        pelvis_position = self.get_pelvis_position()
        # print(pelvis_position.shape)
        foot_position = self.get_foot_position()
        # takes into account X and Y to deal with orientation drift over time. If single pass, can typically set close to 1.0 to eliminate start/stop
        range_to_keep = 0.98

        startInd = np.argmin(pelvis_position[:,0])
        endInd = np.argmax(pelvis_position[:,0])

        startPos = pelvis_position[startInd, 0:2]
        endPos = pelvis_position[endInd, 0:2]

        # IF USING SINGLE PASSES (MVN TRIALS TRIMMED SO NO TURNS), USE FIRST CENTER DETERMINATION
        # centroid_poly is useful for calculating the center of a long trial. However, if you've trimmed
        # so that all trials are a single pass, use simpler center calculation to avoid case where starting
        # or stopping at beginning/end of trial skews centroid of data
        center = [(endPos[0] + startPos[0]) / 2, (endPos[1] + startPos[1]) / 2]
        # center[0], center[1], _ = centroid_poly(pelvis_position[:,0], pelvis_position[:,1])
        
        dist = np.sqrt((endPos[0] - center[0])**2 + (endPos[1] - center[1])**2)

        # returns indices when person is not within x% of either end of a walk-way (assuming approximately straight walking in x-axis)
        position_not_ends = np.array([i for i in range(len(pelvis_position)) 
                                    if (np.sqrt((pelvis_position[i,0]-center[0])**2 + 
                                        (pelvis_position[i,1]-center[1])**2) < (range_to_keep * dist))])

        '''
        Visualization of the walkway, to verify excluding strides you want. Plots a top-down view of participant's trajectory as 
        determined by pelvis (x,y) position. Plots center of circle 

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.plot(pelvis_position[:,0], pelvis_position[:,1])
        c = plt.Circle(tuple(center), range_to_keep * dist, color='r')
        ax.add_patch(c)
        plt.scatter([center[0]], [center[1]], color='k', marker = 'o')
        fig.set_size_inches(7, 7)

        plt.xlim([-5, 25])
        plt.ylim([-15, 15])
        plt.show()
        '''
        

        # retrieve foot contact data, used to identify valid heel strikes and corresponding toe-offs
        seq = [0., 1.]
        heel_contacts = self.get_heel_contacts()
        toe_contacts = self.get_toe_contacts()

        # identify_gait_cycles returns 3xN matrix, 1st row = heel strike beginning of gait cycle, 2nd row = toe off
        # 3rd row = consecutive heel strike to end gait cycle
        self.gait_events = [self.identify_gait_events(position_not_ends, heel_contacts[0], toe_contacts[0], seq),
                            self.identify_gait_events(position_not_ends, heel_contacts[1], toe_contacts[1], seq)]
        
        '''
        Matches and right and left gait cycles, so same number of left and right gait cycles (for gait parameter calculation)
        Input: list of right and left gait events
        Return: N x 6 array, where each row is a matched r/l gait cycle

        '''
        def find_valid_r_l_strides(gait_events):
            
            r_gait_events = gait_events[0]
            l_gait_events = gait_events[1]
            
            stride_events = []
            for i in range(r_gait_events.shape[1]):
                rhs1 = r_gait_events[0][i]
                rto = r_gait_events[1][i]
                rhs2 = r_gait_events[2][i]
     
                l_events = [lhs for lhs in l_gait_events[0,:] if rhs1 < lhs < rhs2]

                if ((len(l_events) == 1)):
                    lhs1 = l_events[0]
                    lcycle = np.where(l_gait_events[0,:] == lhs1)[0][0]
                    lto = l_gait_events[1][lcycle]
                    lhs2 = l_gait_events[2][lcycle]

                    stride_events.append([rhs1, rto, rhs2, lhs1, lto, lhs2])

            stride_events = np.array(stride_events)
            return stride_events
        
        # finds matched right and left strides which are within the tolerance distance (range_to_keep)
        matched_r_l_strides = find_valid_r_l_strides(self.gait_events)
        self.gait_events = [matched_r_l_strides[:,:3].T,
                            matched_r_l_strides[:,3:].T]
        

        # partition the MVN strides into respective gait cycles, store in python dict
        self.partitioned_mvn_data['pelvis_orient'] = [self.divide_into_strides(self.gait_events, pelvis_euler_orientation, side='r', signal_num=0)]
        self.partitioned_mvn_data['hip_angle'] = [self.divide_into_strides(self.gait_events, hip_angles, side='r', signal_num=0), 
                                                  self.divide_into_strides(self.gait_events, hip_angles, side='l', signal_num=1)]
        self.partitioned_mvn_data['knee_angle'] = [self.divide_into_strides(self.gait_events, knee_angles, side='r', signal_num=0), 
                                                   self.divide_into_strides(self.gait_events, knee_angles, side='l', signal_num=1)]
        self.partitioned_mvn_data['ankle_angle'] = [self.divide_into_strides(self.gait_events, ankle_angles, side='r', signal_num=0), 
                                                    self.divide_into_strides(self.gait_events, ankle_angles, side='l', signal_num=1)]
        self.partitioned_mvn_data['sternum_orient'] = [self.divide_into_strides(self.gait_events, sternum_euler_orientation, side='r', signal_num=0)]
        self.partitioned_mvn_data['gyro_data'] = [self.divide_into_strides(self.gait_events, gyro_data, side='l',signal_num=i) if 'left' in sensors[i] 
                                             else self.divide_into_strides(self.gait_events, gyro_data, side='r', signal_num=i) for i in range(len(sensors))]
        self.partitioned_mvn_data['acc_data'] = [self.divide_into_strides(self.gait_events, acc_data, side='l',signal_num=i) if 'left' in sensors[i] 
                                            else self.divide_into_strides(self.gait_events, acc_data, side='r', signal_num=i) for i in range(len(sensors))]
        
        # # specific participant occasionally knocked left thigh-sensor mid-trial. This filters out those strides
        # if('/MI' in mvn_csv_filename):
        #     partitioned_mvn_data['gyro_data'][5] = [stride for stride in self.partitioned_mvn_data['gyro_data'][5] if np.max(stride[:,1]) - np.min(stride[:,1]) < 4.5]
        
        # print('Calculating spatiotemporal parameters...')
        self.gait_params['spatio_temp'] = self.calc_spatio_temp_params(self.gait_events, pelvis_position, foot_position, frame_rate, knee_angles, hip_angles, 
                                                                       foot_velocity, lower_leg_orientation)

        # print('Calculating kinematic parameters...')
        self.gait_params['kinematics'] = self.calc_kinematic_params(self.gait_events, pelvis_euler_orientation, hip_angles, knee_angles, ankle_angles, sternum_euler_orientation)

        # print('Calculating signal parameters...')
        self.gait_params['signal'] = self.calc_signal_params(self.partitioned_mvn_data)