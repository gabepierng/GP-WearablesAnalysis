# functions for calculating different gait scores
import numpy as np
import copy
from hmmlearn import hmm

'''
Functions related to hidden Markov model-based similarity measure (HMM-SM)
'''
def initiate_hmm(n_components=4, cov_type='full', n_iter=50, tolerance = 0.005):

    # self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter, tol=tolerance, params='stmc', init_params='stmc', verbose=False)
    hmm_model = hmm.GaussianHMM(n_components=n_components, covariance_type=cov_type, n_iter=n_iter, 
                                    tol=tolerance, params='stmc', init_params='mc', verbose=False)
    hmm_model.transmat_ = np.zeros((n_components, n_components))
    for i in range(n_components):
        hmm_model.transmat_[i, 0:2] = np.random.dirichlet(np.ones(2)/1.5, size=1)
        hmm_model.transmat_[i, 2:] = np.random.dirichlet(np.ones(n_components - 2), size=1)[0] / 1e80
        hmm_model.transmat_[i] = np.roll(hmm_model.transmat_[i], i)

    return hmm_model

def train_hmm(hmm_model, X, resize_len, num_sequences):
    np.seterr(all='ignore')
    hmm_model.fit(X, lengths=resize_len * np.ones(num_sequences, dtype=int))

def check_forward(hmm_model):
    isForward = False
    valid_rows = 0
    a_mat = hmm_model.transmat_
    for i, row in enumerate(a_mat):
        temp = np.argpartition(np.roll(row, -i), -2)[-2:]
        if((np.array(temp) == np.array([0,1])).all() or (np.array(temp) == np.array([1,0])).all()):
            valid_rows = valid_rows + 1

    if(valid_rows == hmm_model.transmat_.shape[0]):
        isForward = True

    return isForward

'''
If comparing the state sequence predictions for multiple HMMs, can be helpful to align their states. This is because training
can result in learning the same state transitions, but e.g., state 1->2->3->1 of HMM[1] corresponds to 2->3->1->2 of HMM[2].
align_states will rotate the state transition and emissions matrices (the individual entries still match)
Inputs:
    trained_hmm_model: model to align. 

Output:
    new_hmm: altered HMM with rolled states
'''        
def align_states(trained_hmm_model, roll_amount):
    new_hmm = copy.deepcopy(trained_hmm_model)
    array_order = np.roll(np.arange(num_states), roll_amount)

    new_hmm.model.transmat_ = new_hmm.model.transmat_[array_order,: ]
    for i, row in enumerate(new_hmm.model.transmat_):
        new_hmm.model.transmat_[i] = np.roll(new_hmm.model.transmat_[i], roll_amount)
        
    new_hmm.model.means_ = new_hmm.model.means_[array_order, :]
    new_hmm.model.covars_ = new_hmm.model.covars_[array_order, :]
    new_hmm.model.startprob_ = new_hmm.model.startprob_[array_order]
    return new_hmm  

'''
For comparing multiple HMMs (which should be similar) to match the states, automatically searches for the best rotation alignment
Inputs:
    hmm_1 and hmm_2: HMMs to align states. Algorithm will match hmm_2 to hmm_1
    test_strides: test gait cycle to compare state sequence predictions
    n_states: number of states of HMMs

Output:
    best_roll: the determined best alignment for rolling hmm_2 to match hmm_1, feed into align_states (roll_amount)
'''
def find_best_alignment(hmm_1, hmm_2, test_stride):
    n_states = hmm_1.model.transmat_.shape[0]
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

'''
Calculates the Q matrix as outlined in https://ieeexplore.ieee.org/abstract/document/5654664. Q matrix
is an N x N array (N = number of HMM states), where each element q[i,j] corresponds to the similarity of 
HMM_1 state i and HMM_2 state j
Inputs:
    hmm_1 and hmm_2: HMMs for computing Q matrix

Output:
    q_matrix: N x N state correspondence matrix
'''    
def calculate_state_correspondence_matrix(hmm_1, hmm_2, n_states):

    def calculate_stationary_distribution(hmm):
        eigenvals, eigenvectors = np.linalg.eig(hmm.transmat_.T)
        stationary = np.array(eigenvectors[:, np.where(np.abs(eigenvals - 1.) < 1e-8)[0][0]])
        stationary = stationary / np.sum(stationary)
        return np.expand_dims(stationary.real, axis=-1)
    
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
            kl_state_comparisons[i,j] = 0.5 * (calculate_KL_div(hmm_1, hmm_2, i, j) + calculate_KL_div(hmm_2, hmm_1, i, j))
            total_expected_similarity = total_expected_similarity + (pi_1[i] * pi_2[j] * kl_state_comparisons[i,j])

    # alternate for computing s_e, depending on desired similarity characteristics
    # k = 1
    # s_e = np.exp(-k * kl_state_comparisons)
    s_e = 1 / kl_state_comparisons
    # pi_1.T @ pi_2 should produce a 3x3 matrix (pi_1i * pi_2j)

    q_matrix = ((pi_1 @ pi_2.T) * s_e) / total_expected_similarity
    
    return q_matrix

'''
Calculates the normalized Gini index of Q matrix (i.e., sparsity).
Inputs:
    q_matrix: state correspondence matrix

Output:
    gini_index: returns float in range[0,1]. 1 indicates maximum sparsity
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
 
    n_states = q_matrix.shape[0]
    r = (1 / n_states) * np.sum([calc_gini(row) for row in q_matrix])
    c = (1 / n_states) * np.sum([calc_gini(column) for column in q_matrix.T])
    
    gini_index = 0.5 * (r + c)
    return gini_index

def compute_hmmsm(hmm_models_1, hmm_models_2, num_states):
    sum_hmmsm = 0
    count = 0
    for k in range(len(hmm_models_1)):
        for m in range(len(hmm_models_2)):
            # x = calculate_state_correspondence_matrix(hmm_models_aligned_states[trial_types[0]][k], hmm_models_aligned_states[trial_types[i]][m])
            x = calculate_state_correspondence_matrix(hmm_models_1[k], hmm_models_2[m], num_states)
            sum_hmmsm = sum_hmmsm + calculate_gini_index(x, num_states)
            count = count+1

    mean_hmmsm = sum_hmmsm / count
    return mean_hmmsm