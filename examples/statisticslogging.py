import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
# Load the CSV file
file_path = "C:\\Users\\ekuep\Desktop\\logresults_02-07-24_10-51.csv" 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import math
from scipy.stats import ttest_ind
from statistics import mean, stdev
data = pd.read_csv(file_path)
log_file = "C:\\Users\\ekuep\\Desktop\\ols_logs_aggreg.txt"
csv_file = "C:\\Users\\ekuep\\Desktop\\rsquared_values.csv"

data.columns = ['FilePath', 'Sensor', 'GaitParam', 'Algorithm', 'Participant', 'X', 'Y']
data['X'] = pd.to_numeric(data['X'], errors='coerce')
data['Y'] = pd.to_numeric(data['Y'], errors='coerce')


# def percent_difference(values):
#     base_value = values.iloc[0]  # Take the first value as base
#     return [math.ceil(abs(value - base_value)*100/ base_value) for value in values]

# def plot_scatter_and_regress(data, algorithm, participant, sensor):
#     subset = data[(data['Algorithm'] == algorithm) & 
#                   (data['Participant'] == participant) & 
#                   (data['Sensor'] == sensor)]
#     print(f'Generating plot and regression for {algorithm}, {participant}, {sensor}, number of data points: {len(subset)}')
    
#     if not subset.empty:
        
#         subset['Percent_Difference'] = percent_difference(subset['X'])
#         X = subset['Percent_Difference'].values
#         Y = subset['Y'].values
#         subset_df = pd.DataFrame({'X': X, 'Y': Y})
#         model = ols('Y~X',data=subset_df)
#         results = model.fit()
#         # Save summary to log file
#         with open(log_file, 'a') as f:
#             f.write(f'Regression results for {algorithm}, {participant}, {sensor}:\n')
#             f.write(results.summary().as_text() + '\n\n')
        
#         rsquared = results.rsquared
        
#         # Save R-squared value to CSV
#         with open(csv_file, 'a') as f:
#             f.write(f'{algorithm},{participant},{sensor},{rsquared}\n')    
            
#         # Print summary of OLS regression results
#         print(results.summary2())
#         plt.figure()
#         plt.scatter(X, Y, label=f'{algorithm}-{participant}-{sensor}')
#         sns.regplot(x='X',y='Y',data = subset_df,ci=95)
#         plt.xticks(X)
#         # plt.plot(subset['Percent_Difference'], y_pred, col
#         plt.ylabel('Similarity')
#         plt.xlabel('% Change Stance Time Symmetry')
#         plt.title(f'Scatter Plot and Regression for {algorithm}, {participant}, {sensor}')
#         #plt.savefig(f'C:\\Users\\ekuep\Desktop\\{algorithm}-{participant}-{sensor}.png')
        
#     else:
#         print(f'No data available for {algorithm}, {participant}, {sensor}')


# participants = data['Participant'].unique()
# sensors = data['Sensor'].unique()
# algorithms = data['Algorithm'].unique()


# for participant in participants:
#     for sensor in sensors:
#         for algorithm in algorithms:
#             plot_scatter_and_regress(data, algorithm, participant, sensor)


# Rename columns if needed
data.columns = ['FilePath', 'Sensor', 'GaitParam', 'Algorithm', 'Participant', 'X', 'Y']

# Convert X and Y to numeric
data['X'] = pd.to_numeric(data['X'], errors='coerce')
data['Y'] = pd.to_numeric(data['Y'], errors='coerce')

# Function to calculate percent difference rounded to nearest 3 (or 3 and 9)
def percent_difference(values):
    base_value = values.iloc[0]  # Take the first value as base
  # Take the first value as base
    return [(abs(value - base_value)*100/ base_value) for value in values]

def calculate_srm(col1, col2):
    # Filter out None values
    valid_indices = ~col1.isna() & ~col2.isna()
    if np.any(valid_indices):
        mean_diff = np.mean(col1[valid_indices] - col2[valid_indices])
        std_diff = np.std((col1[valid_indices] - col2[valid_indices]),ddof=1)
        return mean_diff / std_diff
    else:
        return np.nan  # Return NaN if no valid values are present

def calculate_cohens_d(group1, group2):
    mean_diff = mean(group1) - mean(group2)
    pooled_std = np.sqrt((stdev(group1)**2 + stdev(group2)**2) / 2)
    return mean_diff / pooled_std

# Function to plot scatter plots and regression
def plot_scatter_and_regress(data, algorithm, sensor, csv_file):
    
    csv_file = "C:\\Users\\ekuep\\Desktop\\SRM_values.csv"
    csv_file2 = "C:\\Users\\ekuep\\Desktop\\ttest_values.csv"
    csv_file3 = "C:\\Users\\ekuep\\Desktop\\CohensD_values.csv"
    
    subset = data[(data['Algorithm'] == algorithm) & 
                  (data['Sensor'] == sensor)]
    print(f'Generating plot and regression for {algorithm}, {sensor}, number of data points: {len(subset)}')
    
    if not subset.empty:
        plt.figure()
        df = pd.DataFrame(columns=['0', '3%', '6%', '9%'], index=range(12))  # Initialize outside the loop
        for i, participant in enumerate(subset['Participant'].unique()):
            participant_data = subset[subset['Participant'] == participant].copy()
            participant_data['Percent_Difference'] = percent_difference(participant_data['X'])
            participant_data['Y_diff'] = participant_data['Y'] - participant_data['Y'].iloc[0]
            Y = participant_data['Y'].values.tolist()
            
            if len(Y) == 3:
                df.loc[i] = Y + [None]  # Add None to make it 4 elements
            elif len(Y) == 4:
                df.loc[i] = Y
        print(df)
        srm_results = {}
        t_test_results = {}
        cohen_d_results = {}
        
        start_column = '0'
        for col in df.columns:
            if col != start_column and not df[col].isna().all():  # Ignore '0%' column and columns with all None values
                # Ensure numeric data and drop NaNs
                data1 = df[start_column].dropna().astype(float)
                data2 = df[col].dropna().astype(float)
                
                # Calculate SRM
                srm_value = calculate_srm(data1, data2)
                srm_results[col] = srm_value
                
                # Perform Welch's t-test
                t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
                t_test_results[col] = {'t_statistic': t_stat, 'p_value': p_val}
                
                # Calculate Cohen's d
                cohen_d = calculate_cohens_d(data1, data2)
                cohen_d_results[col] = cohen_d
                
                
        # # Append results to log file
        # with open("log.txt", "a") as log_file:
        #     log_file.write(f"Algorithm: {algorithm}, Sensor: {sensor}, SRM Results: {srm_results}\n")
        
        # Append results to CSV file
        with open(csv_file, "a") as csv_file:
            for key, value in srm_results.items():
                csv_file.write(f"{algorithm},{sensor},{key},{value}\n") 
        
        with open(csv_file2, "a") as csv_file2:
            for key, value in t_test_results.items():
                csv_file2.write(f"{algorithm},{sensor},{key},{value}\n") 
                
        with open(csv_file3, "a") as csv_file3:
            for key, value in cohen_d_results.items():
                csv_file3.write(f"{algorithm},{sensor},{key},{value}\n") 
                

# Define the CSV file path for R-squared values


# Get unique combinations of sensors and algorithms
sensors = data['Sensor'].unique()
algorithms = data['Algorithm'].unique()

# Plot for each combination of sensor and algorithm
for sensor in sensors:
    for algorithm in algorithms:
        plot_scatter_and_regress(data, algorithm, sensor, csv_file)
        




            # plt.scatter(X, Y, label=f'{algorithm}-{participant}-{sensor}',color='k')
            #aggregate_data = aggregate_data._append(pd.DataFrame({'X': X, 'Y_diff': Y}), ignore_index=True)
            
        # Perform OLS regression for aggregate data
        # model = ols('Y_diff ~ X', data=aggregate_data)
        # results = model.fit()
        
    #     with open(log_file, 'a') as f:
    #         f.write(f'Regression results for {algorithm}, {sensor}:\n')
    #         f.write(results.summary().as_text() + '\n\n')
        
    #     plt.ylabel('Change in Similarity')
    #     plt.xlabel('% Change Stance Time Symmetry')
    #     plt.title(f'Scatter Plot for {algorithm}, {sensor}')
    #     plt.savefig(f'C:\\Users\\ekuep\\Desktop\\{algorithm}-{sensor}_scatter.png')
        
    # else:
    #     print(f'No data available for {algorithm}, {sensor}')