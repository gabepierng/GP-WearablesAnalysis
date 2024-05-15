import os
import pandas as pd


script_dir = os.path.dirname(__file__)
folder_path = script_dir
files = os.listdir(folder_path)
csv_files = [file for file in files if file.startswith('HMM_SMM_log') and file.endswith('.csv')]
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    states_5 = df[df['States'] == 5]
    
    if not states_5.empty:
       
        similarity_stats = states_5['Similarity'].describe()
        
        print(f"Statistics for {file_name}:")
        print(similarity_stats)
    else:
        print(f"No rows with 'States' equal to 5 in {file_name}.")

        