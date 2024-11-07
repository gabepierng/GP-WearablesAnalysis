import os
from google.cloud import storage



storage_client = storage.Client()
bucket_name = 'gaitbfb_propellab/'
base_directory = bucket_name + 'Wearable Biofeedback System (REB-0448)/Data/Raw Data'
bucket_name = 'gaitbfb_propellab'
blobs = storage_client.list_blobs(bucket_name, prefix = base_directory)
prefix_from_bucket = 'Wearable Biofeedback System (REB-0448)/Data/Raw Data/' 

participant_list = ['LLPU_P01','LLPU_P02','LLPU_P03','LLPU_P04','LLPU_P05','LLPU_P06','LLPU_P08','LLPU_P09','LLPU_P10','LLPU_P12','LLPU_P15']
num_training = []


for participant in participant_list:
    participant_count = 0
    print(f"Processing participant {participant}")
    participant_id = participant
    
    directory = prefix_from_bucket + participant + '/Excel_Data'
    blobs = storage_client.list_blobs(bucket_or_name=bucket_name, prefix=directory.replace("\\", "/"))

    if blobs:
        for blob in blobs:
            if blob.name.startswith(f'{directory}/Training') or blob.name.startswith(f'{directory}/COL'):
                participant_count += 1


    num_training.append(participant_count)

print(min(num_training))
print(max(num_training))