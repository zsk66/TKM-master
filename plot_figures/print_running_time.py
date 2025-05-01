import json



print('loading data...')
k=4
num_subsample = 1
num_data_list = [1000, 2000, 5000,10000,20000]
dataset_names = ['hmda']
for dataset_name in dataset_names:
    time_jkl_num, time_mv_num, time_fr_num, time_sfr_num = [], [], [],[]
    for num_data in num_data_list:
        time_jkl_id, time_mv_id, time_fr_id,time_sfr_id = [], [], [], []
        for subsample_id in range(num_subsample):
            file_path_process_time = '../individually-fair-k-clustering-main/output/kmeans/'+dataset_name+'_'+str(num_data)+'_'+str(subsample_id)+'_k_'+str(k)+'.json'
            with open(file_path_process_time, 'r') as file:
                output_process_time = json.load(file)
            file_path_data_time = '../individually-fair-k-clustering-main/data/'+dataset_name+'_'+str(num_data)+'_'+str(subsample_id)+'_time'+'.json'
            with open(file_path_data_time, 'r') as file:
                output_data_time = json.load(file)
                time_data = output_data_time['time']
            # time_jkl = output_process_time['jkl_output']['time'] + time_data
            # time_mv = output_process_time['mv_output']['time'] + time_data
            time_sfr = output_process_time['spa_fr_output']['time'] + time_data
            # time_fr = output_process_time['fr_output']['time'] + time_data
            # time_jkl_id.append(time_jkl)
            # time_mv_id.append(time_mv)
            time_sfr_id.append(time_sfr)

            # time_fr_id.append(time_fr)
        # time_jkl_num.append(sum(time_jkl_id)/len(time_jkl_id))
        # time_mv_num.append(sum(time_mv_id)/len(time_mv_id))
        time_sfr_num.append(sum(time_sfr_id)/len(time_sfr_id))

        # time_fr_num.append(sum(time_fr_id)/len(time_fr_id))





num_data_list = [20000]

for dataset_name in dataset_names:
    for num_data in num_data_list:
        time_jkl_id, time_mv_id= [], []
        for subsample_id in range(num_subsample):
            file_path_process_time = '../individually-fair-k-clustering-main/output/kmeans/' + dataset_name + '_' + str(
                num_data) + '_' + str(subsample_id) + '_k_' + str(k) + '.json'
            with open(file_path_process_time, 'r') as file:
                output_process_time = json.load(file)
            file_path_data_time = '../individually-fair-k-clustering-main/data/' + dataset_name + '_' + str(
                num_data) + '_' + str(subsample_id) + '_time' + '.json'
            with open(file_path_data_time, 'r') as file:
                output_data_time = json.load(file)
                time_data = output_data_time['time']
            time_jkl = output_process_time['jkl_output']['time'] + time_data
            time_mv = output_process_time['mv_output']['time'] + time_data
            time_jkl_id.append(time_jkl)
            time_mv_id.append(time_mv)
        time_jkl_num.append(sum(time_jkl_id) / len(time_jkl_id))
        time_mv_num.append(sum(time_mv_id) / len(time_mv_id))

