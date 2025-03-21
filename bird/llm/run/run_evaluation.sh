db_root_path='./data/dev_databases/'
data_mode='dev'
diff_json_path='./data/dev.json'
#predicted_sql_path_kg='./exp_result/turbo_output_kg/'
#predicted_sql_path='./exp_result/openai/gpt4o/turbo_output/'
#predicted_sql_path='./exp_result/openai/gpt-3.5-turbo/turbo_output/'
predicted_sql_path='./exp_result/llama/3.3_70b/turbo_output/'
ground_truth_path='./data/'
num_cpus=2
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'

# echo '''starting to compare with knowledge for ex'''
# python3 -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
# --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare without knowledge for ex'''
python3 -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

# echo '''starting to compare with knowledge for ves'''
# python3 -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
# --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

# echo '''starting to compare without knowledge for ves'''
# python3 -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
# --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}