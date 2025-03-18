eval_path='./data/dev.json'
dev_path='./output/'
db_root_path='./data/dev_databases/'
use_knowledge='True'
not_use_knowledge='False'
mode='dev' # choose dev or dev
cot='True'
no_cot='Fales'

YOUR_API_KEY='XXX'

#engine3='meta-llama/Llama-3.3-70B-Instruct-Turbo'
engine3='meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'

# data_output_path='./exp_result/gpt_output/'
# data_kg_output_path='./exp_result/gpt_output_kg/'

data_output_path='./exp_result/llama/405b/turbo_output/'
data_kg_output_path='./exp_result/llama/405b/turbo_output_kg/'


# echo 'generate GPT3.5 batch with knowledge'
# python3 -u ./src/gpt_request.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
# --engine ${engine3} --eval_path ${eval_path} --data_output_path ${data_kg_output_path} --use_knowledge ${use_knowledge} \
# --chain_of_thought ${no_cot}

echo 'generate Llama 3 batch without knowledge'
python3 -u ./src/llama_request.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
--engine ${engine3} --eval_path ${eval_path} --data_output_path ${data_output_path} --use_knowledge ${not_use_knowledge} \
--chain_of_thought ${no_cot}
