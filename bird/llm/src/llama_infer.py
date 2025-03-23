import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported


#### Text2SQL ####
import json
from datasets import Dataset
from datasets import load_from_disk
from langchain_community.utilities import SQLDatabase

def get_schema(db):
    return db.get_table_info()

with open("dev.json", 'r') as file:
  data = json.load(file)

# dev db names
db_names = ['superhero'] #, 'card_games', 'financial', 
#   'student_club', 'california_schools', 'european_football_2', 
#   'thrombosis_prediction', 'codebase_community',
#   'formula_1', 'toxicology', 'debit_card_specializing']

# train db names
# db_names = ['address',
# 'airline',
# 'app_store',
# 'authors',
# 'beer_factory',
# 'bike_share_1',
# 'book_publishing_company',
# 'books',
# 'car_retails',
# 'cars',
# 'chicago_crime',
# 'citeseer',
# 'codebase_comments',
# 'coinmarketcap',
# 'college_completion',
# 'computer_student',
# 'cookbook',
# 'craftbeer',
# 'cs_semester',
# 'disney',
# 'donor',
# 'european_football_1',
# 'food_inspection',
# 'food_inspection_2',
# 'genes',
# 'hockey',
# 'human_resources',
# 'ice_hockey_draft',
# 'image_and_language',
# 'language_corpus',
# 'law_episode',
# 'legislator',
# 'mental_health_survey',
# 'menu',
# 'mondial_geo',
# 'movie',
# 'movie_3',
# 'movie_platform',
# 'movielens',
# 'movies_4',
# 'music_platform_2',
# 'music_tracker',
# 'olympics',
# 'professional_basketball',
# 'public_review_platform',
# 'regional_sales',
# 'restaurant',
# 'retail_complains',
# 'retail_world',
# 'retails',
# 'sales',
# 'sales_in_weather',
# 'shakespeare',
# 'shipping',
# 'shooting',
# 'simpson_episodes',
# 'soccer_2016',
# 'social_media',
# 'software_company',
# 'student_loan',
# 'superstore',
# 'synthea',
# 'talkingdata',
# 'trains',
# 'university',
# 'video_games',
# 'works_cycles',
# 'world',
# 'world_development_indicators'

# ]

db_jsons = [[{'question': d['question'], 'SQL': d['SQL'], 'db_id': d['db_id']} for d in data if d['db_id'] == db_name] for db_name in db_names]
db_names = [f'{db_name}.sqlite' for db_name in db_names]

ds_text2sql = []
for n, db_name in enumerate(db_names):
    print(f"Generating dataset for {db_name}")
    for itm in db_jsons[n]:
        db = SQLDatabase.from_uri(f"sqlite:///{db_name}", sample_rows_in_table_info=0)
        question = itm['question']
        sql = itm['SQL']
        entry = {"conversations": [{
        "from": "human",
        "value": f"""Based on the SQL DB schema and the question below, generate the SQL from the question.

DB Schema:
{get_schema(db)}

Question:
{question}
"""
        }, {
        "from": "llama",
        "value": sql
        }]}
        ds_text2sql.append(entry)

for key in ds_text2sql[0]:
    print(f"{key=}")


dataset_dict = {key: [d[key] for d in ds_text2sql] for key in ds_text2sql[0]}
hf_dataset = Dataset.from_dict(dataset_dict)
hf_dataset.save_to_disk(f'march23/text2sql_1.arrow')

#dataset = load_from_disk(f"train/march22/text2sql_1.arrow")
dataset = load_from_disk(f"march23/text2sql_1.arrow")
print("*******")
print(dataset)
print(">>>>>>>")

print(dataset[0])
print("-----------")
print(dataset['conversations'][0][0])


max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "llama"},
    chat_template="llama3", #"chatml",
)

def apply_template_text2sql(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}    

print("\n==============\n")
dataset = dataset.map(apply_template_text2sql, batched=True)
print(dataset)
print("<<<<<<<")
print(dataset['text'][0])

model = FastLanguageModel.for_inference(model)

# messages = [
#     {"from": "human", "value": "Is 9.11 larger than 9.9?"},
# ]

messages = [dataset['conversations'][0][0]]
print(f"{messages=}")
print(f"{[dataset['text'][0][0]]=}")

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

#print(f"{inputs=}")


text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)

#model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
#model.push_to_hub_merged("mlabonne/FineLlama-3.1-8B", tokenizer, save_method="merged_16bit")