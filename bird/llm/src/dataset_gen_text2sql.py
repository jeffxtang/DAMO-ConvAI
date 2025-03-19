
import json
import os
import re
from langchain_together import ChatTogether
from langchain_community.utilities import SQLDatabase
from datasets import Dataset

#with open("train.json", 'r') as file:
with open("dev.json", 'r') as file:
  data = json.load(file)

#pip install -U langchain langchain-community langchain-together

os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_API_KEY')
print(os.environ['TOGETHER_API_KEY'])

llama33_70b = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
)

llama31_405b = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    temperature=0,
)

deepseekr1 = ChatTogether(
    model="deepseek-ai/DeepSeek-R1",
    temperature=0,
)

def get_schema(db):
    return db.get_table_info()


def generate_reasoning(llm, db, question, golden_sql):
  prompt = f"""Based on the table schema, given the question and its SQL query below, provide a reasoning to infer the SQL from the question.

    Schema:
    {get_schema(db)}

    Question:
    {question}

    SQL query:
    {golden_sql}
"""

  reasoning = llm.invoke(prompt).content
  return reasoning

db_names = ['superhero', 'card_games', 'financial', 
  'student_club', 'california_schools', 'european_football_2', 
  'thrombosis_prediction', 'codebase_community',
  'formula_1', 'toxicology', 'debit_card_specializing']

db_jsons = [[{'question': d['question'], 'SQL': d['SQL']} for d in data if d['db_id'] == db_name] for db_name in db_names]
db_names = [f'{db_name}.sqlite' for db_name in db_names]

for llm in [llama33_70b, deepseekr1]: #, llama31_405b]:
  reasoning_dataset = []
  for n, db_name in enumerate(db_names[:1]):
    db = SQLDatabase.from_uri(f"sqlite:///{db_name}", sample_rows_in_table_info=0)
    print(f"Generating dataset for {db_name} with {llm=}")
    for itm in db_jsons[n][:3]:
      question = itm['question']
      sql = itm['SQL']
      reasoning = generate_reasoning(llm, db, question, sql)

      entry = {
        "question": question,
        "SQL": sql,
        "reasoning": reasoning
      }
      reasoning_dataset.append(entry)

  dataset_dict = {key: [d[key] for d in reasoning_dataset] for key in reasoning_dataset[0]}
  hf_dataset = Dataset.from_dict(dataset_dict)
  hf_dataset.save_to_disk(f'text2sql_with_reasoning_{llm.model_name}.arrow')


