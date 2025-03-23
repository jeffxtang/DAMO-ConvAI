
import json
import os
import re
from langchain_together import ChatTogether
from langchain_community.utilities import SQLDatabase
from datasets import Dataset
import sqlite3
import multiprocessing


#with open("train.json", 'r') as file:
with open("dev.json", 'r') as file:
  data = json.load(file)

#pip install -U langchain langchain-community langchain-together

os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_API_KEY')

llama31_8b = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0,
)

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
  #prompt = f"""Based on the table schema, given the question and its SQL query below, provide a reasoning to infer the SQL from the question.
  prompt = f"""Based on the SQL DB schema and the question below, think step by step and reason carefully to infer the SQL from the question.

    Schema:
    {get_schema(db)}

    Question:
    {question}
"""

    # SQL query:
    # {golden_sql}

  reasoning = llm.invoke(prompt).content
  return reasoning

# db_names = ['superhero', 'card_games', 'financial', 
#   'student_club', 'california_schools', 'european_football_2', 
#   'thrombosis_prediction', 'codebase_community',
#   'formula_1', 'toxicology', 'debit_card_specializing']

# db_jsons = [[{'question': d['question'], 'SQL': d['SQL'], 'db_id': d['db_id']} for d in data if d['db_id'] == db_name] for db_name in db_names]
# db_names = [f'{db_name}.sqlite' for db_name in db_names]

# for llm in [llama33_70b]: # llama31_8b]: #[llama33_70b, deepseekr1]: #, llama31_405b]:
#   reasoning_dataset = []
#   for n, db_name in enumerate(db_names):
#     db = SQLDatabase.from_uri(f"sqlite:///{db_name}", sample_rows_in_table_info=0)
#     print(f"Generating dataset for {db_name} with {llm=}")
#     for itm in db_jsons[n]:
#       question = itm['question']
#       #print(f'{question=}')
#       sql = itm['SQL']
#       reasoning = generate_reasoning(llm, db, question, sql)
#       #print(f'{reasoning=}')

#       entry = {
#         "question": question,
#         "SQL": sql,
#         "db_id": itm['db_id'],
#         "reasoning": reasoning
#       }
#       reasoning_dataset.append(entry)

#       print(entry)

#   dataset_dict = {key: [d[key] for d in reasoning_dataset] for key in reasoning_dataset[0]}
#   hf_dataset = Dataset.from_dict(dataset_dict)
#   hf_dataset.save_to_disk(f'march20/text2sql_with_reasoning_{llm.model_name}.arrow')



import json
import os
import re
from langchain_together import ChatTogether
from langchain_community.utilities import SQLDatabase

from datasets import load_from_disk

# with open("dev.json", 'r') as file:
#   data = json.load(file)

db_names = ['superhero', 'card_games', 'financial', 
  'student_club', 'california_schools', 'european_football_2', 
  'thrombosis_prediction', 'codebase_community',
  'formula_1', 'toxicology', 'debit_card_specializing']
#dbs = {db_name: SQLDatabase.from_uri(f"sqlite:///{db_name}.sqlite", sample_rows_in_table_info=0) for db_name in db_names}
db_conns = {db_name: sqlite3.connect(f"{db_name}.sqlite", timeout=5.0) for db_name in db_names}


def execute_query(db_id, query, result_queue):
  try:
    connection = db_conns[db_id]
    cursor = connection.cursor()
    cursor.execute(query)
    result_queue.put(cursor.fetchall())
    connection.close()
  except Exception as e:
    result_queue.put(e)


model = "Llama-3.3-70B-Instruct-Turbo"
#model = "Meta-Llama-3.1-8B-Instruct-Turbo"
#ds = load_from_disk(f"march20/text2sql_with_reasoning_meta-llama/{model}.arrow")
ds = load_from_disk(f"march20/text2sql_with_reasoning_meta-llama/{model}.arrow")

diff = 0
for n in range(len(ds['question'])):
  question = ds['question'][n]
  answer = ds['reasoning'][n]
  gold_sql = ds['SQL'][n]
  db_id = ds['db_id'][n]

  pattern = re.compile(r'```sql\n*(.*?)```', re.DOTALL)
  matches = pattern.findall(answer)
  if matches != []:
    generated_sql = matches[0]
  else:
    generated_sql = answer

  print(f"{n=}, {question=}")
  print(f"{generated_sql=}")
  print(f"{gold_sql=}")  

  if gold_sql != generated_sql:
    print(f"{n=}, {diff=}")

    try:
      # generated_sql_rslt = dbs[db_id].run(generated_sql)
      # gold_sql_rslt = dbs[db_id].run(gold_sql)

      # cursor = db_conns[db_id].cursor()
      # cursor.execute(generated_sql)
      # generated_sql_rslt = cursor.fetchall()

      generated_sql_rslt = ""
      result_queue = multiprocessing.Queue()
      # Create a process to execute the query
      query_process = multiprocessing.Process(target=execute_query, args=(db_id, generated_sql, result_queue))
      # Start the process
      query_process.start()
      # Wait for a specified timeout period
      timeout = 10  # seconds
      query_process.join(timeout)
      # Check if the process is still active
      if query_process.is_alive():
        print("Query is taking too long and will be terminated.")
        query_process.terminate()  # Terminate the process
        query_process.join()  # Wait for the process to finish
        generated_sql_rslt = ""
      else:
        # Check the result
        if not result_queue.empty():
          generated_sql_rslt = result_queue.get()
          if isinstance(generated_sql_rslt, Exception):
            print(f"An error occurred: {generated_sql_rslt}")
          else:
            print("Query result:", generated_sql_rslt)


      cursor = db_conns[db_id].cursor()
      cursor.execute(gold_sql)
      gold_sql_rslt = cursor.fetchall()

      if generated_sql_rslt != gold_sql_rslt:
        diff += 1
      print(generated_sql_rslt)
      print(gold_sql_rslt)
    except Exception as e:
      print(e)  
      diff += 1


print(diff)
print(len(ds['question']))
corr = len(ds['question']) - diff
print(f">>> {model}: {corr/len(ds['question']):.2f}")

# ds = load_from_disk("./text2sql_with_reasoning_meta-llama/Llama-3.3-70B-Instruct-Turbo.arrow")

# diff = 0
# for n in range(len(ds['question'])):
#   for item in data:
#     if item['question'] == ds['question'][n]:
#       pattern = re.compile(r'```sql\n*(.*?)```', re.DOTALL)
#       answer = ds['reasoning'][n]
#       matches = pattern.findall(answer)
#       if matches != []:
#         generated_sql = matches[0]
#       else:
#         generated_sql = answer

#       if item['SQL'] != generated_sql:
#         print(f"{n=}, {diff=}, {generated_sql=}")
#         print(item['SQL'])

#         try:
#           generated_sql_rslt = dbs[item['db_id']].run(generated_sql)
#           gold_rslt = dbs[item['db_id']].run(item['SQL'])
#           if generated_sql_rslt != gold_rslt:
#             diff += 1
#           print(generated_sql_rslt)
#           print(gold_rslt)
#         except Exception as e:
#           print(e)  
#           diff += 1

# print(diff)
# print(len(ds['question']))
# corr = len(ds['question']) - diff
# print(f">>> Llama 3.3 70b: {corr/len(ds['question']):.2f}")