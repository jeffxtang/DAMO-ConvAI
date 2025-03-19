
import json
import os
import re
from langchain_together import ChatTogether
from langchain_community.utilities import SQLDatabase

from datasets import load_from_disk
ds = load_from_disk("./text2sql_with_reasoning_meta-llama/Llama-3.3-70B-Instruct-Turbo.arrow")

#print(len(ds['question']),ds['reasoning'][3])

with open("dev.json", 'r') as file:
  data = json.load(file)

db_names = ['superhero', 'card_games', 'financial', 
  'student_club', 'california_schools', 'european_football_2', 
  'thrombosis_prediction', 'codebase_community',
  'formula_1', 'toxicology', 'debit_card_specializing']
dbs = {db_name: SQLDatabase.from_uri(f"sqlite:///{db_name}.sqlite", sample_rows_in_table_info=0) for db_name in db_names}

diff = 0
for n in range(len(ds['question'])):
  for item in data:
    if item['question'] == ds['question'][n]:
      print("*")
      pattern = re.compile(r'```sql\n*(.*?)```', re.DOTALL)
      answer = ds['reasoning'][n]
      matches = pattern.findall(answer)
      if matches != []:
        generated_sql = matches[0]
      else:
        generated_sql = answer

      if item['SQL'] != generated_sql:
        print(f"{n=}, {diff=}, {generated_sql=}")
        print(item['SQL'])

        try:
          generated_sql_rslt = dbs[item['db_id']].run(generated_sql)
          gold_rslt = dbs[item['db_id']].run(item['SQL'])
          if generated_sql_rslt != gold_rslt:
            diff += 1
          # print(item['db_id'])
          # print(ds['reasoning'][n])
          # print(item['question'])
          print(generated_sql_rslt)
          print(gold_rslt)
        except Exception as e:
          print(e)  
          diff += 1
      else:
        print("+")


print(diff)
print(len(ds['question']))
corr = len(ds['question']) - diff
print(f"{corr/len(ds['question']):.2f}")
