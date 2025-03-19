
import json
import os
import re
from langchain_together import ChatTogether
from langchain_community.utilities import SQLDatabase


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

deepseekv3 = ChatTogether(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0,
)


def get_schema(db):
    return db.get_table_info()

def text2sql_eval(llm, db, json_lst, db_specific_prompt=""):
  correct = 0
  total = 0
  for itm in json_lst:
    question = itm['question']
    query = itm['SQL']
    total += 1
    print(f"{question=}")
    prompt = f"""Based on the table schema below, write a SQL query that would answer the user's question. Be concise and just return the SQL query and nothing else. {db_specific_prompt}

    Scheme:
    {get_schema(db)}

    Question: {question}
    """

    answer = llm.invoke(prompt).content
    pattern = re.compile(r'```sql\n*(.*?)```', re.DOTALL)
    matches = pattern.findall(answer)
    if matches != []:
      generated_sql = matches[0]
    else:
      generated_sql = answer
    print(f"{generated_sql=}")

    try:
      generated_sql_result = db.run(generated_sql)
      print(f"{generated_sql_result=}")

      none_match = True
      for gold_sql in query.split(";"):
        if gold_sql == "":
          continue
        gold_sql_result = db.run(gold_sql)
        print(f"{gold_sql=}")
        print(f"{gold_sql_result=}")

        if generated_sql_result == gold_sql_result:
          correct += 1
          none_match = False
          print("Result matched!\n")
          break

      if none_match:
        print(">>>> Generated SQL run result matches none of the gold SQL run results! <<<<<\n")

    except Exception as e:
      print(e)

  print(f"{correct} correct out of {total}")
  return f"{correct} correct out of {total}"

stats = {}
db_names = ['superhero', 'card_games', 'financial', 
  'student_club', 'california_schools', 'european_football_2', 
  'thrombosis_prediction', 'codebase_community',
  'formula_1', 'toxicology', 'debit_card_specializing']
#db_names = ['cars', 'university', 'world']
db_jsons = [[{'question': d['question'], 'SQL': d['SQL']} for d in data if d['db_id'] == db_name] for db_name in db_names]
db_names = [f'{db_name}.sqlite' for db_name in db_names]

for n, db_name in enumerate(db_names):
  db = SQLDatabase.from_uri(f"sqlite:///{db_name}", sample_rows_in_table_info=0)
  for llm in [deepseekv3]: #llama33_70b, llama31_405b]:
    print(f"Evaluating {db_name} with {llm=}")
    stats[(db_name, llm.model_name)] = text2sql_eval(llm, db, db_jsons[n])
      # db_specific_prompt="Make sure the literal string in the returned SQL query is in lower case.")
    print("====================\n")


print(stats)
