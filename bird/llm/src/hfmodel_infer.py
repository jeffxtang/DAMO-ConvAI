import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///cars.sqlite", sample_rows_in_table_info=0)

def get_schema():
    return db.get_table_info()

available_memory = torch.cuda.get_device_properties(0).total_memory
print(available_memory)

# both generated good SQLs
#model_name = "defog/llama-3-sqlcoder-8b"
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

#model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit" 
# generated bad SQL:
# SELECT car_name
# FROM data
# WHERE cylinders = 8
# ORDER BY price DESC
# LIMIT 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
available_memory = torch.cuda.get_device_properties(0).total_memory
if available_memory > 20e9:
    # if you have atleast 20GB of GPU memory, run load the model in float16
    print("loading float16")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
else:
    print("loading 4 bits")
    # else, load in 4 bits – this is slower and less accurate
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
        use_cache=True,
    )

import sqlparse

def generate_query(question):

    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Generate a SQL query to answer this question: {question}

    DDL statements:
    {get_schema()}

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    The following SQL query best answers the question {question}:
    ```sql
    """

    print(f"\n{prompt=}\n***************\n")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        top_p=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # empty cache so that you do generate more results w/o memory crashing
    # particularly important on Colab – memory management is much more straightforward
    # when running on an inference service
    # return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)
    return outputs[0].split("```sql")[1].split(";")[0]

generated_sql = generate_query("Among the cars with 8 cylinders, what is the name of the one that's the most expensive?")
#print(generated_sql)
print(sqlparse.format(generated_sql, reindent=True))    
