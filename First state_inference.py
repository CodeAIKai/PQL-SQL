import torch
import re
import pandas as pd
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
from torch import cuda
from sql_metadata import Parser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import StoppingCriteria
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_model_path="/root/autodl-tmp/model/deepseek-ai/deepseek-coder-6.7b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,  
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/DTS-SQL-2A42/deepseek_fine",torch_dtype = torch.bfloat16)
model = model.merge_and_unload()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.encode(' ;')

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [6203]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids
    
def append_string_to_file(text, file_path):
  with open(file_path, 'a') as file:
      file.write(text + '\n')

def remove_spaces(text):
  return re.sub(r'\s+', ' ', text)

def call_mistral(inputs):
  output_tokens = model.generate(inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, stopping_criteria = [EosListStoppingCriteria()])
  return tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True)

df=pd.read_csv("XXX.csv")
results = []
for index, row in tqdm(df.iterrows(), total=len(df)):
  question = row['question']
  query = row['query']
  database_schema = row['database_schema']
  db_id = row['db_id']
  user_message = f"""Given the following SQL tables, your job is to determine the columns and tables that the question is referring to. 
IMPORTANT INSTRUCTIONS:
1. Only output the table names, nothing else
2. Do not include any explanations, prefixes, or additional text
3. If multiple tables are needed, list them separated by commas
4. Never include column names in your response
Now analyze this:
{database_schema}
###
Question: {question}
"""
  messages = [
      {"role": "user", "content": user_message.strip()}
  ]
  inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",add_generation_prompt=True,tokenize = True).to(model.device)
  response = call_mistral(inputs)
  if ";" in response:
    response = response.split(";")[0]
    if "Tables:" in response:
      response = response.split("Tables:")[1]
  response = re.sub(r'\s+', ' ', response).strip()
  try:
    ref_rables = ", ".join(Parser(query).tables)
  except Exception:
    continue
  print("\n")
  print(response)
  print(ref_rables)
  print("============================")
  results.append([response, ref_rables, query,row['question'],row['db_id']])
  new_df = pd.DataFrame(results, columns = ['predicted_tables','reference_tables','query','question','db_id'])

total_samples = len(new_df)
total_accuracy = 0
filtered_accuracy = 0
total_precision = 0
total_recall = 0

for index, row in new_df.iterrows():    
    if not row['predicted_tables'] or pd.isna(row['predicted_tables']):
        continue
    predicted_tables = row['predicted_tables'].split(", ")
    reference_tables = row['reference_tables'].split(", ")    
    predicted_tables = [x.lower().replace("--","").replace("**","").strip() for x in predicted_tables]
    reference_tables = [x.lower().strip() for x in reference_tables]
    if set(predicted_tables) == set(reference_tables):
        total_accuracy += 1
    true_positives = len(set(predicted_tables) & set(reference_tables))
    false_positives = len(set(predicted_tables) - set(reference_tables))
    false_negatives = len(set(reference_tables) - set(predicted_tables))

    if true_positives == len(reference_tables):
        filtered_accuracy += 1
    
    if len(predicted_tables) > 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    
    total_precision += precision
    total_recall += recall

# Calculate average precision and recall
avg_precision = total_precision / total_samples
avg_recall = total_recall / total_samples

# Calculate total accuracy
accuracy = total_accuracy / total_samples
filtered_accuracy = filtered_accuracy / total_samples

print("Total Accuracy:", accuracy)
print("Filtered Accuracy:", filtered_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)

new_df.to_csv("XXX.csv", index=False)
