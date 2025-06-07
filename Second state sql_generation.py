import re
import pandas as pd
from tqdm import tqdm
import wandb
import requests
import json

wandb.login(key="XXX")
wandb.init(project="sql-generation", name="deepseek-api-sql-generation")
DEEPSEEK_API_KEY = "XXX"
DEEPSEEK_API_URL = "XXX"

headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}
def append_string_to_file(text, file_path):
  with open(file_path, 'a') as file:
      file.write(text + '\n')

def call_deepseek_api(prompt):
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 500,
        "stop": ["```"]
    }
    
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        print(f"API call failed with status code {response.status_code}")
        return None

# Load dataset
df = pd.read_csv("XXX.csv")
results = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    query = row['query']
    database_schema = row['filtered_database_schema']
    db_id = row['db_id']
    user_message = f"""You are a SQL expert. Given the following database schema and question, generate ONLY the SQLite SQL query that answers the question by following these steps:

1. Analyze the database schema to understand tables, columns, and relationships
2. Interpret the question to determine what information needs to be retrieved
3. Identify which tables need to be joined and on which columns
4. Determine any filtering conditions needed (WHERE clauses)
5. Consider if aggregation (GROUP BY) or sorting (ORDER BY) is required
6. Construct the final SQL query following SQLite syntax

Here are two examples demonstrating this thought process:

Example 1:
Database schema:
CREATE TABLE `area_code_state` (
  area_code INTEGER,
  state varchar(2)
);
CREATE TABLE `votes` (
  vote_id INTEGER,
  phone_number INTEGER PRIMARY KEY,
  state varchar(2) REFERENCES AREA_CODE_STATE(state),
  contestant_number INTEGER REFERENCES CONTESTANTS(contestant_number),
  created timestamp
);

Question: "What is the area code in which the most voters voted?"

Thought Process:
1. Need to find which area_code has the most votes
2. Votes are recorded in the votes table, but area codes are in area_code_state
3. Must join these tables on state column
4. Need to count votes per area code
5. Should order by count in descending order and take the top result

SQL Query:
SELECT t1.area_code FROM area_code_state as t1 JOIN votes as t2 ON t1.state = t2.state GROUP BY t1.area_code ORDER BY COUNT(*) DESC LIMIT 1

Example 2:
Database schema:
CREATE TABLE `dogs` (
  dog_id INTEGER,
  owner_id INTEGER REFERENCES Owners(owner_id),
  breed_code VARCHAR(10) REFERENCES Breeds(breed_code),
  name VARCHAR(50)
);
CREATE TABLE `treatments` (
  treatment_id INTEGER,
  dog_id INTEGER REFERENCES Dogs(dog_id),
  date_of_treatment DATETIME
);

Question: "List the names of the dogs of the rarest breed and their treatment dates"

Thought Process:
1. Need to find dogs with the rarest breed (lowest count)
2. Need to get their names and treatment dates
3. First identify the rarest breed using a subquery
4. Then join dogs with treatments to get the dates
5. Filter dogs by the rarest breed identified

SQL Query:
SELECT t1.name, t2.date_of_treatment FROM dogs as t1 JOIN treatments as t2 ON t1.dog_id = t2.dog_id WHERE t1.breed_code = (SELECT breed_code FROM dogs GROUP BY breed_code ORDER BY COUNT(*) ASC LIMIT 1)

Now generate SQL for this new database schema and question by following the same thought process:

Database schema:
{database_schema}

Question: {question}

Thought Process:
1. [Analyze the schema to understand tables and relationships]
2. [Interpret what information the question is asking for]
3. [Identify which tables need to be accessed/joined]
4. [Determine any filtering conditions needed]
5. [Consider if aggregation or sorting is required]
6. [Plan the final query structure]

IMPORTANT INSTRUCTIONS:
1. Return ONLY the SQL query, nothing else
2. Do not include any explanations, comments, or additional text
3. Do not include ```sql or ``` markers
4. The query should be syntactically correct SQLite SQL
5. If the query ends with a semicolon, remove it

SQL Query:"""

    
    response = call_deepseek_api(user_message.strip())
    print("*******************************************************************************")
    print(f"Raw response:{response}")

    if response:
        response = response.replace("```sql", "").replace("```", "").strip()
        if response.endswith(";"):
            response = response[:-1].strip()
        response = re.sub(r'\s+', ' ', response).strip()
    else:
        response = ""

    print("\nProcessed SQL:")
    print(response)
    print("Reference query:")
    print(query)
    print("============================")   
    wandb.log({
        "example": index,
        "question": question,
        "generated_query": response,
        "reference_query": query,
        "db_id": db_id
    })
    
    results.append([response, query, row['question'], row['db_id'],row['filtered_database_schema']])

# Save results
new_df = pd.DataFrame(results, columns=['generated_query', 'reference_query', 'question', 'db_id','filtered_database_schema'])
new_df.to_csv("XXX.csv", index=False)

wandb.log({"results": wandb.Table(dataframe=new_df)})

# Process results for evaluation
for index, row in new_df.iterrows():
    print(f"Processing the {index}th rows")
    if pd.isna(row['generated_query']):
        print(row['generated_query'])
        sql_query = input("give me the correct SQL query")
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        append_string_to_file(sql_query, "Predicted.txt")
        append_string_to_file(row['reference_query'] + "\t" + row['db_id'], "Gold.txt")
    elif row['generated_query'][:6] == "SELECT":
        append_string_to_file(re.sub(r'\s+', ' ', row['generated_query']).strip(), "Predicted.txt")
        append_string_to_file(row['reference_query'] + "\t" + row['db_id'], "Gold.txt")
    else:
        print(row['generated_query'])
        sql_query = input("give me the correct SQL query")
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        append_string_to_file(sql_query, "Predicted.txt")
        append_string_to_file(row['reference_query'] + "\t" + row['db_id'], "Gold.txt")

wandb.finish()

def append_string_to_file(text, file_path):
    with open(file_path, 'a') as file:
        file.write(text + '\n')