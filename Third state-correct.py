import pandas as pd
from tqdm import tqdm
import re
import wandb
import requests
import json
# Load the results from phase 2
df = pd.read_csv("XXX.csv")

wandb.init(project="sql-generation", name="deepseek-api-sql-correction")
DEEPSEEK_API_KEY = "XXX"
DEEPSEEK_API_URL = "XXX"

headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}
def generate_correction_prompt(question, db_id, schema, generated_sql):
    sql_correction_prompt = """Analyze the given SQL query and determine if optimization is needed based on these criteria:
1. For extremum queries (MAX/MIN), rewrite using ORDER BY + LIMIT pattern
2. Ensure only requested columns appear in SELECT
3. Validate table-column relationships
4. Verify JOIN conditions use proper foreign keys
5. Prefer INNER JOIN over LEFT JOIN when possible

Example Corrections:

-- Example 1: Maximum value query --
Question: Which department has the highest average salary?
Original SQL: SELECT d.dept_name FROM departments d JOIN salaries s ON d.dept_no = s.dept_no 
              GROUP BY d.dept_name HAVING AVG(s.amount) = (SELECT MAX(avg_sal) FROM 
              (SELECT AVG(amount) as avg_sal FROM salaries GROUP BY dept_no))
Optimized SQL: SELECT d.dept_name FROM departments d JOIN salaries s ON d.dept_no = s.dept_no
               GROUP BY d.dept_name ORDER BY AVG(s.amount) DESC LIMIT 1

-- Example 2: Column selection refinement --
Question: List employee names in the sales department
Original SQL: SELECT * FROM employees e JOIN departments d ON e.dept_id = d.dept_id 
              WHERE d.dept_name = 'Sales'
Optimized SQL: SELECT e.emp_name FROM employees e JOIN departments d ON e.dept_id = d.dept_id
               WHERE d.dept_name = 'Sales'

-- Example 3: JOIN optimization --
Question: Find customers with active orders
Original SQL: SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id = o.cust_id
              WHERE o.status = 'Active'
Optimized SQL: SELECT c.customer_id FROM customers c JOIN orders o ON c.customer_id = o.cust_id
               WHERE o.status = 'Active'

Correction Guidelines:
1. STRUCTURAL INTEGRITY:
   - Verify all referenced tables/columns exist in schema
   - Ensure proper JOIN conditions using foreign keys
   - Remove unnecessary columns from SELECT

2. QUERY OPTIMIZATION:
   - Convert subqueries to JOINs when possible
   - Replace HAVING with WHERE for non-aggregate filters
   - Use EXISTS instead of IN for large datasets

3. SEMANTIC CORRECTNESS:
   - Confirm aggregate functions match question requirements
   - Validate comparison operators (e.g., = vs LIKE)
   - Check temporal filters use proper date functions

4. PERFORMANCE CONSIDERATIONS:
   - Add appropriate indexes to WHERE/JOIN columns
   - Limit result sets when possible
   - Avoid SELECT * in production queries

Current Task:
Question: {question}
Original SQL: {sql_query}

Please analyze and provide corrected SQL following these principles:"""

    prompt = f"""You are a SQL expert tasked with validating and correcting SQL queries. 

Database Schema (db_id: {db_id}):
{schema}

Original Question:
{question}

Generated SQL Query to Validate/Correct:
{generated_sql}

{correction_instructions}

Please carefully analyze the generated SQL query above. If it is correct and fully answers the question, return it exactly as is. If it has any issues, provide the corrected SQL query following these rules:

1. Return ONLY the corrected SQL query, nothing else
2. Do not include any explanations, comments, or additional text
3. Do not include ```sql or ``` markers
4. The query should be syntactically correct SQLite SQL
5. If the query ends with a semicolon, remove it

some examples:{sql_prompt}

Corrected SQL Query:"""
    
    return prompt

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
corrected_results = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    db_id = row['db_id']
    schema = row['filtered_database_schema']
    generated_sql = row['generated_query']
    reference_sql = row['reference_query']
    
    if pd.isna(generated_sql) or not generated_sql.strip():
        corrected_results.append([generated_sql, reference_sql, question, db_id, schema])
        continue
    correction_prompt = generate_correction_prompt(question, db_id, schema, generated_sql)
    corrected_sql = call_deepseek_api(correction_prompt.strip())
    if corrected_sql:
        corrected_sql = corrected_sql.replace("```sql", "").replace("```", "").strip()
        if corrected_sql.endswith(";"):
            corrected_sql = corrected_sql[:-1].strip()
        corrected_sql = re.sub(r'\s+', ' ', corrected_sql).strip()
    else:
        corrected_sql = generated_sql 
    
    print("*******************************************************************************")
    print(f"Original SQL: {generated_sql}")
    print(f"Corrected SQL: {corrected_sql}")
    print(f"Reference SQL: {reference_sql}")
    print("============================")
    
    wandb.log({
        "example": index,
        "question": question,
        "original_query": generated_sql,
        "corrected_query": corrected_sql,
        "reference_query": reference_sql,
        "db_id": db_id
    })
    
    corrected_results.append([corrected_sql, reference_sql, question, db_id, schema])

corrected_df = pd.DataFrame(corrected_results, 
                           columns=['corrected_query', 'reference_query', 'question', 'db_id', 'filtered_database_schema'])
corrected_df.to_csv("/root/autodl-tmp/new-PQL-SQL/correct/deepseek-test/deepseek_api_corrected.csv", index=False)

wandb.log({"corrected_results": wandb.Table(dataframe=corrected_df)})

with open("Predicted.txt", "w") as pred_file, open("Gold.txt", "w") as gold_file:
    for index, row in corrected_df.iterrows():
        print(f"Processing the {index}th row")
        if pd.isna(row['corrected_query']):
            print(row['corrected_query'])
            sql_query = input("give me the correct SQL query: ")
            sql_query = re.sub(r'\s+', ' ', sql_query).strip()
            pred_file.write(sql_query + "\n")
            gold_file.write(row['reference_query'] + "\t" + row['db_id'] + "\n")
        elif row['corrected_query'][:6].upper() == "SELECT":
            pred_file.write(re.sub(r'\s+', ' ', row['corrected_query']).strip() + "\n")
            gold_file.write(row['reference_query'] + "\t" + row['db_id'] + "\n")
        else:
            print(row['corrected_query'])
            sql_query = input("give me the correct SQL query: ")
            sql_query = re.sub(r'\s+', ' ', sql_query).strip()
            pred_file.write(sql_query + "\n")
            gold_file.write(row['reference_query'] + "\t" + row['db_id'] + "\n")

wandb.finish()