from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from pprint import pprint
import json

with open('mini_dev\mini_dev_data\mini_dev_mysql.json') as f:
    mini_dev_mysql = json.load(f)

model = ChatOllama(
    model="qwen3:1.7b", #  phi3, gemma3:12b, gpt-oss:20b, qwen3:1.7b,
    temperature=0,
    base_url="http://localhost:11434/"
)

db = SQLDatabase.from_uri("mysql+pymysql://readonly-user:password@localhost:3306/BIRD")

toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

sql_agent_system_prompt = """
You are an agent designed to generate {dialect} to handle the user task.
Given an input question, create a syntactically correct {dialect} query.
Run the query to verifiry it answer the original question. 
Return the generated SQL.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)


agent = create_agent(
    model,
    tools,
    system_prompt=sql_agent_system_prompt,
)

for i, item in enumerate(mini_dev_mysql):
    response = agent.invoke(
        {"messages": [{"role": "user", 
                       "content": """{question} You may find this informaiton helpful: {evidence}""".format(
                            question=item['question'],
                            evidence=item['evidence'],)
        }]},
        stream_mode="values",
    )
    print("Item {i}/{total_items}".format(i=i, total_items=len(mini_dev_mysql)))
    pprint(response["messages"][-1])