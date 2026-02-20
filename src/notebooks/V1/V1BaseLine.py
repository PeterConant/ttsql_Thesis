from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import ast
from textwrap import dedent
from langchain_community.utilities import SQLDatabase


# Define LLM
llm = ChatOllama(
    model="qwen3:1.7b", #  phi3, gemma3:12b, gpt-oss:20b, qwen3:1.7b,
    temperature=0,
    base_url="http://localhost:11434/"
)

# Define Database
db = SQLDatabase.from_uri(r"sqlite:///C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev\minidev_0703\minidev\MINIDEV\dev_databases\debit_card_specializing\debit_card_specializing.sqlite")



# Define tools

# This tool to be executed manual to retrieve all database tables
@tool
def get_schema() -> str:
    """Retrieves all available tables in database by running "SHOW TABLES;" statement.
    
    Args:
        No Args
    """
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    tables = db.run(query)
    
    tables_list = ast.literal_eval(tables)

    database_info = []

    for table_tuple in tables_list:
        table_name = table_tuple[0]
        
        # Get schema
        schema_query = f"PRAGMA table_info({table_name});"
        schema = db.run(schema_query)
        
        # Get first 3 rows
        sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
        sample_data = db.run(sample_query)
        
        # Format for LLM
        table_info = dedent(f"""
            Table: {table_name}
            Schema: {schema}
            Sample Data (first 3 rows): {sample_data}
            ---
            """)
        database_info.append(table_info)

    # Combine all table information
    formatted_output = "\n".join(database_info)

    return formatted_output

@tool
def execute_query(query: str) -> str:
    """Execute a SQL SELECT query and return the results.
    
    Only SELECT statements are permitted for data safety.
    Returns formatted results or error messages.
    
    Args:
        query: The SQL SELECT query to execute
    """
    # Strip whitespace and check query type
    query_stripped = query.strip()
    query_upper = query_stripped.upper()
    
    # Block non-SELECT queries
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    if any(query_upper.startswith(keyword) for keyword in dangerous_keywords):
        return f"Error: Only SELECT queries are allowed. Detected forbidden operation."
    
    if not query_upper.startswith('SELECT'):
        return "Error: Query must start with SELECT."
    
    try:
        result = db.run(query_stripped)
        if not result or result.strip() == '':
            return "Query executed successfully but returned no results."
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}\nQuery was: {query_stripped}"
    
tools = [execute_query]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from typing_extensions import Literal

def user_node(state: MessagesState):
    """Represents the user"""

    return {"messages":state["messages"]}

def get_schema_node(state: dict):
    """Performs the tool call"""

    observation = get_schema.invoke({})
    result = ToolMessage(content=observation, tool_call_id="0000001")
    return {"messages":result}


dialect = 'SQLite'

# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    system_message_content = """You are a helpful SQL generation agent designed to generate {dialect} to handle the user task.""".format(dialect=dialect)

    return {
        "messages":[
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=system_message_content
                    )
                ]
                + state["messages"]
            )
        ]
    }


def execution_tool_node(state: dict):
    """Performs the SQL statement execution tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages":result}

def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop base upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop
    return END