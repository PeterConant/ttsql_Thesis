from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import ToolMessage

from sentence_transformers import SentenceTransformer

import numpy as np
import faiss
import json
import ast

database = SQLDatabase.from_uri("mysql+pymysql://readonly-agent:bird@localhost:3306/bird_mini_dev", max_string_length = 3000)


# Get Tables
def get_tables():
    """Performs the tool call"""
    return database.get_usable_table_names()

@tool
def get_tables_tool():
    """Performs the tool call"""

    tables = get_tables()
    response = "SHOW TABLES from bird_mini_dev:\n" + str(tables)
    result = ToolMessage(content=response, tool_call_id="get_tables_node")
    return {"messages":result}


# Get Schemas
def get_table_schemas_and_samples(tables:list[str]):
    return database.get_table_info_no_throw(tables)

# Tools
@tool
def get_table_schemas_tool(table_list: list[str]) -> str:
    """Get detailed information about a list of tables including columns, types, constraints, and the top 3 rows.
    
    Args:
        table_names: List of table name strings

    Returns:
        Formatted string containing schema and sample data for each table
    """

    return get_table_schemas_and_samples(table_list)



################## EMBEDDINGS SIMILARITY #########################

file_name = 'csv_table_descriptions'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.load(f'src/V4/embeddings/{file_name}.npy')
with open(f'src/V4/embeddings/{file_name}.json', 'r') as f:
    table_metadata = json.load(f) #list of {'table':table_name, 'chunk'=chunk_to_be(or_has_been)_embedded}

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype('float32'))

def get_tables_semantic_search(query, k=10):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k)

    retrieved_tables = []
    for rank, idx in enumerate(I[0]):

        table_name = table_metadata[idx]['table']
        retrieved_tables.append(table_name)

    return retrieved_tables