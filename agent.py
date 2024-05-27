from db import Neo4jConnection
from llm.py import generate_cypher, llm, vectorstore
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# Initialize Neo4j connection
conn = Neo4jConnection(uri=NEO4J_URI, user=NEO4J_USERNAME, pwd=NEO4J_PASSWORD)

def chat_bot(question):
    # Generate Cypher query using LLM
    cypher_query = generate_cypher(question)
    
    # Execute Cypher query
    results = conn.query(cypher_query)
    
    # Return results
    return results

def semantic_search(query):
    docs = vectorstore.similarity_search(query)
    return docs

# Example usage
question = "How many questions have the 'java' tag?"
response = chat_bot(question)
for record in response:
    print(record)

search_query = "Popular tags"
search_results = semantic_search(search_query)
for doc in search_results:
    print(doc)
