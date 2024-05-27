from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import FewShotPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from config import OPENAI_API_KEY

# Initialize OpenAI LLM
llm = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Few-shot learning examples
examples = [
    {"question": "Find the most popular tags on Stack Overflow.", "cypher": "MATCH (t:Tag)<-[:TAGGED]-(q:Question) RETURN t.name, COUNT(q) AS question_count ORDER BY question_count DESC LIMIT 10;"},
    {"question": "How many questions have the 'python' tag?", "cypher": "MATCH (t:Tag {name: 'python'})<-[:TAGGED]-(q:Question) RETURN COUNT(q) AS question_count;"}
]

# Create few-shot prompt template
prompt_template = FewShotPromptTemplate(
    examples=examples,
    suffix="Question: {input}\nCypher Query:",
    input_variables=["input"]
)

def generate_cypher(question):
    return prompt_template.format(input=question)

# Initialize FAISS vectorstore for semantic search
vectorstore = FAISS(embeddings)
