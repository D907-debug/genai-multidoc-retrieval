## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
# Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

## AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

## PROBLEM STATEMENT:
The challenge is to develop an agent that can efficiently retrieve and synthesize information from a large corpus of documents, ensuring that it answers queries with precision and relevance, leveraging LlamaIndex for effective retrieval and summarization.

## DESIGN STEPS:
Gather a set of research articles or documents relevant to the topic.
Preprocess the data by converting the articles into a suitable format (e.g., plain text or structured format).
Tokenize the content and remove any irrelevant or noisy information.

### STEP 1:
Use LlamaIndex (formerly known as GPT Index) to create an index for the documents.
LlamaIndex will help build an optimized index for efficient retrieval, making it easy to query multiple documents at once.
Incorporate features like semantic search to improve relevance and accuracy of retrieval.

### STEP 2:
Develop the query interface where users can input questions related to the research articles.
Integrate the query interface with the LlamaIndex-powered retrieval system.
Process the retrieved documents to extract relevant information and synthesize a concise response, potentially using additional techniques like summarization.

### STEP 3:
Test the system with a range of diverse queries to evaluate its performance in terms of accuracy, relevance, and conciseness of responses.
Collect feedback and refine the system based on test results.

### PROGRAM:
```py
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/forum?id=6xnZ048nAM",
    "https://openreview.net/forum?id=IQGxkTCJmt",
    "https://openreview.net/forum?id=Sa6ivpFokl",
]

papers = [
    "1pdf.pdf",
    "2pdf.pdf",
    "3pdf.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
    initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the evaluation dataset used in AI powered, "
    "and then tell me about the evaluation results"
)response = agent.query("Give me a summary of both 1pdf and 2pdf")
print(str(response))
urls = [
    "https://openreview.net/forum?id=6xnZ048nAM",
    "https://openreview.net/forum?id=IQGxkTCJmt",
    "https://openreview.net/forum?id=Sa6ivpFokl",
    "https://openreview.net/forum?id=ucxIedDtfr",
    "https://openreview.net/forum?id=Ly64ZoB604"
]

papers = [
    "1pdf.pdf",
    "2pdf.pdf",
    "3pdf.pdf",
    "4pdf.pdf",
    "5pdf.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    # define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
tools = obj_retriever.retrieve(
    "Tell me about the content in 3pdf  and 5pdf"
)
tools[2].metadata
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the evaluation dataset used "
    "in Small Language Agentic Machine and compare it against  AI powered Video Insights and Knowledge Interface"
)
print(str(response))

```
### OUTPUT:
#### 3 paper model
<img width="1321" height="691" alt="image" src="https://github.com/user-attachments/assets/54a1bb31-c7e1-46b1-9c73-df3e51164120" />
#### 5 paper model
<img width="1359" height="371" alt="image" src="https://github.com/user-attachments/assets/34ce36b9-daa5-4e23-934b-dd09e04b043e" />


### RESULT:
