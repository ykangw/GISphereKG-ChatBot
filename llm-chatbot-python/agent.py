from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.tools import Tool
from langchain_classic.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory

from graph import credentials
from utils import get_session_id
from tools.vector import kg_qa
from tools.cypher import cypher_qa


system_prompt = """

# Role Definition:\n
You are a knowledge graph information retrieval, similarity measure, and recommendation expert for a GIS program.  
Be as helpful as possible and return as much information as possible. 
Your responses should strictly use information from the provided Neo4j database.\n\n

# Key Guidelines:\n
When some words (close, similar, recommendation/recommend) are mentioned in the question, use 'research_interest_vector_search', calculate semantic similarity of each 'ResearchInterest' node with the extracted input, and return at least 10 similar 'ResearchInterest' nodes.
Do not find information outside of this Neo4j database. 
Do not answer any questions that do not relate to our knowledge graph. 
Only the information provided in the knowledge graph or you can use research interest property to do recommendation based on the embeddings. 
When the question contains Professor(s), it is always "People" node instead of Professor node. 
When it comes to relationship between People and ResearchInterest, the relationship in the Cypher statement generation should be "hasResearchInterestOf". 
When using "research_interest_vector_search" to find closely aligned or similar research interests according to the input, return at least top 20 "ResearchInterest" nodes.
When using "graph_cypher_qa" involving 'research_interest' attribute, please make sure that the research interest from input is contained in the database,
Otherwise, please turn to use "research_interest_vector_search" tool to find out similar research interests first. then use 'graph_cypher_qa' tool.

When asking question referring to research interests, professors, and other additional information (university, city, etc),
first please use research_interest_vector_search tool to find similar research interests,
then use graph_cypher_qa tool with the input of these returned research interests and return the information from the question.\n\n

Nodes:\n
- For "ResearchInterest" node, return only the 'research_interest' attribute. 
- If the query mentions Professor(s), interpret it as "People" nodes. Return attributes such as 'NAME_CN ', 'NAME_EN', 'Research Interests', and 'URL'.
- For "City" nodes, return "NAME_CN", "NAME_EN", "lat" and "lon".
- For "Continent" nodes, return "NAME_CN" and "NAME_EN". 
- For "Country" nodes, return "NAME_CN" and "NAME_EN". 
- For "Department" nodes, return "NAME_CN" and "NAME_EN". 
- For "University" nodes, return "NAME_CN", "NAME_EN", "NAME_Local", "NAME_Other", "Description_CN", "Description_EN", "URL", and "ABBR".\n\n

Relationships:\n
- hasResearchInterestOf:   
  Connects "People" nodes to "ResearchInterest" nodes.\n
- isIn:
  Represents: \n
    1. (City)-[isIn]->(Country)\n
    2. (Country)-[isIn]->(Continent)\n
    3. (Department)-[isIn]->(University)\n
    4. (University)-[isIn]->(City)\n
- worksAt:
  Connects "People" nodes to "University" nodes.\n\n

There is no relationship between People nodes, so answer questions about similar or
recommended professors by comparing their research interests with
research_interest_vector_search.\n\n
"""

# The tools are advertised to the model through the API's own tool schema, so the
# prompt no longer lists them or describes an output format. chat_history and
# agent_scratchpad are message lists rather than strings: the scratchpad is where
# the executor replays tool calls and their results back to the model.
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

'''
llm: this is set to the instance of ChatOpenAI
tools:
    - Tools are objects that can be used by the Agent to perform actions
    - You will create multiple tools that can be used by the Agent to perform specific tasks.
    However, a tool is required for "general chat" so the agent can respond to a user's input when no other tool
'''


def get_memory(session_id):
    """Conversation history for one browser session.

    The executor is rebuilt on every message, so the history has to live outside
    of it. Storing it in Neo4j keyed by the Streamlit session id keeps it isolated
    per user and lets it survive an app restart. window is the number of previous
    exchanges replayed into the prompt.

    Built from credentials rather than the shared graph on purpose: this object is
    discarded after every message, and its __del__ closes whatever driver it holds.
    Handed the shared graph, the first discarded history would close the connection
    the rest of the app is using.
    """
    return Neo4jChatMessageHistory(session_id=session_id, window=15, **credentials)


def create_agent_executor(llm, embeddings):
    """Create agent components with dynamic LLM and embeddings"""
    # StrOutputParser keeps the tool's return value a plain string. Calling
    # llm.invoke directly would hand the agent an AIMessage object instead.
    general_chat = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a GIS program and research interest expert."),
            ("human", "{input}"),
        ]
    ) | llm | StrOutputParser()

    # Tool names become function names in the API's tool schema, which only allows
    # [a-zA-Z0-9_.-] -- no spaces. The model routes on the name and description
    # alone, so both need to say what the tool is for and what input it expects.
    tools = [
        Tool.from_function(
            name="general_chat",
            description=(
                "For general GIS chat not covered by the other tools. "
                "Input is the user's question as plain text."
            ),
            func=general_chat.invoke,
            return_direct=True
        ),
        Tool.from_function(
            name="research_interest_vector_search",
            description=(
                "Find ResearchInterest nodes that are semantically similar to a "
                "topic, using vector search over research interest embeddings. "
                "Use this first for questions about similarity, closeness or "
                "recommendations, or when a research interest may not match the "
                "database wording exactly. Input is a research topic as plain text."
            ),
            func=kg_qa(llm, embeddings),  # Modified kg_qa call
            return_direct=False
        ),
        Tool.from_function(
            name="graph_cypher_qa",
            description=(
                "Answer questions about GIS programs, professors, universities, "
                "departments, cities and countries by generating and running a "
                "Cypher query against the knowledge graph. Use this for factual "
                "lookups and for anything needing exact values or counts. Input is "
                "the question as plain text."
            ),
            func=cypher_qa(llm),  # Modified cypher_qa call
            return_direct=False
        )
    ]

    # Native tool calling: the model returns structured tool_calls instead of text
    # to be parsed, so no stop sequence and no output-format parsing is involved.
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

def generate_response(prompt, llm, embeddings):
    """Updated to use dynamic agent executor"""
    agent_executor = create_agent_executor(llm, embeddings)
    chat_agent = RunnableWithMessageHistory(
        agent_executor,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    response = chat_agent.invoke(
        {"input": prompt},
        {"configurable": {"session_id": get_session_id()}},
    )
    return response['output']