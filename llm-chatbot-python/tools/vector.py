from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from graph import graph

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

def kg_qa(llm, embeddings):
    """Create a research interest retrieval chain with dynamic LLM and embeddings"""
    neo4jvector = Neo4jVector.from_existing_index(
        embeddings,
        graph=graph,
        index_name='ri_embedding',
        node_label='ResearchInterest',
        text_node_property='research_interest',
        embedding_node_property='embedding',
        retrieval_query="""
            RETURN
            node.research_interest AS text,
            score,
            {
                researchInterest: node.research_interest,
                professor: [(people)-[:hasResearchInterestOf]->(node) | people.NAME_EN]
            } AS metadata
            """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        # as_retriever defaults to k=4, far fewer than the 20 research interests the
        # agent prompt asks for.
        neo4jvector.as_retriever(search_kwargs={"k": 25}),
        question_answer_chain
    )

    # The agent's Tool expects a callable, and LCEL runnables are invoked rather
    # than called, so wrap the chain in a function.
    def run(input):
        return retrieval_chain.invoke({"input": input})

    return run
