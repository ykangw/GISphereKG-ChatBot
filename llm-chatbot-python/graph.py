from typing import Any, Dict, List

import streamlit as st
from neo4j import Query, RoutingControl
from langchain_neo4j import Neo4jGraph

credentials = dict(
    url = st.secrets['NEO4J_URI'],
    username = st.secrets['NEO4J_USERNAME'],
    password = st.secrets['NEO4J_PASSWORD'],
    database = st.secrets.get('NEO4J_DATABASE', 'neo4j')
)


class ReadOnlyNeo4jGraph(Neo4jGraph):
    """Neo4jGraph that sends every query as a read.

    Cypher for the QA chain is written by the LLM and executed as-is. Routing the
    query as a read makes the server reject a generated CREATE, SET or DELETE with
    Neo.ClientError.Statement.AccessMode, so the guarantee does not depend on the
    prompt or on the client behaving.
    """

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        data, _, _ = self._driver.execute_query(
            Query(text=query, timeout=self.timeout),
            database_=self._database,
            parameters_=params,
            routing_=RoutingControl.READ
        )
        return [r.data() for r in data]


# Connect to Neo4j. Chat history writes Session and Message nodes, so it needs the
# writable connection; anything running LLM-generated Cypher uses the read-only one.
graph = Neo4jGraph(**credentials)
read_only_graph = ReadOnlyNeo4jGraph(**credentials)
