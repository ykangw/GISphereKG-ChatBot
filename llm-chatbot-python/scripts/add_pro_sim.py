import os
import sys
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase

threshold = 0.9
prof_sim_df = pd.read_csv(os.path.dirname(os.getcwd()) + "/data/prof_sim.csv")
scores = prof_sim_df['score'].tolist()
score_above = [s for s in scores if s > threshold]

# print(len(score_above), len(scores))

url = st.secrets['NEO4J_URI']
auth = (st.secrets['NEO4J_USERNAME'], st.secrets['NEO4J_PASSWORD'])
database = st.secrets.get('NEO4J_DATABASE', 'neo4j')

with GraphDatabase.driver(url, auth = auth) as driver:
    driver.verify_connectivity()
    for index, row in prof_sim_df.iterrows():
        prof1 = row['prof1']
        prof2 = row['prof2']
        score = row['score']
        print(index, prof1, prof2, score)
        if prof1 != prof2:
            # cypher_statement = ,
            if score > threshold:
                r = driver.execute_query(
                    "match (p1:People{PersonID:'" + str(int(prof1)) + "'}), (p2:People{PersonID:'" + str(int(prof2)) +"'}) create (p1)-[r:isSimilarTo{score:" + str(score) + "}]->(p2) return r",
                    database_ = database
                )
                print(r)
#


