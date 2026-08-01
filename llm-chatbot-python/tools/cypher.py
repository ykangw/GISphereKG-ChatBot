from langchain_neo4j import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

from graph import read_only_graph

# {schema} is introspected from the live database, so it already carries the exact
# labels, property names and relationship directions. This template only adds what
# introspection cannot express: naming conventions and how to match on names.
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j developer translating questions about the GISphere knowledge
graph of geography and GIS programs into Cypher.
Use only the labels, properties and relationship types given in the schema.

Schema:
{schema}

Conventions:
- A question about a professor, researcher, scholar or faculty member refers to a
  People node.
- Never return the embedding property.
- Some property names contain spaces, and the Chinese name of a People node ends with
  one. Wrap those in backticks exactly as the schema spells them, for example
  p.`Research Interests` and p.`NAME_CN `.

Matching names:
- Never match a university, city or country name with =. Stored names carry prefixes
  and variants, so "University of Hong Kong" does not equal "The University of Hong
  Kong" and an exact match returns nothing.
- Match with toLower(...) CONTAINS toLower(...) instead. For universities also try the
  ABBR property.
- A loose match can span several institutions. Always return the name of the entity
  you matched on so the answer can be grouped by it, and order the results by that
  name first.
- Add an explicit LIMIT when the question is open ended.

Examples:

1. Professors at a university:
MATCH (p:People)-[:worksAt]->(u:University)
WHERE toLower(u.NAME_EN) CONTAINS toLower("Hong Kong")
   OR toLower(u.ABBR) CONTAINS toLower("Hong Kong")
RETURN u.NAME_EN AS university, p.NAME_EN AS professor,
       p.`Research Interests` AS interests, p.URL AS url
ORDER BY university, professor

2. Professors working on a research interest:
MATCH (p:People)-[:hasResearchInterestOf]->(r:ResearchInterest)
WHERE toLower(r.research_interest) CONTAINS toLower("remote sensing")
MATCH (p)-[:worksAt]->(u:University)
RETURN r.research_interest AS interest, u.NAME_EN AS university,
       p.NAME_EN AS professor
ORDER BY interest, university, professor

3. Universities in a country, with their city:
MATCH (u:University)-[:isIn]->(c:City)-[:isIn]->(co:Country)
WHERE toLower(co.NAME_EN) CONTAINS toLower("Germany")
RETURN co.NAME_EN AS country, c.NAME_EN AS city, u.NAME_EN AS university, u.ABBR AS abbr
ORDER BY country, city, university

4. Departments of a university:
MATCH (d:Department)-[:isIn]->(u:University)
WHERE toLower(u.NAME_EN) CONTAINS toLower("Wisconsin")
RETURN u.NAME_EN AS university, d.NAME_EN AS department
ORDER BY university, department

Question:
{question}
"""

# The default QA prompt demonstrates a flat list and tells the model the context is
# authoritative, which pushes it to dump every row. A loose name match can return
# rows from several institutions, so ask for them grouped instead.
CYPHER_QA_TEMPLATE = """
You are turning the result of a knowledge graph query into an answer.
Use only the information given; it is authoritative. If it is empty, say you could not
find anything in the GISphere knowledge graph and suggest rephrasing.

The query matched names loosely, so the results may cover more than one university,
city or research interest. When they do, group the answer under each one and put the
closest match to the question first. Do not invent entries and do not drop any.

Information:
{context}

Question: {question}
Answer: """


def cypher_qa(llm):
    """Create GraphCypherQAChain with dynamic LLM"""
    return GraphCypherQAChain.from_llm(
        llm,
        graph=read_only_graph,
        cypher_prompt=PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE),
        qa_prompt=PromptTemplate.from_template(CYPHER_QA_TEMPLATE),
        # Rows beyond top_k are dropped in Python after the query has run, with no
        # warning, so a low value silently hides matches: one university alone can
        # have close to 50 professors.
        top_k=200,
        verbose=True,
        allow_dangerous_requests=True
    )
