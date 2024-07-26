# %%
from database import Neo4jDatabase
from data_classes import Node, Question, Chunk

# %%
db = Neo4jDatabase(
    host="bolt://localhost:7687",
    user="neo4j",
    password="admin@123",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
# %%
node = Node("Person", "1", {"text": "Peter is hateful", "name": "Peter"})
node2 = Node("Person", "2", {"text": "Bob is dumb", "name": "Bob"})
node3 = Node("Person", "3", {"text": "Marley is talented", "name": "Marley"})
node4 = Node("Person", "4", {"text": "Joe is slow", "name": "Joe"})

db.add_node(node)
db.add_node(node2)
db.add_node(node3)
db.add_node(node4)

db.add_relationship(node, node2, "HATE")
db.add_relationship(node, node3, "HATE")
db.add_relationship(node2, node3, "HATE")
db.add_relationship(node4, node3, "HATE")

db.initiate_vector_indexing("Person", "Person_Sim", 384)

db.generate_vector_index(node)
db.generate_vector_index(node2)
db.generate_vector_index(node3)
db.generate_vector_index(node4)

# %%
db.get_node_with_id("1", "Person")

# %%
db.retrieve_similar_nodes("Who hates most people?", 3, "Person_Sim")

# %%
db.retrieve_similar_nodes_neighbours("Who hates most people?", "HATE", 3, "Person_Sim")

# %%
question = Chunk(id="1", metadata={"teaxt": "Hey"})
db.add_node(question)