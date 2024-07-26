from data_classes import Node

from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings


class Neo4jDatabase:
    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        embedding_model: str
    ):
        self.database = Neo4jGraph(host, user, password)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)


    def add_node(
        self,
        node: Node
    ):
        """
        Add a new node to the database
        """

        self.database.query(
            f"""
            MERGE(node:{node.type_label} {{id: {node.id}}})
                ON CREATE SET 
                    {", ".join(["node." + key + "='" + value  + "'" for key, value in zip(node.metadata.keys(), node.metadata.values())])}
            """
        )


    def add_relationship(
        self,
        root_node: Node,
        rel_node: Node,
        relationship: str,
    ):
        """
        Add a directed relationship from a root node to a rel node
        """

        self.database.query(
            f"""
            MATCH (root_node:{root_node.type_label}), (rel_node:{rel_node.type_label})
            WHERE {root_node.id} = root_node.id 
                AND {rel_node.id} = rel_node.id
            WITH root_node, rel_node
                MERGE (root_node)-[r:{relationship}]->(rel_node)
            """
        )


    def initiate_vector_indexing(
        self, 
        node_type: str, 
        index_name: str, 
        dimension: int, 
        similarity_metric: str = "COSINE"
    ):
        """
        Initiate vector indexing for a node type
        """

        self.database.query(
            f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (node:{node_type}) ON (node.embedding) 
                OPTIONS {{ 
                    indexConfig: {{
                        `vector.dimensions`: {dimension},
                        `vector.similarity_function`: '{similarity_metric}' 
                    }}
                }}
            """
        )

    
    def generate_vector_index(self, node: Node):
        """
        Generate vector index for a node
        """

        embedding = self.embedding_model.embed_query(node.metadata["text"])
        self.database.query(
            f"""
            MATCH (node:{node.type_label}) 
            WHERE node.embedding IS NULL 
                AND node.id = {node.id}
            CALL db.create.setNodeVectorProperty(node, "embedding", {embedding})
            """
        )


    def retrieve_similar_nodes(
        self, 
        query: str, 
        top_k: int, 
        index_name: str,
        threshold: float = 0.0,
    ):
        """
        Retrieve nodes with text semantically similar to input query
        """

        query_embedding = self.embedding_model.embed_query(query)

        similar_nodes = self.database.query(
            f"""
            CALL db.index.vector.queryNodes("{index_name}", {top_k}, {query_embedding})
                YIELD node, score
            RETURN node.text AS text, score
            """
        )

        similar_nodes = [node for node in similar_nodes if node["score"] >= threshold]

        return similar_nodes
    

    def retrieve_similar_nodes_neighbours(
        self, 
        query: str, 
        relationship: str,
        top_k: int, 
        index_name: str
    ):
        """
        Retrieve neighbours of nodes with text semantically similar to input query
        """

        query_embedding = self.embedding_model.embed_query(query)

        docs = self.database.query(
            f"""
            CALL db.index.vector.queryNodes("{index_name}", {top_k}, {query_embedding})
                YIELD node, score
            MATCH (node)-[r:{relationship}]->(neighbourNode)
            RETURN 
                node.text AS text, 
                score, 
                COLLECT(neighbourNode.text) AS neighbours
            """
        )

        return docs
    

    def get_node_with_id(
        self, 
        id: str,
        type_label: str
    ):
        node_info = self.database.query(
            f"""
            MATCH (node:{type_label})
            WHERE node.id = {id}
            RETURN node
            """
        )

        node = Node(
            type_label=type_label,
            id=id,
            metadata=node_info[0]
        )

        return node
