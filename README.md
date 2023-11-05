# EvaDBVertexAIExt

This is a project that is attempting to integrate Vertex AI text-embeddings into EvaDB.
Specifically, we want to be able to implement a query to do a semantic search over the EvaDB dataset. 

We want to be able to support this type of query:
```
SELECT * from evadb
ORDER BY similarity_value(“sample text”, x) 
LIMIT k
```

where similarity_value is a function that takes in two strings and returns the semantic similarity between two strings. The purpose of this query would be to find the k most similar strings in the dataset to a given input string. 