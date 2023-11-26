# EvaDBVertexAIExt

This is a project which creates an application that utilizes features within VertexAI to allow the user to query a dataset of Duke University e-Books and generate unique titles based on their prompt and the result of their query. 

The first part of the application implements semantic search on a dataset of Duke University e-Books using text-embeddings. Semantic search involves using the significance of words, expressions, and their context to discover the most pertinent outcomes. It relies on vector embeddings to effectively align the user's query with the most closely related result.

This is the type of query that is being evaluatated in this stage of the app:
```
SELECT * from title_set
ORDER BY similarity_value(“sample text”, x) 
LIMIT k
```

The second part of the application implements few-shot learning with large-language models to generate unique book titles based on the prompt the user has supplied as well as the examples they’ve queried from the e-Book dataset.

