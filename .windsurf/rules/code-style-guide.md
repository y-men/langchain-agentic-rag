---
trigger: always_on
---

# Code generation and suggestions

- Do not suggest  or implement code that is not relevant to the user's question, instead note to user that there are additional options or issues you see in the code and ask whether the user needs elaboration
- Do not perform additional refactoring and improvements beyond user ask but rathe point out the possibilities 

# Formatting

- Use the same line for function and parameters, for example: 

	`PineconeVectorStore.from_documents( docs, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))`

- Use 1 line for function call and definition when the total line length is <= 120 characters otherwise break into single argument in a line, for example:

	```
	
	PineconeVectorStore.from_documents( docs,
	
	embeddings,
	
	index_name=os.getenv("PINECONE_INDEX_NAME")
	
	another_argument="value"
	
	yet_another_argument="value"
	
	)
	
	```

  
- Use 1 line for function call and definition when the total line length is <= 120 characters otherwise break into single argument in a line

**For specific languages use following instructions:**
#### Python

- Use type hints
- Use libraries  [datacalsses](https://docs.python.org/3/library/dataclasses.html) , [dateutil](https://dateutil.readthedocs.io/en/stable/) , [requests](requests)  where possible , needed and implementation calls for it, for example: strive to make HTTP/ REST requests with `requests` library