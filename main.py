# Car Manual RAG - MG ZS Warning Messages
# hooking up the manual HTML to Claude so drivers can ask about warnings

# --- setup stuff (already done in the notebook) ---
# loader is loaded above
# we just need the RAG pipeline here

# 1) split the docs into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_chunks = text_splitter.split_documents(car_docs)

# 2) shove chunks into chromadb (vector store)
vectorstore = Chroma.from_documents(
    documents=docs_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3) make a retriever that grabs relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4) prompt template - keep it simple
template = """You are a helpful assistant for MG ZS car owners.
Use the following context from the car manual to answer the question.
If you don't know from the context, just say so.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 5) chain it all together
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6) the actual query we care about
query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"

# run it and save
answer = rag_chain.invoke(query)

# if u wanna see it while testing
print(answer.content)
