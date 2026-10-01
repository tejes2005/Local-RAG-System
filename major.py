from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
#from langchain_huggingface import HuggingFaceEndpoint
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,

)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k" :1,
    }
)

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#      pipeline_kwargs={
#         "max_new_tokens": 400,
#     }
# )

base_llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "max_length": None,
        "return_full_text": False,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "clean_up_tokenization_spaces": False,
        "temperature":0.2,
    },
)
llm = ChatHuggingFace(llm=base_llm)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG System")

print("0 to Exit")

while True:
    query=input("you:")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content.strip()}")