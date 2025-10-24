# =================================================================================================
# HELIOS ENGINEERING ASSISTANT - THE DEFINITIVE, UNBREAKABLE CORE (HELIOS PROTOCOL)
#
# This is the final and correct implementation of the Helios backend. After a series of
# catastrophic failures, this version represents a return to first principles: simplicity,
# robustness, and a singular focus on world-class performance.
#
# ARCHITECTURE: MONOLITHIC AND MIGHTY
# - All logic is contained in this single file to eliminate any possibility of module conflicts.
# - It focuses on ONE task: powering a conversational Analytical & Selection Agent.
# - All complex routing has been removed to ensure core functionality is flawless.
# - It is extensively commented to serve as a permanent, understandable blueprint.
# =================================================================================================

import os
import json
from typing import List
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. API DATA MODELS ---
class ChatHistory(BaseModel): role: str; content: str
class QueryRequest(BaseModel): question: str; chat_history: List[ChatHistory]
class SourceDocument(BaseModel): source: str; content: str
class QueryResponse(BaseModel): answer: str; detected_material: str | None = None; sources: List[SourceDocument]
class ExportRequest(BaseModel): material_name: str; export_format: str

# --- 2. ENVIRONMENT & DATA SETUP ---
if "OPENAI_API_KEY" not in os.environ: raise ValueError("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
with open("enrichment_data.json", 'r') as f: enrichment_data = json.load(f)

# --- 3. CORE AI INITIALIZATION ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="db", embedding_function=embeddings)
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")

# --- 4. THE ANALYTICAL & SELECTION AGENT (SELF-QUERY RETRIEVER) ---
metadata_field_info = [
    AttributeInfo(name="material_name", description="The common name of the material, e.g., 'Aluminum 6061-T6'", type="string"),
    AttributeInfo(name="tensile_strength_ultimate", description="The ultimate tensile strength in MPa.", type="float"),
    AttributeInfo(name="tensile_strength_yield", description="The yield strength in MPa.", type="float"),
    AttributeInfo(name="density", description="The density in g/cc.", type="float"),
    AttributeInfo(name="thermal_conductivity", description="The thermal conductivity in W/m-K.", type="float"),
    AttributeInfo(name="melting_point", description="The melting point in Celsius.", type="float"),
]
document_content_description = "A technical datasheet containing the properties of an engineering material."

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

# --- 5. THE ONE TRUE CHAIN: CONVERSATIONAL & ANALYTICAL RAG ---
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just reformulate it if needed."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise engineering assistant. Answer based ONLY on the context provided. If the context is empty or does not contain the answer, say so.\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, self_query_retriever, rephrase_prompt)
question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

print("✅ Helios Core Logic Initialized. The Agent is online and ready.")

# --- 6. FASTAPI APPLICATION SETUP ---
app = FastAPI(title="Helios - Engineering Assistant")

@app.get("/", include_in_schema=False)
async def read_index():
    with open('frontend/index.html', 'r') as f: html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/query", summary="Query the engineering assistant", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    langchain_chat_history = [HumanMessage(content=msg.content) if msg.role == 'user' else AIMessage(content=msg.content) for msg in request.chat_history]
    result = rag_chain.invoke({"input": request.question, "chat_history": langchain_chat_history})
    answer = result.get("answer", "I am sorry, I could not generate a response.")
    source_docs = result.get("context", [])
    formatted_sources = [SourceDocument(source=doc.metadata.get("source", "Unknown"), content=doc.page_content) for doc in source_docs]
    detected_material = None
    if source_docs:
        primary_material_name = source_docs[0].metadata.get('material_name')
        if primary_material_name and primary_material_name in enrichment_data:
            detected_material = enrichment_data.get(primary_material_name, {}).get("material_name")
    return QueryResponse(answer=answer, sources=formatted_sources, detected_material=detected_material)

@app.post("/export", summary="Export material data")
def export_data(request: ExportRequest):
    material_data = enrichment_data.get(request.material_name)
    if not material_data: return Response(status_code=404)
    export_format = request.export_format
    if export_format == "json": content, media_type, filename = json.dumps(material_data, indent=2), "application/json", f"{request.material_name}.json"
    elif export_format == "csv": content, media_type, filename = "Property,Value\n" + "\n".join([f'"{k}","{v}"' for k, v in material_data.items()]), "text/csv", f"{request.material_name}.csv"
    else: content, media_type, filename = "\n".join([f"{k}: {v}" for k, v in material_data.items()]), "text/plain", f"{request.material_name}.txt"
    headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
    return Response(content=content, media_type=media_type, headers=headers)
