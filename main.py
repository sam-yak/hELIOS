# =================================================================================================
# HELIOS ENGINEERING ASSISTANT - V2 WITH HYBRID RETRIEVAL
# =================================================================================================

import os
import json
from typing import List
from datetime import datetime
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our modules
from utils.logger import setup_logger, log_query_metrics
from retrieval.document_loader import load_documents_from_json
from retrieval.hybrid_retriever import create_hybrid_retriever

# --- LOGGING SETUP ---
logger = setup_logger("helios", "helios.log")

# --- 1. API DATA MODELS ---
class ChatHistory(BaseModel): role: str; content: str
class QueryRequest(BaseModel): question: str; chat_history: List[ChatHistory]; use_hybrid: bool = True
class SourceDocument(BaseModel): source: str; content: str
class QueryResponse(BaseModel): answer: str; detected_material: str | None = None; sources: List[SourceDocument]; retrieval_method: str = "hybrid"
class ExportRequest(BaseModel): material_name: str; export_format: str
class CompareRequest(BaseModel): query: str; k: int = 5

# --- 2. ENVIRONMENT & DATA SETUP ---
logger.info("🚀 Starting Helios V2 (Hybrid Retrieval) initialization...")
logger.info(f"Environment: {os.getenv('HELIOS_ENV', 'development')}")

if "OPENAI_API_KEY" not in os.environ:
    logger.critical("❌ OPENAI_API_KEY not found in environment!")
    raise ValueError("FATAL ERROR: OPENAI_API_KEY environment variable not set.")

logger.info("✅ OpenAI API key loaded from environment")

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    logger.info("✅ LangSmith tracing enabled")

# Load materials database
MATERIALS_DB_PATH = "materials_database.json"
try:
    with open(MATERIALS_DB_PATH, 'r') as f:
        materials_database = json.load(f)
    logger.info(f"✅ Loaded unified materials database: {len(materials_database)} materials")
except Exception as e:
    logger.error(f"❌ Failed to load materials database: {e}")
    materials_database = {}

# --- 3. CORE AI INITIALIZATION ---
try:
    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("✅ Embedding model loaded")
    
    logger.info("Loading vector store from db/...")
    vector_store = Chroma(persist_directory="db", embedding_function=embeddings)
    logger.info("✅ Vector store loaded")
    
    logger.info("Initializing LLM...")
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")
    logger.info("✅ LLM initialized")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize AI components: {e}")
    raise

# --- 4. HYBRID RETRIEVAL SETUP ---
try:
    logger.info("🔧 Setting up Hybrid Retrieval System...")
    
    # Load all documents for BM25
    logger.info("   Loading documents for BM25 indexing...")
    all_documents = load_documents_from_json(MATERIALS_DB_PATH)
    logger.info(f"   ✓ Loaded {len(all_documents)} documents")
    
    # Create hybrid retriever
    hybrid_retriever = create_hybrid_retriever(vector_store, all_documents)
    logger.info("✅ Hybrid Retrieval System ready")
    
except Exception as e:
    logger.error(f"❌ Failed to setup hybrid retrieval: {e}")
    raise

# --- 5. RAG CHAIN WITH HYBRID RETRIEVAL ---
try:
    logger.info("Building RAG chain with hybrid retrieval...")
    
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

    # Create history-aware retriever using our hybrid retriever
    # Note: We'll manually handle retrieval in the query endpoint to have more control
    
    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    logger.info("✅ RAG chain built successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to build RAG chain: {e}")
    raise

logger.info("🎉 Helios V2 (Hybrid) Initialized. The Agent is online and ready.")

# --- 6. FASTAPI APPLICATION SETUP ---
app = FastAPI(title="Helios V2 - Engineering Assistant with Hybrid Retrieval")

@app.get("/", include_in_schema=False)
async def read_index():
    try:
        with open('frontend/index.html', 'r') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to serve index.html: {e}")
        raise

@app.post("/query", summary="Query with hybrid retrieval", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    start_time = datetime.now()
    query_text = request.question
    use_hybrid = request.use_hybrid
    
    logger.info(f"📥 New query: '{query_text[:100]}...' [Method: {'Hybrid' if use_hybrid else 'Semantic-only'}]")
    
    try:
        # Convert chat history
        langchain_chat_history = [
            HumanMessage(content=msg.content) if msg.role == 'user' else AIMessage(content=msg.content) 
            for msg in request.chat_history
        ]
        
        # Retrieve documents using hybrid or semantic-only
        if use_hybrid:
            retrieved_docs = hybrid_retriever.retrieve(query_text, k=5)
            method = "hybrid"
        else:
            retrieved_docs = hybrid_retriever.retrieve_semantic_only(query_text, k=5)
            method = "semantic_only"
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents using {method}")
        
        # Generate answer using the retrieved context
        answer_result = question_answer_chain.invoke({
            "input": query_text,
            "context": retrieved_docs,
            "chat_history": langchain_chat_history
        })
        
        answer = answer_result if isinstance(answer_result, str) else answer_result.get("answer", "No response generated")
        
        # Format sources
        formatted_sources = [
            SourceDocument(
                source=doc.metadata.get("source", "Unknown"), 
                content=doc.page_content
            ) 
            for doc in retrieved_docs
        ]
        
        # Detect material
        detected_material = None
        if retrieved_docs:
            primary_material_name = retrieved_docs[0].metadata.get('material_name')
            if primary_material_name and primary_material_name in materials_database:
                detected_material = primary_material_name
        
        # Log metrics
        response_time = (datetime.now() - start_time).total_seconds()
        log_query_metrics(logger, query_text, response_time, len(retrieved_docs), success=True)
        
        logger.info(f"✅ Query completed in {response_time:.2f}s using {method}")
        
        return QueryResponse(
            answer=answer, 
            sources=formatted_sources, 
            detected_material=detected_material,
            retrieval_method=method
        )
        
    except Exception as e:
        response_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Query failed: {str(e)}", exc_info=True)
        log_query_metrics(logger, query_text, response_time, 0, success=False)
        
        return QueryResponse(
            answer=f"An error occurred: {str(e)}", 
            sources=[],
            retrieval_method="error"
        )

@app.post("/compare", summary="Compare retrieval methods")
def compare_retrieval_methods(request: CompareRequest):
    """
    Compare semantic, keyword, and hybrid retrieval for a query.
    Useful for demonstrating hybrid retrieval benefits.
    """
    logger.info(f"📊 Comparison request for: '{request.query}'")
    
    try:
        comparison = hybrid_retriever.compare_methods(request.query, request.k)
        
        return JSONResponse(content={
            "query": request.query,
            "results": {
                "semantic": comparison["semantic"],
                "keyword": comparison["keyword"],
                "hybrid": comparison["hybrid"]
            },
            "analysis": {
                "semantic_count": len(comparison["semantic"]),
                "keyword_count": len(comparison["keyword"]),
                "hybrid_count": len(comparison["hybrid"]),
                "unique_to_hybrid": list(set(comparison["hybrid"]) - 
                                        set(comparison["semantic"]) - 
                                        set(comparison["keyword"]))
            }
        })
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/export", summary="Export material data")
def export_data(request: ExportRequest):
    logger.info(f"📦 Export request for {request.material_name}")
    
    try:
        material_data = materials_database.get(request.material_name)
        if not material_data:
            return Response(status_code=404)
        
        if request.export_format == "json":
            content = json.dumps(material_data, indent=2)
            media_type = "application/json"
            filename = f"{request.material_name}.json"
        elif request.export_format == "csv":
            content = "Property,Value\n"
            for key, value in material_data.items():
                if isinstance(value, list):
                    value = "; ".join(str(v) for v in value)
                content += f'"{key}","{value}"\n'
            media_type = "text/csv"
            filename = f"{request.material_name}.csv"
        else:
            content = f"Material: {request.material_name}\n\n"
            for key, value in material_data.items():
                if isinstance(value, list):
                    content += f"{key}:\n"
                    for item in value:
                        content += f"  - {item}\n"
                else:
                    content += f"{key}: {value}\n"
            media_type = "text/plain"
            filename = f"{request.material_name}.txt"
        
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        logger.info(f"✅ Export successful: {filename}")
        
        return Response(content=content, media_type=media_type, headers=headers)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return Response(status_code=500)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.1.0",
        "features": ["hybrid_retrieval", "langsmith_tracing"],
        "materials_count": len(materials_database),
        "environment": os.getenv("HELIOS_ENV", "development"),
        "langsmith_enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
def get_stats():
    if not materials_database:
        return {"error": "No materials database loaded"}
    
    categories = {}
    for material, data in materials_database.items():
        cat = data.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total_materials": len(materials_database),
        "categories": categories,
        "retrieval_method": "hybrid",
        "sample_materials": list(materials_database.keys())[:10]
    }

@app.on_event("startup")
async def startup_event():
    logger.info("🌟 Helios V2 (Hybrid Retrieval) API server started")
    logger.info(f"📊 {len(materials_database)} materials loaded")
    logger.info(f"🔧 Hybrid retrieval active (semantic + BM25)")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Helios V2 shutting down")
