#from dotenv import load_dotenv  
#load_dotenv()

# =================================================================================================
# HELIOS ENGINEERING ASSISTANT - V2 WITH UNIFIED DATABASE
# =================================================================================================

import os
import json
from typing import List
from datetime import datetime
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

# Import our logging utility
from utils.logger import setup_logger, log_query_metrics

# --- LOGGING SETUP ---
logger = setup_logger("helios", "helios.log")

# --- 1. API DATA MODELS ---
class ChatHistory(BaseModel): role: str; content: str
class QueryRequest(BaseModel): question: str; chat_history: List[ChatHistory]
class SourceDocument(BaseModel): source: str; content: str
class QueryResponse(BaseModel): answer: str; detected_material: str | None = None; sources: List[SourceDocument]
class ExportRequest(BaseModel): material_name: str; export_format: str

# --- 2. ENVIRONMENT & DATA SETUP ---
logger.info("🚀 Starting Helios V2 initialization...")

if "OPENAI_API_KEY" not in os.environ:
    logger.critical("❌ OPENAI_API_KEY environment variable not set!")
    raise ValueError("FATAL ERROR: OPENAI_API_KEY environment variable not set.")

logger.info("✅ OpenAI API key found")

# Load unified materials database
MATERIALS_DB_PATH = "materials_database.json"
try:
    with open(MATERIALS_DB_PATH, 'r') as f:
        materials_database = json.load(f)
    logger.info(f"✅ Loaded unified materials database: {len(materials_database)} materials")
except FileNotFoundError:
    logger.error(f"❌ {MATERIALS_DB_PATH} not found! Using empty database.")
    materials_database = {}
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

# --- 4. THE ANALYTICAL & SELECTION AGENT (SELF-QUERY RETRIEVER) ---
try:
    logger.info("Setting up self-query retriever...")
    
    metadata_field_info = [
        AttributeInfo(name="material_name", description="The common name of the material", type="string"),
        AttributeInfo(name="category", description="Material category (e.g., 'Aluminum Alloys', 'Stainless Steels')", type="string"),
        AttributeInfo(name="tensile_strength_ultimate", description="The ultimate tensile strength in MPa", type="float"),
        AttributeInfo(name="tensile_strength_yield", description="The yield strength in MPa", type="float"),
        AttributeInfo(name="density", description="The density in g/cc", type="float"),
        AttributeInfo(name="thermal_conductivity", description="The thermal conductivity in W/m-K", type="float"),
        AttributeInfo(name="melting_point", description="The melting point in Celsius", type="float"),
        AttributeInfo(name="cost_per_kg_usd", description="Cost per kilogram in USD", type="float"),
        AttributeInfo(name="sustainability_score", description="Sustainability score from 1-10 (10 is best)", type="float"),
    ]
    document_content_description = "A comprehensive technical datasheet for an engineering material including physical, mechanical, thermal, economic, and sustainability properties."

    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=False
    )
    
    logger.info("✅ Self-query retriever configured")
    
except Exception as e:
    logger.error(f"❌ Failed to setup retriever: {e}")
    raise

# --- 5. THE ONE TRUE CHAIN: CONVERSATIONAL & ANALYTICAL RAG ---
try:
    logger.info("Building RAG chain...")
    
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
    
    logger.info("✅ RAG chain built successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to build RAG chain: {e}")
    raise

logger.info("🎉 Helios V2 Core Logic Initialized. The Agent is online and ready.")

# --- 6. FASTAPI APPLICATION SETUP ---
app = FastAPI(title="Helios V2 - Engineering Assistant")

@app.get("/", include_in_schema=False)
async def read_index():
    logger.debug("Serving frontend index.html")
    try:
        with open('frontend/index.html', 'r') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to serve index.html: {e}")
        raise

@app.post("/query", summary="Query the engineering assistant", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    start_time = datetime.now()
    query_text = request.question
    
    logger.info(f"📥 New query received: '{query_text[:100]}...'")
    
    try:
        # Convert chat history to LangChain format
        langchain_chat_history = [
            HumanMessage(content=msg.content) if msg.role == 'user' else AIMessage(content=msg.content) 
            for msg in request.chat_history
        ]
        
        logger.debug(f"Chat history length: {len(langchain_chat_history)}")
        
        # Execute RAG chain
        logger.debug("Invoking RAG chain...")
        result = rag_chain.invoke({
            "input": request.question, 
            "chat_history": langchain_chat_history
        })
        
        # Extract results
        answer = result.get("answer", "I am sorry, I could not generate a response.")
        source_docs = result.get("context", [])
        
        logger.debug(f"Retrieved {len(source_docs)} source documents")
        
        # Format sources
        formatted_sources = [
            SourceDocument(
                source=doc.metadata.get("source", "Unknown"), 
                content=doc.page_content
            ) 
            for doc in source_docs
        ]
        
        # Detect material for export feature (now using unified database)
        detected_material = None
        if source_docs:
            primary_material_name = source_docs[0].metadata.get('material_name')
            if primary_material_name and primary_material_name in materials_database:
                detected_material = primary_material_name
                logger.debug(f"Detected material: {detected_material}")
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Log metrics
        log_query_metrics(logger, query_text, response_time, len(source_docs), success=True)
        
        logger.info(f"✅ Query completed successfully in {response_time:.2f}s")
        
        return QueryResponse(
            answer=answer, 
            sources=formatted_sources, 
            detected_material=detected_material
        )
        
    except Exception as e:
        response_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Query failed after {response_time:.2f}s: {str(e)}", exc_info=True)
        
        # Log failed query
        log_query_metrics(logger, query_text, response_time, 0, success=False)
        
        # Return error response
        return QueryResponse(
            answer=f"An error occurred while processing your query: {str(e)}", 
            sources=[]
        )

@app.post("/export", summary="Export material data")
def export_data(request: ExportRequest):
    logger.info(f"📦 Export request for {request.material_name} as {request.export_format}")
    
    try:
        material_data = materials_database.get(request.material_name)
        
        if not material_data:
            logger.warning(f"Material not found: {request.material_name}")
            return Response(status_code=404)
        
        export_format = request.export_format
        
        if export_format == "json":
            content = json.dumps(material_data, indent=2)
            media_type = "application/json"
            filename = f"{request.material_name}.json"
        elif export_format == "csv":
            # Create CSV from material data
            content = "Property,Value\n"
            for key, value in material_data.items():
                if isinstance(value, list):
                    value = "; ".join(str(v) for v in value)
                content += f'"{key}","{value}"\n'
            media_type = "text/csv"
            filename = f"{request.material_name}.csv"
        else:
            # Plain text format
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
        logger.error(f"❌ Export failed: {str(e)}", exc_info=True)
        return Response(status_code=500)

@app.get("/health", summary="Health check endpoint")
def health_check():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "materials_count": len(materials_database),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", summary="Database statistics")
def get_stats():
    """Get statistics about the materials database"""
    if not materials_database:
        return {"error": "No materials database loaded"}
    
    # Count by category
    categories = {}
    for material, data in materials_database.items():
        cat = data.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total_materials": len(materials_database),
        "categories": categories,
        "sample_materials": list(materials_database.keys())[:10]
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🌟 Helios V2 API server started successfully")
    logger.info(f"📊 {len(materials_database)} materials loaded and ready")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Helios V2 API server shutting down")
