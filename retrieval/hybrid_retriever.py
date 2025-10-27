"""
Hybrid Retrieval System for Helios
Combines semantic search (ChromaDB) with keyword search (BM25)
for improved recall on both fuzzy and precise queries.
"""

from typing import List
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
import logging

logger = logging.getLogger("helios")

class HybridMaterialRetriever:
    """
    Hybrid retrieval combining semantic and keyword search.
    
    Use Cases:
    - Semantic: "lightweight corrosion-resistant metal" 
    - Keyword: "density 2.70 g/cc" or "yield strength 276 MPa"
    - Hybrid: Gets best results from both approaches
    """
    
    def __init__(self, vector_store: Chroma, documents: List[Document], 
                 semantic_weight: float = 0.6, keyword_weight: float = 0.4):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: ChromaDB vector store
            documents: All material documents for BM25
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        self.vector_store = vector_store
        self.documents = documents
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        logger.info("🔧 Initializing Hybrid Retriever...")
        self.setup_retrievers()
        logger.info("✅ Hybrid Retriever ready")
    
    def setup_retrievers(self):
        """Set up semantic, keyword, and ensemble retrievers."""
        
        # 1. Semantic Retriever (ChromaDB with embeddings)
        self.semantic_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Top 5 results
        )
        logger.debug("  ✓ Semantic retriever configured")
        
        # 2. Keyword Retriever (BM25)
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents
        )
        self.bm25_retriever.k = 5  # Top 5 results
        logger.debug("  ✓ BM25 keyword retriever configured")
        
        # 3. Ensemble Retriever (combines both)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.bm25_retriever],
            weights=[self.semantic_weight, self.keyword_weight]
        )
        logger.debug(f"  ✓ Ensemble retriever configured (semantic: {self.semantic_weight}, keyword: {self.keyword_weight})")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: User's search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        logger.debug(f"Hybrid retrieval for: '{query[:50]}...'")
        
        # Get results from ensemble
        results = self.ensemble_retriever.get_relevant_documents(query)
        
        # Limit to k results and remove duplicates
        seen_materials = set()
        unique_results = []
        
        for doc in results:
            material_name = doc.metadata.get('material_name')
            if material_name and material_name not in seen_materials:
                seen_materials.add(material_name)
                unique_results.append(doc)
                if len(unique_results) >= k:
                    break
        
        logger.debug(f"  → Retrieved {len(unique_results)} unique documents")
        return unique_results
    
    def retrieve_semantic_only(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve using only semantic search (for comparison)."""
        return self.semantic_retriever.get_relevant_documents(query)[:k]
    
    def retrieve_keyword_only(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve using only keyword search (for comparison)."""
        return self.bm25_retriever.get_relevant_documents(query)[:k]
    
    def compare_methods(self, query: str, k: int = 5) -> dict:
        """
        Compare all three retrieval methods for analysis.
        
        Returns:
            Dictionary with results from each method
        """
        logger.info(f"📊 Comparing retrieval methods for: '{query}'")
        
        semantic_results = self.retrieve_semantic_only(query, k)
        keyword_results = self.retrieve_keyword_only(query, k)
        hybrid_results = self.retrieve(query, k)
        
        comparison = {
            "query": query,
            "semantic": [doc.metadata.get('material_name') for doc in semantic_results],
            "keyword": [doc.metadata.get('material_name') for doc in keyword_results],
            "hybrid": [doc.metadata.get('material_name') for doc in hybrid_results]
        }
        
        logger.info(f"  Semantic: {comparison['semantic']}")
        logger.info(f"  Keyword:  {comparison['keyword']}")
        logger.info(f"  Hybrid:   {comparison['hybrid']}")
        
        return comparison


def create_hybrid_retriever(vector_store: Chroma, documents: List[Document]) -> HybridMaterialRetriever:
    """
    Factory function to create a hybrid retriever.
    
    Args:
        vector_store: ChromaDB vector store
        documents: All documents for BM25 indexing
        
    Returns:
        Configured HybridMaterialRetriever instance
    """
    return HybridMaterialRetriever(
        vector_store=vector_store,
        documents=documents,
        semantic_weight=0.6,  # Favor semantic slightly
        keyword_weight=0.4     # But include keyword matching
    )
