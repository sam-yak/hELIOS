import os
import json
import shutil
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define paths
DB_PATH = "db"
DATA_PATH = "materials_database.json"

def create_documents_from_unified_json(data_path):
    """
    Loads materials from unified JSON and converts into LangChain Documents
    with comprehensive, filterable metadata.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    
    for material_name, properties in data.items():
        # Build comprehensive content string
        content = f"Material: {material_name}\n"
        content += f"Category: {properties.get('category', 'Unknown')}\n\n"
        content += f"Description: {properties.get('material_notes', 'No description available')}\n\n"
        
        # Physical Properties
        content += "Physical Properties:\n"
        content += f"- Density: {properties.get('density', 'N/A')} g/cc\n"
        
        # Mechanical Properties
        content += "\nMechanical Properties:\n"
        content += f"- Tensile Strength (Ultimate): {properties.get('tensile_strength_ultimate', 'N/A')} MPa\n"
        content += f"- Tensile Strength (Yield): {properties.get('tensile_strength_yield', 'N/A')} MPa\n"
        content += f"- Modulus of Elasticity: {properties.get('modulus_of_elasticity', 'N/A')} GPa\n"
        
        # Thermal Properties
        content += "\nThermal Properties:\n"
        content += f"- Thermal Conductivity: {properties.get('thermal_conductivity', 'N/A')} W/m-K\n"
        content += f"- Melting Point: {properties.get('melting_point', 'N/A')} °C\n"
        
        # Economic Data
        content += "\nEconomic Data:\n"
        content += f"- Cost: ${properties.get('cost_per_kg_usd', 'N/A')} per kg\n"
        
        # Sustainability
        content += "\nSustainability:\n"
        content += f"- Score: {properties.get('sustainability_score', 'N/A')}/10\n"
        content += f"- Notes: {properties.get('sustainability_notes', 'No information available')}\n"
        
        # Applications
        if 'common_applications' in properties:
            content += "\nCommon Applications:\n"
            for app in properties['common_applications']:
                content += f"- {app}\n"
        
        # Create comprehensive metadata for filtering
        metadata = {
            "source": f"Materials Database - {material_name}",
            "material_name": material_name,
            "category": properties.get('category', 'Unknown')
        }
        
        # Add all numerical properties as metadata for filtering
        numeric_properties = [
            'density',
            'tensile_strength_ultimate',
            'tensile_strength_yield',
            'modulus_of_elasticity',
            'thermal_conductivity',
            'melting_point',
            'cost_per_kg_usd',
            'sustainability_score'
        ]
        
        for prop in numeric_properties:
            if prop in properties and properties[prop] not in [None, 'N/A', '']:
                try:
                    # Convert to float for proper filtering
                    metadata[prop] = float(properties[prop])
                except (ValueError, TypeError):
                    # Skip if conversion fails
                    pass
        
        # Create Document
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

def main():
    """
    Main function to create the vector database from unified materials database.
    """
    print("=" * 70)
    print("HELIOS MATERIALS DATABASE INGESTION v2.0")
    print("=" * 70)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: {DATA_PATH} not found!")
        print(f"   Please ensure materials_database.json is in the project root.")
        return
    
    # Clean up old database
    if os.path.exists(DB_PATH):
        print(f"\n🗑️  Removing old database at {DB_PATH}...")
        shutil.rmtree(DB_PATH)
        print("   ✅ Old database removed.")
    
    # Load and create documents
    print(f"\n📖 Loading materials from {DATA_PATH}...")
    documents = create_documents_from_unified_json(DATA_PATH)
    print(f"   ✅ Created {len(documents)} documents with rich metadata.")
    
    # Split documents (mostly for consistency, our docs are already well-sized)
    print("\n✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"   ✅ Split into {len(texts)} chunks.")
    
    # Load embeddings model
    print("\n🤖 Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("   ✅ Embedding model loaded.")
    
    # Create ChromaDB vector store
    print("\n💾 Ingesting documents into ChromaDB...")
    print("   (This may take 1-2 minutes for 100 materials)")
    
    vector_store = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=DB_PATH
    )
    
    print("   ✅ Vector store created and persisted.")
    
    # Verify database
    print("\n✅ INGESTION COMPLETE!")
    print(f"   Database location: {os.path.abspath(DB_PATH)}")
    print(f"   Total materials: {len(documents)}")
    print(f"   Total chunks: {len(texts)}")
    
    # Show sample material categories
    categories = {}
    for doc in documents:
        cat = doc.metadata.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Materials by Category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {cat}: {count} materials")
    
    print("\n" + "=" * 70)
    print("Ready to use! Run 'python run.py' to start the server.")
    print("=" * 70)

if __name__ == "__main__":
    main()
