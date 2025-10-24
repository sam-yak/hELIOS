import os
import json
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

# Define the path for the Chroma vector database
DB_PATH = "db"
DATA_PATH = "scraped_data.json"

def parse_value(value_str):
    """
    Parses a string to extract the first available float value.
    Handles ranges like '582 - 652 °C' by taking the first number.
    Returns None if no number can be found.
    """
    if not isinstance(value_str, str):
        return None
    
    # Use regex to find the first number (integer or float) in the string
    match = re.search(r'[-+]?\d*\.\d+|\d+', value_str)
    if match:
        try:
            return float(match.group())
        except (ValueError, TypeError):
            return None
    return None

def create_documents_from_json(data_path):
    """
    Loads scraped data from a JSON file and converts it into LangChain Documents
    with rich, filterable metadata.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    for material_name, properties in data.items():
        # Create a single string content from all the properties
        content = f"Datasheet for {material_name}:\n"
        
        # --- NEW: Initialize metadata with core info ---
        metadata = {
            "source": f"MatWeb - {material_name}",
            "material_name": material_name
        }

        # --- NEW: Loop through nested properties to build content AND metadata ---
        for prop_category, prop_values in properties.items():
            if isinstance(prop_values, dict):
                for key, value in prop_values.items():
                    content += f"- {key}: {value}\n"
                    # Create a clean metadata key (e.g., "Tensile Strength, Ultimate" -> "tensile_strength_ultimate")
                    meta_key = key.lower().replace(',', '').replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                    # Parse the numerical value for filtering
                    numeric_value = parse_value(value)
                    if numeric_value is not None:
                        metadata[meta_key] = numeric_value
            else:
                # Handle top-level properties like "Material Notes"
                content += f"- {prop_category}: {prop_values}\n"
        
        # Create a LangChain Document object with the enriched metadata
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    return documents

def main():
    """
    Main function to create the vector database from the scraped data.
    """
    # --- NEW: Clean up the old database directory first ---
    if os.path.exists(DB_PATH):
        print(f"Removing old database at {DB_PATH}...")
        shutil.rmtree(DB_PATH)
        print("Old database removed.")

    # Create documents with rich metadata from our JSON data source
    documents = create_documents_from_json(DATA_PATH)
    print(f"Created {len(documents)} documents with rich metadata from {DATA_PATH}.")

    # We still split the documents in case they are very large
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded.")

    # Create the Chroma vector store
    print("Ingesting documents into ChromaDB... (This may take a moment)")
    vector_store = Chroma.from_documents(
        texts, embeddings, persist_directory=DB_PATH
    )
    print("Ingestion complete. Vector store created and persisted with rich metadata.")


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run scraper.py first.")
    else:
        main()
