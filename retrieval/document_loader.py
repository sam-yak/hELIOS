"""
Document loader for creating BM25 index from materials database.
"""

import json
from typing import List
from langchain.schema import Document

def load_documents_from_json(json_path: str = "materials_database.json") -> List[Document]:
    """
    Load all materials from JSON and convert to LangChain Documents.
    This is used for BM25 indexing in hybrid retrieval.
    
    Args:
        json_path: Path to materials database JSON
        
    Returns:
        List of Document objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    
    for material_name, properties in data.items():
        # Build comprehensive content string (same as in ingest_v2.py)
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
        
        # Create metadata
        metadata = {
            "source": f"Materials Database - {material_name}",
            "material_name": material_name,
            "category": properties.get('category', 'Unknown')
        }
        
        # Add numeric properties to metadata
        numeric_properties = [
            'density', 'tensile_strength_ultimate', 'tensile_strength_yield',
            'modulus_of_elasticity', 'thermal_conductivity', 'melting_point',
            'cost_per_kg_usd', 'sustainability_score'
        ]
        
        for prop in numeric_properties:
            if prop in properties and properties[prop] not in [None, 'N/A', '']:
                try:
                    metadata[prop] = float(properties[prop])
                except (ValueError, TypeError):
                    pass
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents
