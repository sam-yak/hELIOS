import json
import os

def verify_migration():
    """
    Verifies the migration from old two-file system to new unified system.
    Shows what changed and what was added.
    """
    
    print("=" * 70)
    print("HELIOS DATA MIGRATION VERIFICATION")
    print("=" * 70)
    
    # Check if old files exist
    old_scraped = "scraped_data.json"
    old_enrichment = "enrichment_data.json"
    new_unified = "materials_database.json"
    
    print("\n📁 Checking file status...")
    
    files_status = {
        old_scraped: os.path.exists(old_scraped),
        old_enrichment: os.path.exists(old_enrichment),
        new_unified: os.path.exists(new_unified)
    }
    
    for file, exists in files_status.items():
        status = "✅ Found" if exists else "❌ Missing"
        print(f"   {status}: {file}")
    
    if not files_status[new_unified]:
        print("\n❌ ERROR: materials_database.json not found!")
        print("   Please save the materials_database.json artifact first.")
        return
    
    # Load new unified database
    with open(new_unified, 'r') as f:
        new_data = json.load(f)
    
    print(f"\n✅ New unified database loaded: {len(new_data)} materials")
    
    # Load old data if available
    old_materials = set()
    if files_status[old_scraped]:
        with open(old_scraped, 'r') as f:
            old_scraped_data = json.load(f)
            old_materials.update(old_scraped_data.keys())
            print(f"   Old scraped_data.json: {len(old_scraped_data)} materials")
    
    if files_status[old_enrichment]:
        with open(old_enrichment, 'r') as f:
            old_enrichment_data = json.load(f)
            print(f"   Old enrichment_data.json: {len(old_enrichment_data)} materials")
    
    # Analyze new database structure
    print("\n📊 New Database Analysis:")
    
    # Count by category
    categories = {}
    for material, data in new_data.items():
        cat = data.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n   Materials by Category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {cat}: {count}")
    
    # Check data completeness
    print(f"\n   Data Completeness:")
    
    complete_materials = 0
    materials_with_cost = 0
    materials_with_sustainability = 0
    materials_with_applications = 0
    
    for material, data in new_data.items():
        # Check if all core properties exist
        core_props = ['density', 'tensile_strength_ultimate', 'thermal_conductivity']
        if all(prop in data and data[prop] not in [None, 'N/A', ''] for prop in core_props):
            complete_materials += 1
        
        if 'cost_per_kg_usd' in data and data['cost_per_kg_usd'] not in [None, 'N/A', '']:
            materials_with_cost += 1
        
        if 'sustainability_score' in data and data['sustainability_score'] not in [None, 'N/A', '']:
            materials_with_sustainability += 1
        
        if 'common_applications' in data and len(data.get('common_applications', [])) > 0:
            materials_with_applications += 1
    
    print(f"   - Complete technical data: {complete_materials}/{len(new_data)} ({complete_materials/len(new_data)*100:.1f}%)")
    print(f"   - With cost data: {materials_with_cost}/{len(new_data)} ({materials_with_cost/len(new_data)*100:.1f}%)")
    print(f"   - With sustainability scores: {materials_with_sustainability}/{len(new_data)} ({materials_with_sustainability/len(new_data)*100:.1f}%)")
    print(f"   - With applications listed: {materials_with_applications}/{len(new_data)} ({materials_with_applications/len(new_data)*100:.1f}%)")
    
    # Show new materials added
    if old_materials:
        new_materials = set(new_data.keys()) - old_materials
        print(f"\n🆕 New Materials Added: {len(new_materials)}")
        if new_materials:
            print(f"   Sample new additions:")
            for material in sorted(new_materials)[:10]:
                cat = new_data[material].get('category', 'Unknown')
                print(f"   - {material} ({cat})")
            if len(new_materials) > 10:
                print(f"   ... and {len(new_materials) - 10} more")
    
    # Sample material check
    print(f"\n🔍 Sample Material Check:")
    sample_material = list(new_data.keys())[0]
    sample_data = new_data[sample_material]
    
    print(f"\n   Material: {sample_material}")
    print(f"   Category: {sample_data.get('category', 'N/A')}")
    print(f"   Density: {sample_data.get('density', 'N/A')} g/cc")
    print(f"   Yield Strength: {sample_data.get('tensile_strength_yield', 'N/A')} MPa")
    print(f"   Cost: ${sample_data.get('cost_per_kg_usd', 'N/A')}/kg")
    print(f"   Sustainability: {sample_data.get('sustainability_score', 'N/A')}/10")
    
    if 'common_applications' in sample_data:
        print(f"   Applications: {', '.join(sample_data['common_applications'][:3])}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 70)
    
    print("\n📋 Next Steps:")
    print("   1. Backup old files (optional):")
    print("      mv scraped_data.json scraped_data.json.old")
    print("      mv enrichment_data.json enrichment_data.json.old")
    print("")
    print("   2. Run new ingestion:")
    print("      python ingest_v2.py")
    print("")
    print("   3. Update main.py to remove enrichment_data.json dependency")
    print("")
    print("   4. Test with: python run.py")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    verify_migration()
