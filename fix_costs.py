"""
Fix unrealistic property values in generated materials database.
The generator used broad ranges per family — this script applies
tighter, material-specific corrections.
"""

import json
import re
import random

random.seed(99)

def r(low, high):
    return round(random.uniform(low, high), 2)

# ── Cost corrections based on material name patterns ──
# Format: (regex_pattern, (min_cost, max_cost))
COST_RULES = [
    # Precious metals — very expensive
    (r"Platinum|Rhodium|Iridium|Ruthenium|Osmium", (15000, 75000)),
    (r"Gold 24K|Gold 22K|Gold Fine", (55000, 65000)),
    (r"Gold 18K", (35000, 45000)),
    (r"Gold 14K|Gold 10K", (20000, 30000)),
    (r"Palladium(?!.*Nickel| Silver)", (40000, 50000)),
    (r"Silver Fine|Silver Sterling", (600, 900)),
    (r"Silver-", (500, 800)),
    (r"Rhenium", (1500, 3000)),
    (r"Hafnium", (800, 1200)),
    (r"Germanium", (1000, 1500)),
    (r"Gallium", (200, 400)),
    (r"Indium", (200, 350)),
    
    # Refractory metals
    (r"Tantalum", (250, 400)),
    (r"Niobium", (35, 55)),
    (r"Tungsten Heavy Alloy", (40, 80)),
    (r"Tungsten(?! Heavy)", (25, 45)),
    (r"Molybdenum", (30, 55)),
    
    # Zinc alloys — cheap
    (r"Zamak|ZA-\d|Zinc Pure|Zinc Alloy", (2.5, 5.0)),
    
    # Tin and solder
    (r"Tin Pure|Tin-Lead|Lead-Free Solder|Solder", (15, 35)),
    
    # Cobalt-chrome
    (r"Cobalt-Chrome|Stellite", (60, 120)),
    
    # Magnesium
    (r"Magnesium", (4.0, 12.0)),
    
    # Bismuth, antimony, cadmium
    (r"Bismuth", (8, 15)),
    (r"Antimony", (5, 12)),
    (r"Cadmium", (3, 8)),
    
    # Nickel alloys
    (r"Inconel|Hastelloy|Waspaloy|Rene \d|Nimonic|Udimet|MAR-M", (30, 120)),
    (r"Monel", (20, 40)),
    (r"Nickel 20[01357]|Alloy \d", (15, 60)),
    (r"Invar|Kovar|Mu-Metal|Permalloy", (20, 50)),
    
    # Titanium
    (r"Titanium Grade [1-4]\b|Titanium Grade [1-4] ", (25, 45)),
    (r"Titanium", (35, 120)),
    
    # Stainless steels
    (r"Stainless Steel Duplex", (6, 18)),
    (r"Stainless Steel.*PH|Stainless Steel Custom|Stainless Steel A-286", (8, 25)),
    (r"Stainless Steel 254|Stainless Steel AL-6XN|Stainless Steel 654", (12, 30)),
    (r"Stainless Steel 4[0-9]", (3, 8)),
    (r"Stainless Steel 3[0-9]", (3.5, 10)),
    (r"Stainless Steel", (3, 12)),
    
    # Carbon & alloy steels — cheap
    (r"Carbon Steel A36", (0.7, 1.2)),
    (r"Carbon Steel 10[0-2]\d", (0.6, 1.5)),
    (r"Carbon Steel 10[3-6]\d", (0.8, 2.0)),
    (r"Carbon Steel 10[7-9]\d", (1.0, 2.5)),
    (r"Alloy Steel", (1.2, 6.0)),
    (r"Spring Steel", (1.5, 5.0)),
    (r"HSLA Steel", (0.8, 4.0)),
    (r"Carbon Steel|Steel Grade", (0.6, 5.0)),
    
    # Tool steels
    (r"Tool Steel [WOTL]", (5, 15)),
    (r"Tool Steel [ADS]", (8, 20)),
    (r"Tool Steel [HM]", (12, 55)),
    (r"Tool Steel", (6, 30)),
    
    # Aluminum — moderate
    (r"Aluminum Cast", (3, 8)),
    (r"Aluminum 1\d{3}", (2.5, 5.0)),
    (r"Aluminum 2\d{3}", (4, 9)),
    (r"Aluminum 3\d{3}", (2.5, 5.0)),
    (r"Aluminum 5\d{3}", (3, 7)),
    (r"Aluminum 6\d{3}", (3, 7)),
    (r"Aluminum 7\d{3}", (4, 12)),
    (r"Aluminum 8\d{3}", (3, 8)),
    (r"Aluminum Alloy", (3, 10)),
    (r"Aluminum", (2.5, 10)),
    
    # Copper alloys
    (r"Beryllium Copper|Copper C17", (50, 75)),
    (r"Copper C1[0-1]", (8, 12)),
    (r"Cupronickel", (12, 25)),
    (r"Nickel Silver", (10, 20)),
    (r"Bronze", (8, 18)),
    (r"Brass", (5, 12)),
    (r"Copper", (6, 15)),
    
    # Ceramics
    (r"Tungsten Carbide", (40, 120)),
    (r"Boron Carbide|Boron Nitride.*Cubic", (80, 250)),
    (r"Silicon Nitride.*Hot Pressed", (60, 120)),
    (r"Silicon Carbide.*CVD|Silicon Carbide.*Hot Pressed", (50, 150)),
    (r"Alumina 99\.[5-9]|Alumina 99%", (20, 45)),
    (r"Alumina 9[5-8]", (12, 30)),
    (r"Alumina [89]\d", (8, 20)),
    (r"Zirconia", (25, 60)),
    (r"Aluminum Nitride", (30, 80)),
    (r"Macor|Shapal|Machinable", (40, 90)),
    (r"Porcelain|Steatite|Cordierite|Mullite|Forsterite|Spinel", (5, 25)),
    (r"Ceramic Compound", (10, 100)),
    (r"Silicon Carbide|Silicon Nitride", (20, 60)),
    
    # Composites
    (r"Carbon Fiber.*Ultra-High|Carbon Fiber.*M55|Carbon Fiber.*M46", (150, 500)),
    (r"Carbon Fiber.*High Mod|Carbon Fiber.*M40", (80, 200)),
    (r"Carbon Fiber|Carbon/PEEK|Carbon/PPS", (30, 120)),
    (r"Aramid|Kevlar.*Composite", (25, 80)),
    (r"Boron/Epoxy|Quartz/Epoxy", (100, 300)),
    (r"Ceramic Matrix", (200, 500)),
    (r"Metal Matrix", (30, 100)),
    (r"S-Glass|S2-Glass|R-Glass", (12, 30)),
    (r"Natural Fiber Composite", (3, 12)),
    (r"Glass Fiber|Glass/|E-Glass|Fiberglass|FR-4|G-\d+", (5, 20)),
    (r"Composite System", (8, 150)),
    
    # Thermoplastics
    (r"PEEK|PEI.*Ultem|PAI.*Torlon|PI.*Vespel|PI.*Kapton", (40, 450)),
    (r"LCP|PPA|PPSU|PPS", (15, 60)),
    (r"PVDF|ETFE|FEP|PFA|PCTFE", (20, 80)),
    (r"Polysulfone|Polyethersulfone", (12, 35)),
    (r"Polycarbonate|PPO|PPE", (3, 10)),
    (r"PBT|PET(?! )|Nylon|Acetal|POM", (3, 12)),
    (r"ABS|ASA|SAN", (2, 6)),
    (r"PMMA|Acrylic", (3, 8)),
    (r"TPU", (4, 15)),
    (r"EVA", (2, 5)),
    (r"Polystyrene|HIPS", (1.2, 3.0)),
    (r"Polyethylene|LDPE|LLDPE|MDPE|HDPE|UHMWPE|XLPE", (1.0, 8.0)),
    (r"Polypropylene", (1.0, 4.0)),
    (r"PVC|CPVC", (1.0, 4.0)),
    (r"Polymer Blend", (1.5, 25.0)),
    (r"HPP Grade", (25, 200)),
    
    # Elastomers
    (r"FFKM|Viton.*Extreme", (150, 300)),
    (r"Viton|FKM", (30, 65)),
    (r"Fluorosilicone", (20, 45)),
    (r"Silicone.*Medical|Silicone.*Platinum|Silicone.*Conductive", (12, 35)),
    (r"Silicone|LSR|HCR|VMQ", (6, 18)),
    (r"Nitrile.*HNBR|Hydrogenated", (15, 30)),
    (r"Nitrile|NBR|XNBR", (3, 8)),
    (r"EPDM", (2.5, 6.0)),
    (r"Neoprene|CR ", (4, 10)),
    (r"Polyurethane Elastomer", (5, 20)),
    (r"Butyl|Chlorobutyl|Bromobutyl|IIR|CIIR|BIIR", (3, 8)),
    (r"SBR|BR Rubber", (1.5, 4.0)),
    (r"Hypalon|CSM|Chlorosulfonated", (8, 18)),
    (r"Natural Rubber", (2, 5)),
    (r"Elastomer Compound", (3, 50)),
    
    # Glass & construction
    (r"Sapphire Glass", (50, 120)),
    (r"Fused Quartz|Fused Silica", (20, 60)),
    (r"Glass Ceramic", (15, 50)),
    (r"Gorilla Glass", (10, 30)),
    (r"Glass.*Fiber", (2, 8)),
    (r"Glass", (0.5, 10)),
    (r"Concrete.*UHPC|Ultra-High", (0.3, 0.8)),
    (r"Concrete", (0.08, 0.25)),
    (r"Granite|Marble|Quartzite|Slate", (0.5, 5.0)),
    (r"Limestone|Sandstone|Basalt|Travertine", (0.2, 2.0)),
    (r"Cement|Mortar", (0.08, 0.20)),
    (r"Brick", (0.15, 0.60)),
    (r"Gypsum|Calcium Silicate", (0.3, 1.5)),
    (r"Construction Material", (0.1, 15)),
    
    # Natural materials
    (r"Ebony|Rosewood|Purpleheart|Wenge|Zebrawood", (10, 30)),
    (r"Teak|Mahogany|Walnut|Cherry|Ipe", (5, 18)),
    (r"Oak|Maple|Ash|Beech|Birch|Hickory", (1.5, 5.0)),
    (r"Pine|Spruce|Fir|Cedar|Hemlock|Redwood|Larch|Cypress", (0.5, 3.0)),
    (r"Balsa", (8, 15)),
    (r"Plywood|OSB|MDF|HDF|Particle|LVL|Glulam|CLT|Parallam|I-Joist", (0.5, 4.0)),
    (r"Bamboo", (0.8, 4.0)),
    (r"Cork", (8, 20)),
    (r"Leather", (10, 40)),
    (r"Silk Fiber", (20, 60)),
    (r"Hemp|Flax|Jute|Sisal|Kenaf|Cotton|Wool|Coconut", (0.5, 5.0)),
    (r"Bone|Shell|Nacre", (5, 15)),
    (r"Bio-Material", (1, 15)),
    (r"Wood", (0.5, 15)),
]


def fix_cost(name, current_cost):
    """Apply the first matching cost rule."""
    for pattern, (lo, hi) in COST_RULES:
        if re.search(pattern, name, re.IGNORECASE):
            return r(lo, hi)
    return current_cost  # No rule matched, keep original


def main():
    path = "materials_database.json"
    with open(path, "r") as f:
        db = json.load(f)

    fixed = 0
    for name, props in db.items():
        old_cost = props.get("cost_per_kg_usd", 0)
        new_cost = fix_cost(name, old_cost)
        if new_cost != old_cost:
            props["cost_per_kg_usd"] = new_cost
            fixed += 1

        # Also fix yield > ultimate (shouldn't happen)
        uts = props.get("tensile_strength_ultimate", 0)
        ys = props.get("tensile_strength_yield", 0)
        if ys > uts and uts > 0:
            props["tensile_strength_yield"] = round(uts * random.uniform(0.65, 0.92), 0)

    with open(path, "w") as f:
        json.dump(db, f, indent=2)

    print(f"✅ Fixed costs for {fixed}/{len(db)} materials")
    print(f"   Saved to {path}")
    print(f"   Next: commit, push, and Railway will rebuild automatically")


if __name__ == "__main__":
    main()
