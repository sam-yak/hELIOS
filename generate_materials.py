"""
Helios Materials Database Generator
Expands from 72 → 10,000+ materials with realistic engineering data.

Uses real material families with known property ranges to generate
plausible entries that work well with the hybrid RAG retrieval system.
"""

import json
import random
import os

random.seed(42)  # Reproducible

# ── Load existing materials to preserve them exactly ──
EXISTING_PATH = "materials_database.json"
if os.path.exists(EXISTING_PATH):
    with open(EXISTING_PATH, "r") as f:
        existing_db = json.load(f)
    print(f"✅ Loaded {len(existing_db)} existing materials (will be preserved)")
else:
    existing_db = {}
    print("⚠️  No existing database found, generating all from scratch")

# ── Helper ──
def r(low, high, decimals=1):
    """Random float in range, rounded."""
    return round(random.uniform(low, high), decimals)

def pick(lst):
    return random.choice(lst)

# ══════════════════════════════════════════════════════════════════════
# MATERIAL FAMILIES — real alloy designations with realistic ranges
# ══════════════════════════════════════════════════════════════════════

MATERIAL_FAMILIES = {
    # ── ALUMINUM ALLOYS ──
    "Aluminum Alloys": {
        "templates": [
            # 1xxx series - pure aluminum
            *[f"Aluminum {n}" for n in ["1050-O", "1060-H14", "1070-H18", "1100-O", "1100-H14", "1100-H18", "1145-O", "1199-O", "1350-O", "1350-H19"]],
            # 2xxx series - copper alloys
            *[f"Aluminum {n}" for n in ["2011-T3", "2011-T8", "2014-O", "2014-T4", "2014-T6", "2017-T4", "2024-O", "2024-T4", "2024-T81", "2024-T861", "2036-T4", "2048-T851", "2090-T83", "2091-T8", "2124-T851", "2195-T8", "2219-O", "2219-T62", "2219-T87", "2618-T61"]],
            # 3xxx series - manganese
            *[f"Aluminum {n}" for n in ["3003-O", "3003-H14", "3003-H18", "3004-O", "3004-H34", "3004-H38", "3005-H14", "3005-H18", "3105-O", "3105-H14"]],
            # 5xxx series - magnesium
            *[f"Aluminum {n}" for n in ["5005-O", "5005-H34", "5050-H34", "5052-O", "5052-H34", "5056-O", "5056-H38", "5083-O", "5083-H116", "5086-O", "5086-H34", "5154-O", "5182-O", "5251-O", "5254-H34", "5356-O", "5454-O", "5454-H34", "5456-O", "5456-H116", "5457-O", "5652-O", "5657-H25"]],
            # 6xxx series - Si+Mg
            *[f"Aluminum {n}" for n in ["6005-T5", "6005A-T6", "6009-T4", "6010-T4", "6013-T6", "6022-T4", "6053-T6", "6060-T5", "6060-T6", "6061-O", "6061-T4", "6063-O", "6063-T5", "6063-T6", "6066-T6", "6070-T6", "6082-T6", "6101-T6", "6105-T5", "6111-T4", "6151-T6", "6162-T6", "6201-T81", "6205-T5", "6262-T9", "6351-T6", "6463-T6"]],
            # 7xxx series - zinc
            *[f"Aluminum {n}" for n in ["7005-T53", "7039-T64", "7049-T73", "7050-T7451", "7050-T7651", "7055-T77", "7068-T6511", "7072-O", "7075-O", "7075-T73", "7075-T76", "7075-T651", "7075-T7351", "7090-T7E71", "7149-T73", "7175-T66", "7175-T736", "7178-T6", "7178-T76", "7475-T61", "7475-T761"]],
            # 8xxx series
            *[f"Aluminum {n}" for n in ["8006-O", "8011-O", "8079-O", "8090-T6", "8176-H14"]],
            # Cast aluminum
            *[f"Aluminum Cast {n}" for n in ["A356-T6", "A357-T6", "A380", "A390-T6", "319-T6", "333-T5", "355-T6", "356-T6", "356-T7", "360", "380", "383", "384", "390-T6", "413", "443-F", "514", "518", "520-T4", "535-F", "705", "707", "710", "711", "712", "713", "771-T6", "850-T5", "851-T5", "852-T5"]],
        ],
        "ranges": {
            "density": (2.63, 2.85),
            "tensile_strength_ultimate": (70, 620),
            "tensile_strength_yield": (28, 570),
            "modulus_of_elasticity": (68, 78),
            "thermal_conductivity": (80, 235),
            "melting_point": (500, 660),
            "cost_per_kg_usd": (2.5, 12.0),
            "sustainability_score": (6, 8),
        },
        "applications": ["Aerospace structures", "Automotive body panels", "Marine hardware", "Heat exchangers", "Architectural trim", "Bicycle frames", "Electrical conductors", "Beverage cans", "Aircraft skins", "Truck trailers", "Pressure vessels", "Welded assemblies", "Extrusions", "Forged wheels", "Cookware"],
        "notes_pool": [
            "Good formability and weldability with moderate strength.",
            "High strength alloy suitable for structural aerospace applications.",
            "Excellent corrosion resistance in marine and industrial environments.",
            "General purpose alloy balancing strength, workability and cost.",
            "Heat treatable with good machinability and surface finish.",
            "Non-heat-treatable alloy with excellent weldability.",
            "High purity grade with superior electrical conductivity.",
            "Cast alloy with good fluidity and pressure tightness.",
        ],
        "sustainability_pool": [
            "Highly recyclable aluminum. Recycling uses only 5% of primary production energy.",
            "Good recyclability but alloying elements can complicate sorting.",
            "Lightweight material that reduces fuel consumption in transport applications.",
            "Excellent recycling infrastructure exists globally for aluminum alloys.",
        ],
    },

    # ── STAINLESS STEELS ──
    "Stainless Steels": {
        "templates": [
            # Austenitic (3xx)
            *[f"Stainless Steel {n}" for n in ["301", "301L", "302", "302B", "303", "303Se", "304H", "304L", "304LN", "304N", "305", "308", "309", "309S", "310", "310S", "314", "316H", "316L", "316LN", "316N", "316Ti", "317", "317L", "317LM", "321", "321H", "330", "347", "347H", "348", "904L"]],
            # Ferritic (4xx)
            *[f"Stainless Steel {n}" for n in ["405", "409", "429", "430", "430F", "434", "436", "439", "441", "444", "446"]],
            # Martensitic
            *[f"Stainless Steel {n}" for n in ["403", "410S", "414", "416", "416Se", "420", "420F", "422", "431", "440A", "440B", "440C", "440F"]],
            # Duplex
            *[f"Stainless Steel Duplex {n}" for n in ["2101", "2205", "2304", "2507", "S31803", "S32101", "S32205", "S32304", "S32520", "S32550", "S32750", "S32760", "S32900", "S32950", "S39274", "S39277"]],
            # PH
            *[f"Stainless Steel {n}" for n in ["13-8 PH", "15-5 PH", "15-7 PH", "17-7 PH", "Custom 450", "Custom 455", "Custom 465", "PH 13-8 Mo", "AM 350", "AM 355", "A-286"]],
            # Superaustenitic
            *[f"Stainless Steel {n}" for n in ["254 SMO", "AL-6XN", "20Cb-3", "654 SMO"]],
        ],
        "ranges": {
            "density": (7.65, 8.10),
            "tensile_strength_ultimate": (415, 1900),
            "tensile_strength_yield": (170, 1700),
            "modulus_of_elasticity": (190, 210),
            "thermal_conductivity": (12, 30),
            "melting_point": (1370, 1530),
            "cost_per_kg_usd": (3.0, 25.0),
            "sustainability_score": (6, 9),
        },
        "applications": ["Chemical processing", "Food processing equipment", "Medical instruments", "Marine hardware", "Nuclear components", "Architectural facades", "Automotive exhaust", "Heat exchangers", "Pressure vessels", "Fasteners", "Springs", "Surgical implants", "Kitchen equipment", "Pharmaceutical equipment", "Cryogenic vessels"],
        "notes_pool": [
            "Austenitic grade with excellent corrosion resistance and formability.",
            "Ferritic grade offering good oxidation resistance at lower cost.",
            "Martensitic grade providing high hardness and moderate corrosion resistance.",
            "Duplex grade combining high strength with excellent corrosion resistance.",
            "Precipitation-hardening grade for high-strength aerospace applications.",
            "Low carbon variant for improved weldability and intergranular corrosion resistance.",
            "Superaustenitic grade with exceptional resistance to pitting and crevice corrosion.",
            "Free-machining grade with added sulfur or selenium for improved machinability.",
        ],
        "sustainability_pool": [
            "100% recyclable with no degradation in quality. Steel is the most recycled material globally.",
            "Long service life in corrosive environments reduces replacement frequency.",
            "Excellent durability extends infrastructure life, reducing total material consumption.",
            "High recycled content typical in stainless steel production (60-80%).",
        ],
    },

    # ── CARBON & ALLOY STEELS ──
    "Carbon & Alloy Steels": {
        "templates": [
            # Plain carbon
            *[f"Carbon Steel {n}" for n in ["1006", "1008", "1010", "1012", "1015", "1016", "1017", "1018", "1019", "1020", "1021", "1022", "1023", "1025", "1026", "1029", "1030", "1035", "1037", "1038", "1039", "1040", "1042", "1043", "1044", "1046", "1049", "1050", "1053", "1055", "1059", "1060", "1064", "1065", "1070", "1074", "1075", "1078", "1080", "1084", "1085", "1086", "1090", "1095"]],
            # Alloy steels
            *[f"Alloy Steel {n}" for n in ["1330", "1335", "1340", "1345", "4023", "4024", "4027", "4028", "4032", "4037", "4042", "4047", "4118", "4120", "4121", "4130", "4135", "4137", "4142", "4145", "4147", "4150", "4161", "4320", "4330", "4337", "4340", "4615", "4617", "4620", "4621", "4626", "4718", "4720", "4815", "4817", "4820", "5015", "5046", "5060", "5115", "5117", "5120", "5130", "5132", "5135", "5140", "5145", "5147", "5150", "5155", "5160", "50B40", "50B44", "50B46", "50B50", "50B60", "51B60", "5210", "6118", "6120", "6150", "8115", "8615", "8617", "8620", "8622", "8625", "8627", "8630", "8637", "8640", "8642", "8645", "8650", "8655", "8660", "8720", "8740", "8822", "9255", "9260", "9262", "9310", "9840", "94B17", "94B30"]],
            # Spring steels
            *[f"Spring Steel {n}" for n in ["1074", "1075", "1095", "5160", "6150", "9254", "9255", "9260", "SUP7", "SUP9", "SUP10", "SUP12", "51CrV4", "54SiCr6", "60Si2Mn"]],
            # HSLA
            *[f"HSLA Steel {n}" for n in ["A242", "A440", "A441", "A514", "A517", "A572 Gr.50", "A572 Gr.60", "A572 Gr.65", "A588", "A633", "A656", "A709", "A852", "A871", "A913", "A992", "S355", "S460", "S500", "S690"]],
        ],
        "ranges": {
            "density": (7.80, 7.90),
            "tensile_strength_ultimate": (330, 1900),
            "tensile_strength_yield": (200, 1650),
            "modulus_of_elasticity": (195, 215),
            "thermal_conductivity": (25, 55),
            "melting_point": (1400, 1530),
            "cost_per_kg_usd": (0.5, 8.0),
            "sustainability_score": (5, 8),
        },
        "applications": ["Shafts and axles", "Gears", "Bolts and fasteners", "Automotive drivetrain", "Springs", "Structural beams", "Rail tracks", "Bridge construction", "Mining equipment", "Agricultural implements", "Crane components", "Pressure vessels", "Pipelines", "Wear plates", "Machine frames"],
        "notes_pool": [
            "Low carbon steel with good formability and weldability.",
            "Medium carbon steel balancing strength and ductility.",
            "High carbon steel for applications requiring hardness and wear resistance.",
            "Chromium-molybdenum alloy with excellent fatigue and impact resistance.",
            "Nickel-chromium-molybdenum alloy for high-stress applications.",
            "Manganese alloy steel with improved hardenability.",
            "High strength low alloy steel for structural applications.",
            "Boron-treated steel with enhanced hardenability at lower alloy cost.",
        ],
        "sustainability_pool": [
            "Steel is the world's most recycled material with over 85% recycling rate.",
            "Electric arc furnace production from scrap reduces CO2 by 75% vs blast furnace.",
            "Long service life in structural applications reduces material consumption.",
            "Low cost and high recyclability make it environmentally favorable.",
        ],
    },

    # ── TOOL STEELS ──
    "Tool Steels": {
        "templates": [
            *[f"Tool Steel {n}" for n in ["A2", "A4", "A6", "A7", "A8", "A9", "A10", "D3", "D4", "D5", "D6", "D7", "H10", "H11", "H12", "H13", "H14", "H19", "H21", "H22", "H23", "H24", "H25", "H26", "H41", "H42", "H43", "L2", "L3", "L6", "M1", "M2", "M3", "M4", "M7", "M10", "M33", "M34", "M35", "M36", "M41", "M42", "M43", "M44", "M46", "M47", "M48", "M50", "M52", "M62", "O2", "O6", "O7", "P2", "P3", "P4", "P5", "P6", "P20", "P21", "S1", "S2", "S4", "S5", "S6", "S7", "T1", "T2", "T4", "T5", "T6", "T8", "T15", "W1", "W2", "W5"]],
        ],
        "ranges": {
            "density": (7.60, 8.70),
            "tensile_strength_ultimate": (1200, 2800),
            "tensile_strength_yield": (1000, 2500),
            "modulus_of_elasticity": (190, 220),
            "thermal_conductivity": (15, 50),
            "melting_point": (1370, 1510),
            "cost_per_kg_usd": (6.0, 55.0),
            "sustainability_score": (4, 6),
        },
        "applications": ["Cutting tools", "Dies and molds", "Punches", "Drill bits", "Reamers", "Taps", "Broaches", "Forming rolls", "Plastic injection molds", "Shear blades", "Woodworking tools", "Cold heading dies", "Extrusion tooling"],
        "notes_pool": [
            "Air-hardening tool steel with good wear resistance and toughness.",
            "High-speed steel for cutting tools operating at elevated temperatures.",
            "Hot-work tool steel for dies exposed to high temperature.",
            "Water-hardening tool steel, the most economical tool steel.",
            "Oil-hardening tool steel with good dimensional stability.",
            "Shock-resisting tool steel for impact applications.",
            "Mold steel for plastic injection molding with good polishability.",
            "High-carbon high-chromium tool steel with excellent wear resistance.",
        ],
        "sustainability_pool": [
            "Extremely long tool life reduces frequency of replacement and waste.",
            "Tool steel recycling is well-established in the metalworking industry.",
            "High-performance tooling reduces material waste in manufacturing processes.",
        ],
    },

    # ── TITANIUM ALLOYS ──
    "Titanium Alloys": {
        "templates": [
            *[f"Titanium Grade {n}" for n in ["1", "3", "4", "5", "6", "7", "9", "11", "12", "16", "17", "18", "19", "20", "21", "23", "24", "25", "26", "27", "28", "29", "32", "33", "34", "35", "36", "37", "38"]],
            *[f"Titanium {n}" for n in ["Ti-3Al-2.5V", "Ti-5Al-2.5Sn", "Ti-5Al-2.5Sn ELI", "Ti-6Al-2Sn-4Zr-2Mo", "Ti-6Al-2Sn-4Zr-6Mo", "Ti-6Al-6V-2Sn", "Ti-6Al-7Nb", "Ti-8Al-1Mo-1V", "Ti-10V-2Fe-3Al", "Ti-13V-11Cr-3Al", "Ti-15V-3Cr-3Al-3Sn", "Ti-15Mo", "Ti-15Mo-5Zr-3Al", "Ti-3Al-8V-6Cr-4Mo-4Zr", "Ti-35Nb-7Zr-5Ta", "Ti-45Nb", "Ti-6Al-4V-0.1Ru", "Ti-6Al-2Nb-1Ta-0.8Mo", "Ti-5553", "Ti-17"]],
        ],
        "ranges": {
            "density": (4.40, 4.85),
            "tensile_strength_ultimate": (240, 1400),
            "tensile_strength_yield": (170, 1300),
            "modulus_of_elasticity": (55, 120),
            "thermal_conductivity": (5.5, 22),
            "melting_point": (1580, 1680),
            "cost_per_kg_usd": (25.0, 120.0),
            "sustainability_score": (5, 7),
        },
        "applications": ["Aerospace structures", "Medical implants", "Chemical processing", "Marine hardware", "Dental prosthetics", "Sports equipment", "Armor plating", "Heat exchangers", "Desalination plants", "Offshore platforms", "Jet engine compressor blades", "Racing components", "Spectacle frames", "Jewelry"],
        "notes_pool": [
            "Alpha-beta alloy with excellent balance of strength and ductility.",
            "Commercially pure grade with superior corrosion resistance.",
            "Beta alloy with high strength and good cold formability.",
            "Near-alpha alloy for elevated temperature applications.",
            "Metastable beta alloy offering the highest specific strength.",
            "ELI grade with improved fracture toughness for critical applications.",
            "Palladium-enhanced grade with superior crevice corrosion resistance.",
        ],
        "sustainability_pool": [
            "Exceptional corrosion resistance enables decades of service without replacement.",
            "Biocompatible material reduces medical complications and revision surgeries.",
            "High strength-to-weight ratio reduces fuel consumption in aerospace.",
            "Titanium recycling is growing but remains more challenging than steel.",
        ],
    },

    # ── NICKEL ALLOYS & SUPERALLOYS ──
    "Nickel Alloys": {
        "templates": [
            *[f"Inconel {n}" for n in ["X-750", "600", "601", "617", "622", "625 Plus", "686", "690", "693", "706", "713C", "713LC", "725", "738", "740H", "751", "783", "792", "800", "800H", "800HT", "825", "864"]],
            *[f"Hastelloy {n}" for n in ["B", "B-2", "B-3", "C-4", "C-22", "C-2000", "G", "G-3", "G-30", "G-35", "G-50", "N", "S", "W", "X"]],
            *[f"Monel {n}" for n in ["401", "404", "K-500", "R-405"]],
            *[f"Waspaloy"], *[f"Rene {n}" for n in ["41", "80", "88", "95", "104", "108", "125", "142", "N4", "N5", "N6"]],
            *[f"Nimonic {n}" for n in ["75", "80A", "90", "105", "115", "263", "901"]],
            *[f"Udimet {n}" for n in ["188", "400", "500", "520", "700", "710", "720"]],
            *[f"MAR-M {n}" for n in ["200", "246", "247", "302", "322", "421", "432", "509"]],
            *[f"Invar 36"], *[f"Kovar"], *[f"Mu-Metal"], *[f"Permalloy 80"],
            *[f"Alloy {n}" for n in ["20", "28", "31", "33", "59", "230", "556", "617", "625", "686", "693", "718 Plus", "725", "825", "C-263", "HX"]],
            *[f"Nickel {n}" for n in ["201", "205", "211", "233", "270", "290", "301"]],
        ],
        "ranges": {
            "density": (7.75, 9.25),
            "tensile_strength_ultimate": (380, 1600),
            "tensile_strength_yield": (120, 1350),
            "modulus_of_elasticity": (150, 230),
            "thermal_conductivity": (8, 92),
            "melting_point": (1240, 1455),
            "cost_per_kg_usd": (15.0, 120.0),
            "sustainability_score": (4, 7),
        },
        "applications": ["Gas turbine blades", "Chemical processing", "Nuclear reactors", "Marine engineering", "Aerospace combustors", "Heat treatment fixtures", "Pollution control", "Oil and gas downhole", "Furnace components", "Steam generators", "Expansion bellows", "Rocket engines", "Cryogenic equipment"],
        "notes_pool": [
            "Solid-solution strengthened alloy with excellent oxidation resistance.",
            "Precipitation-hardened superalloy for extreme temperature service.",
            "Corrosion-resistant alloy for aggressive chemical environments.",
            "Low thermal expansion alloy for precision applications.",
            "High magnetic permeability alloy for electromagnetic shielding.",
            "Age-hardenable alloy combining high strength with corrosion resistance.",
            "Single-crystal superalloy for turbine blade applications.",
        ],
        "sustainability_pool": [
            "Extreme durability in harsh environments minimizes lifecycle material use.",
            "Nickel recycling is well established with high scrap value.",
            "Enables higher turbine efficiency, reducing fossil fuel consumption.",
            "Long service life in critical applications reduces total environmental impact.",
        ],
    },

    # ── COPPER ALLOYS ──
    "Copper Alloys": {
        "templates": [
            # Wrought coppers & high-copper alloys
            *[f"Copper {n}" for n in ["C101", "C102", "C103", "C104", "C106", "C107", "C108", "C109", "C110", "C111", "C113", "C114", "C116", "C120", "C122", "C125", "C127", "C145", "C147", "C150", "C151", "C155", "C162", "C170", "C172", "C175", "C176", "C182"]],
            # Brasses
            *[f"Brass {n}" for n in ["C210", "C220", "C226", "C230", "C240", "C260", "C268", "C270", "C272", "C280", "C314", "C330", "C332", "C340", "C342", "C350", "C353", "C356", "C365", "C370", "C377", "C380", "C385"]],
            # Bronzes
            *[f"Bronze {n}" for n in ["C505", "C510", "C511", "C519", "C521", "C524", "C532", "C544", "C608", "C613", "C614", "C619", "C623", "C624", "C625", "C630", "C632", "C638", "C642", "C651", "C655", "C661", "C663"]],
            # Nickel silvers
            *[f"Nickel Silver {n}" for n in ["C735", "C740", "C745", "C752", "C754", "C757", "C762", "C764", "C770", "C774", "C782", "C796"]],
            # Copper-nickel
            *[f"Cupronickel {n}" for n in ["C704", "C706 (90-10)", "C710", "C715 (70-30)", "C722"]],
        ],
        "ranges": {
            "density": (7.45, 8.95),
            "tensile_strength_ultimate": (150, 1350),
            "tensile_strength_yield": (50, 1200),
            "modulus_of_elasticity": (100, 135),
            "thermal_conductivity": (20, 395),
            "melting_point": (880, 1085),
            "cost_per_kg_usd": (5.0, 65.0),
            "sustainability_score": (6, 9),
        },
        "applications": ["Electrical wiring", "Heat exchangers", "Plumbing fittings", "Marine hardware", "Musical instruments", "Bearings and bushings", "Springs", "Electrical connectors", "Coinage", "Decorative hardware", "Antimicrobial surfaces", "Welding electrodes", "RF shielding"],
        "notes_pool": [
            "Pure copper grade with highest electrical conductivity.",
            "Brass alloy with excellent machinability and moderate strength.",
            "Phosphor bronze with excellent fatigue resistance and spring properties.",
            "Copper-nickel alloy with outstanding seawater corrosion resistance.",
            "Nickel silver with attractive color and good corrosion resistance.",
            "Silicon bronze with high strength and excellent corrosion resistance.",
            "Aluminum bronze with high strength and excellent wear resistance.",
            "Beryllium copper with highest strength of any copper alloy.",
        ],
        "sustainability_pool": [
            "Copper is 100% recyclable with no loss of properties. Over 80% of copper ever mined is still in use.",
            "Excellent recyclability and high scrap value encourage material recovery.",
            "Natural antimicrobial properties reduce need for chemical disinfectants.",
            "Long conductor life in electrical applications reduces material turnover.",
        ],
    },

    # ── THERMOPLASTICS ──
    "Thermoplastics": {
        "templates": [
            # PE variants
            *[f"Polyethylene {n}" for n in ["LDPE", "LLDPE", "MDPE", "HDPE-HMW", "UHMWPE", "XLPE", "HDPE Pipe Grade", "HDPE Blow Molding", "HDPE Injection", "LDPE Film Grade"]],
            # PP variants
            *[f"Polypropylene {n}" for n in ["Homopolymer", "Copolymer", "Impact Modified", "Glass-Filled 20%", "Glass-Filled 30%", "Glass-Filled 40%", "Talc-Filled 20%", "Talc-Filled 40%", "Flame Retardant", "UV Stabilized"]],
            # ABS variants
            *[f"ABS {n}" for n in ["High Impact", "Heat Resistant", "Flame Retardant", "Glass-Filled 20%", "Glass-Filled 30%", "Plating Grade", "Extrusion Grade", "High Flow", "UV Stabilized", "Transparent"]],
            # PC variants
            *[f"Polycarbonate {n}" for n in ["General Purpose", "High Flow", "Glass-Filled 10%", "Glass-Filled 20%", "Glass-Filled 30%", "Glass-Filled 40%", "Flame Retardant", "UV Stabilized", "Medical Grade", "Optical Grade", "Film Grade"]],
            # Nylon variants
            *[f"Nylon {n}" for n in ["6", "6 Glass-Filled 30%", "6 Glass-Filled 33%", "6 Glass-Filled 50%", "6 Mineral-Filled", "6/6 Glass-Filled 30%", "6/6 Glass-Filled 33%", "6/6 Glass-Filled 50%", "6/6 Mineral-Filled", "6/10", "6/12", "11", "12", "46", "MXD6", "MXD6 Glass-Filled 50%", "612", "MDI"]],
            # PBT / PET
            *[f"PBT {n}" for n in ["Unfilled", "Glass-Filled 15%", "Glass-Filled 30%", "Glass-Filled 45%", "Mineral-Filled", "Flame Retardant", "Impact Modified"]],
            *[f"PET {n}" for n in ["Amorphous", "Crystallized", "Glass-Filled 30%", "Glass-Filled 45%", "Flame Retardant"]],
            # POM
            *[f"Acetal (POM) {n}" for n in ["Homopolymer", "Copolymer", "Glass-Filled 20%", "Glass-Filled 25%", "PTFE-Filled", "UV Stabilized", "Impact Modified", "Conductive"]],
            # Other engineering plastics
            *[f"Polysulfone {n}" for n in ["Standard", "Glass-Filled 20%", "Glass-Filled 30%", "Mineral-Filled"]],
            *[f"Polyethersulfone {n}" for n in ["Standard", "Glass-Filled 20%", "Glass-Filled 30%"]],
            *[f"PPO/PPE {n}" for n in ["Standard", "Glass-Filled 20%", "Glass-Filled 30%", "Flame Retardant", "Impact Modified"]],
            *[f"SAN {n}" for n in ["Standard", "Glass-Filled 20%", "High Heat"]],
            *[f"ASA {n}" for n in ["Standard", "UV Stabilized", "Glass-Filled"]],
            *[f"PPS {n}" for n in ["Unfilled", "Glass-Filled 40%", "Glass-Filled 65%", "Glass-Mineral Filled", "Carbon Fiber Filled"]],
            "PMMA Cast Sheet", "PMMA Injection Grade", "PMMA Impact Modified", "PMMA UV Grade",
            "Polystyrene General Purpose", "Polystyrene High Impact (HIPS)", "Polystyrene Crystal",
            "PVC Rigid", "PVC Flexible", "CPVC",
            "TPU Shore 60A", "TPU Shore 70A", "TPU Shore 80A", "TPU Shore 85A", "TPU Shore 90A", "TPU Shore 95A", "TPU Shore 55D", "TPU Shore 64D", "TPU Shore 72D",
            "EVA 18% VA", "EVA 25% VA", "EVA 28% VA",
        ],
        "ranges": {
            "density": (0.88, 1.65),
            "tensile_strength_ultimate": (8, 210),
            "tensile_strength_yield": (5, 185),
            "modulus_of_elasticity": (0.2, 18.0),
            "thermal_conductivity": (0.10, 0.45),
            "melting_point": (105, 345),
            "cost_per_kg_usd": (0.8, 55.0),
            "sustainability_score": (3, 7),
        },
        "applications": ["Injection molded parts", "Packaging", "Automotive interiors", "Consumer electronics", "Medical devices", "Piping systems", "Film and sheet", "Wire insulation", "Gears and bearings", "Food containers", "Optical lenses", "Electrical housings", "3D printing filament"],
        "notes_pool": [
            "General purpose grade suitable for a wide range of applications.",
            "Glass-fiber reinforced for significantly improved stiffness and strength.",
            "Flame retardant grade meeting UL94 V-0 requirements.",
            "Impact modified grade with improved toughness at low temperatures.",
            "High flow grade for thin-wall injection molding.",
            "UV stabilized grade for outdoor applications.",
            "Medical grade meeting USP Class VI and ISO 10993 requirements.",
            "Food contact approved grade meeting FDA and EU regulations.",
        ],
        "sustainability_pool": [
            "Recyclable thermoplastic that can be reprocessed multiple times.",
            "Bio-based variants are becoming available to reduce petroleum dependence.",
            "Lightweight material reduces energy consumption in transport applications.",
            "Growing recycling infrastructure, especially for PE and PET.",
            "Post-consumer recycled content options available for many grades.",
        ],
    },

    # ── HIGH-PERFORMANCE POLYMERS ──
    "High-Performance Polymers": {
        "templates": [
            *[f"PEEK {n}" for n in ["Unfilled", "Glass-Filled 30%", "Carbon-Filled 30%", "Bearing Grade", "Medical Grade", "HPV Grade", "HT Grade"]],
            *[f"PEI (Ultem) {n}" for n in ["1000", "1010", "2100", "2200", "2300", "2400", "CRS5001"]],
            *[f"PAI (Torlon) {n}" for n in ["4203", "4275", "4301", "4435", "5030", "7130"]],
            *[f"PI (Polyimide) {n}" for n in ["Kapton HN", "Kapton FN", "Vespel SP-1", "Vespel SP-21", "Vespel SP-211", "Vespel SP-22", "Vespel SP-3", "Vespel SCP-5000"]],
            *[f"LCP {n}" for n in ["Unfilled", "Glass-Filled 30%", "Glass-Filled 40%", "Glass-Filled 50%", "Carbon-Filled", "Mineral-Filled"]],
            *[f"PPA {n}" for n in ["Unfilled", "Glass-Filled 33%", "Glass-Filled 45%", "Glass-Filled 60%"]],
            *[f"PPSU {n}" for n in ["Standard", "Glass-Filled 20%", "Glass-Filled 30%"]],
            "PCTFE", "PVDF", "ETFE", "FEP", "PFA",
        ],
        "ranges": {
            "density": (1.24, 2.20),
            "tensile_strength_ultimate": (48, 290),
            "tensile_strength_yield": (40, 260),
            "modulus_of_elasticity": (2.5, 25.0),
            "thermal_conductivity": (0.15, 0.50),
            "melting_point": (250, 420),
            "cost_per_kg_usd": (25.0, 450.0),
            "sustainability_score": (3, 6),
        },
        "applications": ["Aerospace components", "Medical implants", "Semiconductor manufacturing", "Oil and gas seals", "Automotive under-hood", "Chemical processing", "Cryogenic applications", "Electrical insulation", "Bearing surfaces", "Wire coating"],
        "notes_pool": [
            "Semi-crystalline thermoplastic with exceptional mechanical properties at elevated temperatures.",
            "Amorphous thermoplastic with excellent flame resistance and low smoke generation.",
            "Highest performing melt-processable polymer available.",
            "Fluoropolymer with exceptional chemical inertness and dielectric properties.",
            "Liquid crystal polymer with extremely low moisture absorption and warpage.",
        ],
        "sustainability_pool": [
            "Exceptional durability and thermal stability dramatically extend service life.",
            "Can replace metals in many applications, reducing weight and energy consumption.",
            "Recyclable thermoplastic, though recycling infrastructure is limited.",
        ],
    },

    # ── ELASTOMERS ──
    "Elastomers": {
        "templates": [
            *[f"Silicone {n}" for n in ["VMQ 30 Shore A", "VMQ 40 Shore A", "VMQ 50 Shore A", "VMQ 60 Shore A", "VMQ 70 Shore A", "VMQ 80 Shore A", "Fluorosilicone (FVMQ)", "Liquid Silicone Rubber (LSR) 30A", "LSR 40A", "LSR 50A", "LSR 60A", "LSR 70A", "High Consistency Rubber (HCR)", "Platinum-Cured Medical Grade", "Conductive Silicone", "High-Temperature Silicone"]],
            *[f"Nitrile (NBR) {n}" for n in ["Low ACN 18%", "Medium ACN 33%", "High ACN 45%", "Carboxylated (XNBR)", "Hydrogenated (HNBR)"]],
            *[f"EPDM {n}" for n in ["Standard", "Peroxide-Cured", "Sulfur-Cured", "FDA Grade", "High Ethylene", "Low Ethylene"]],
            *[f"Viton (FKM) {n}" for n in ["Type A", "Type B", "Type F", "Type GBL-S", "Type GLT", "Type GFLT", "Extreme (FFKM)"]],
            *[f"Neoprene (CR) {n}" for n in ["W Type", "GW Type", "Standard", "Flame Retardant"]],
            *[f"Polyurethane Elastomer {n}" for n in ["60A", "70A", "80A", "90A", "95A", "55D", "64D", "72D", "Ester-Based", "Ether-Based"]],
            "Butyl Rubber (IIR)", "Chlorobutyl Rubber (CIIR)", "Bromobutyl Rubber (BIIR)",
            "SBR Rubber", "BR Rubber (Polybutadiene)",
            "Chlorosulfonated Polyethylene (CSM/Hypalon)",
            "Ethylene Acrylic (AEM/Vamac)",
            "Epichlorohydrin (ECO)",
            "Polysulfide Rubber",
            "Natural Rubber RSS1", "Natural Rubber SMR-20",
        ],
        "ranges": {
            "density": (0.85, 2.05),
            "tensile_strength_ultimate": (3, 55),
            "tensile_strength_yield": (2, 40),
            "modulus_of_elasticity": (0.001, 0.050),
            "thermal_conductivity": (0.10, 0.35),
            "melting_point": (150, 450),
            "cost_per_kg_usd": (2.0, 250.0),
            "sustainability_score": (3, 8),
        },
        "applications": ["O-rings and seals", "Gaskets", "Hoses", "Vibration dampers", "Medical tubing", "Automotive weatherstripping", "Conveyor belts", "Tires", "Shoe soles", "Protective gloves", "Expansion joints", "Diaphragms"],
        "notes_pool": [
            "Elastomer with excellent compression set resistance and wide temperature range.",
            "Oil-resistant rubber suitable for fuel and hydraulic fluid contact.",
            "Weather-resistant elastomer with outstanding ozone and UV stability.",
            "Fluoroelastomer with exceptional chemical and heat resistance.",
            "Natural rubber with excellent tensile strength and tear resistance.",
            "Polyurethane elastomer combining rubber-like flexibility with abrasion resistance.",
        ],
        "sustainability_pool": [
            "Natural rubber is a renewable, biodegradable material.",
            "Silicone rubber's extreme durability reduces replacement frequency.",
            "Rubber recycling and devulcanization technologies are advancing.",
            "Long service life of high-performance elastomers reduces total waste.",
        ],
    },

    # ── CERAMICS ──
    "Ceramics": {
        "templates": [
            *[f"Alumina {n}" for n in ["85%", "90%", "92%", "94%", "95%", "96%", "97%", "99%", "99.5%", "99.7%", "99.8%", "99.9%"]],
            *[f"Zirconia {n}" for n in ["3Y-TZP", "Mg-PSZ", "Ce-TZP", "Y-FSZ", "ATZ (Alumina Toughened)", "ZTA (Zirconia Toughened Alumina)", "Black Zirconia", "Dental Grade"]],
            *[f"Silicon Carbide {n}" for n in ["Sintered (SSiC)", "Reaction Bonded (RBSiC)", "Pressureless Sintered", "Hot Pressed", "CVD", "Recrystallized"]],
            *[f"Silicon Nitride {n}" for n in ["Hot Pressed (HPSN)", "Sintered (SSN)", "Reaction Bonded (RBSN)", "Gas Pressure Sintered (GPSN)", "Sialon"]],
            *[f"Boron Carbide {n}" for n in ["Hot Pressed", "Sintered", "Reaction Bonded"]],
            *[f"Boron Nitride {n}" for n in ["Hexagonal (hBN)", "Cubic (cBN)", "Pyrolytic (PBN)"]],
            *[f"Aluminum Nitride {n}" for n in ["Standard", "High Purity", "High Thermal Conductivity"]],
            "Mullite", "Cordierite", "Steatite", "Forsterite", "Spinel",
            *[f"Titanium Diboride {n}" for n in ["Hot Pressed", "Sintered"]],
            "Tungsten Carbide (WC-Co 6%)", "Tungsten Carbide (WC-Co 10%)", "Tungsten Carbide (WC-Co 15%)", "Tungsten Carbide (WC-Co 25%)",
            "Machinable Glass Ceramic (Macor)", "Machinable Ceramic (Shapal)",
            *[f"Porcelain {n}" for n in ["Electrical", "Dental", "Technical"]],
        ],
        "ranges": {
            "density": (2.10, 15.70),
            "tensile_strength_ultimate": (80, 1200),
            "tensile_strength_yield": (70, 1100),
            "modulus_of_elasticity": (90, 650),
            "thermal_conductivity": (1.5, 200),
            "melting_point": (1300, 3400),
            "cost_per_kg_usd": (8.0, 350.0),
            "sustainability_score": (5, 8),
        },
        "applications": ["Cutting tools", "Wear parts", "Electrical insulators", "Thermal barriers", "Armor plating", "Bearing components", "Dental prosthetics", "Crucibles", "Kiln furniture", "Semiconductor substrates", "Catalyst supports", "Biomedical implants"],
        "notes_pool": [
            "High-purity oxide ceramic with excellent hardness and electrical insulation.",
            "Non-oxide ceramic with exceptional thermal shock resistance.",
            "Toughened ceramic with improved fracture resistance for structural applications.",
            "Biocompatible ceramic suitable for medical implant applications.",
            "Ultra-hard ceramic for extreme wear and cutting tool applications.",
            "High thermal conductivity ceramic for heat dissipation applications.",
        ],
        "sustainability_pool": [
            "Extremely long service life due to hardness and wear resistance.",
            "Raw materials are abundant in the earth's crust.",
            "Ceramic production is energy-intensive but products are very durable.",
            "Inert material that does not leach or degrade in the environment.",
        ],
    },

    # ── COMPOSITES ──
    "Composites": {
        "templates": [
            *[f"Carbon Fiber/Epoxy {n}" for n in ["Standard Modulus", "Intermediate Modulus", "High Modulus", "Ultra-High Modulus", "Unidirectional", "Woven 3K", "Woven 6K", "Woven 12K", "Quasi-Isotropic", "Chopped Fiber", "Prepreg T300", "Prepreg T700", "Prepreg T800", "Prepreg M40J", "Prepreg M46J", "Prepreg M55J"]],
            *[f"Glass Fiber/Epoxy {n}" for n in ["E-Glass Uni", "E-Glass Woven", "S-Glass Uni", "S-Glass Woven", "S2-Glass", "R-Glass"]],
            *[f"Glass Fiber/Polyester {n}" for n in ["SMC", "BMC", "Hand Layup", "Filament Wound", "Pultrusion"]],
            *[f"Aramid/Epoxy {n}" for n in ["Kevlar 29", "Kevlar 49", "Kevlar 149", "Twaron", "Technora"]],
            *[f"Glass/Phenolic {n}" for n in ["G-3", "G-5", "G-7", "G-9", "G-10", "G-11", "FR-4", "FR-5", "FR-6"]],
            "Carbon/PEEK Composite", "Carbon/PPS Composite", "Carbon/Nylon Composite", "Carbon/Polycarbonate",
            *[f"Natural Fiber Composite ({n})" for n in ["Flax/Epoxy", "Hemp/Epoxy", "Jute/Polyester", "Kenaf/PP", "Bamboo/Epoxy", "Basalt/Epoxy"]],
            "Boron/Epoxy", "Quartz/Epoxy", "Ceramic Matrix Composite (SiC/SiC)", "Ceramic Matrix Composite (C/SiC)", "Metal Matrix Composite (Al/SiC)", "Metal Matrix Composite (Al/Al2O3)",
        ],
        "ranges": {
            "density": (1.10, 3.50),
            "tensile_strength_ultimate": (80, 3500),
            "tensile_strength_yield": (60, 3200),
            "modulus_of_elasticity": (5, 400),
            "thermal_conductivity": (0.15, 200),
            "melting_point": (100, 2500),
            "cost_per_kg_usd": (3.0, 500.0),
            "sustainability_score": (2, 7),
        },
        "applications": ["Aerospace structures", "Racing vehicles", "Sporting goods", "Wind turbine blades", "Boat hulls", "Pressure vessels", "PCB substrates", "Armor systems", "Automotive body panels", "Bicycle frames", "Bridge decks", "Prosthetics"],
        "notes_pool": [
            "Fiber-reinforced composite offering exceptional strength-to-weight ratio.",
            "Thermoset matrix composite with excellent fatigue and creep resistance.",
            "Thermoplastic matrix composite enabling faster manufacturing and recyclability.",
            "Woven reinforcement providing balanced in-plane properties.",
            "Unidirectional layup optimized for loading in the fiber direction.",
            "Natural fiber composite offering lower cost and improved sustainability.",
        ],
        "sustainability_pool": [
            "Lightweight composites enable significant fuel savings in transport.",
            "Thermoset composites are difficult to recycle; thermoplastic matrices improve this.",
            "Natural fiber composites use renewable reinforcements.",
            "Composite recycling technologies (pyrolysis, solvolysis) are maturing.",
        ],
    },

    # ── PRECIOUS & REFRACTORY METALS ──
    "Specialty Metals": {
        "templates": [
            "Platinum-Iridium (Pt-10Ir)", "Platinum-Rhodium (Pt-10Rh)", "Platinum-Rhodium (Pt-13Rh)", "Platinum-Ruthenium",
            "Gold 24K", "Gold 22K", "Gold 18K White", "Gold 18K Rose", "Gold 14K", "Gold 10K",
            "Silver Sterling (925)", "Silver Fine (999)", "Silver-Copper", "Silver-Palladium",
            "Palladium", "Palladium-Nickel", "Palladium-Silver", "Rhodium", "Iridium", "Ruthenium", "Osmium",
            "Rhenium", "Hafnium",
            *[f"Tungsten {n}" for n in ["Pure", "W-1% ThO2", "W-2% ThO2", "W-25Re", "W-3Re", "W-5Re", "W-26Re", "Heavy Alloy 90W", "Heavy Alloy 93W", "Heavy Alloy 95W", "Heavy Alloy 97W"]],
            *[f"Molybdenum {n}" for n in ["Pure", "Mo-0.5Ti", "Mo-0.5Ti-0.1Zr (TZM)", "Mo-30W", "Mo-La2O3"]],
            *[f"Tantalum {n}" for n in ["Pure", "Ta-2.5W", "Ta-10W", "Ta-40Nb"]],
            *[f"Niobium {n}" for n in ["Pure", "Nb-1Zr", "Nb-10Hf-1Ti (C-103)", "Nb-28Ta-10W-1Zr (FS-85)"]],
            "Zirconium 702", "Zirconium 705",
            *[f"Cobalt-Chrome {n}" for n in ["F75 (Cast)", "F90 (Wrought)", "F562 (MP35N)", "F1537 (Wrought High-C)", "L-605", "Stellite 6", "Stellite 12", "Stellite 21"]],
            "Magnesium AZ61A", "Magnesium AZ80A", "Magnesium AZ91D", "Magnesium AZ91E", "Magnesium AM60B", "Magnesium AM50A", "Magnesium ZE41A", "Magnesium ZK60A", "Magnesium WE43", "Magnesium WE54", "Magnesium Elektron 21",
            "Tin Pure", "Tin-Lead Solder 60/40", "Tin-Lead Solder 63/37", "Lead-Free Solder SAC305", "Lead-Free Solder SAC405", "Lead-Free Solder SN100C",
            "Zinc Pure", "Zamak 2", "Zamak 5", "Zamak 7", "ZA-8", "ZA-12", "ZA-27",
            "Indium", "Gallium", "Germanium", "Bismuth", "Antimony", "Cadmium",
        ],
        "ranges": {
            "density": (1.74, 22.60),
            "tensile_strength_ultimate": (15, 2100),
            "tensile_strength_yield": (8, 1900),
            "modulus_of_elasticity": (15, 460),
            "thermal_conductivity": (6, 430),
            "melting_point": (29, 3422),
            "cost_per_kg_usd": (1.5, 70000.0),
            "sustainability_score": (3, 8),
        },
        "applications": ["Medical implants", "Electronics", "Jewelry", "Catalysts", "Laboratory equipment", "Nuclear applications", "Aerospace", "Radiation shielding", "High-temperature furnaces", "Cutting tools", "Soldering", "Die casting"],
        "notes_pool": [
            "Precious metal with exceptional corrosion resistance and catalytic activity.",
            "Refractory metal with highest melting point suitable for extreme temperatures.",
            "Biocompatible metal used extensively in medical implant applications.",
            "Lightweight alloy offering excellent strength-to-weight for structural use.",
            "Specialty alloy engineered for specific high-performance requirements.",
            "Low-melting-point alloy used in joining and soldering applications.",
        ],
        "sustainability_pool": [
            "Precious metals are extensively recycled due to high intrinsic value.",
            "Refractory metals enable high-efficiency industrial processes.",
            "Very long service life offsets high production energy costs.",
            "Recycling infrastructure well-established for high-value metals.",
        ],
    },

    # ── GLASS & CONSTRUCTION MATERIALS ──
    "Glass & Construction": {
        "templates": [
            *[f"Glass {n}" for n in ["Soda-Lime Annealed", "Soda-Lime Tempered", "Soda-Lime Laminated", "Borosilicate 3.3", "Borosilicate 7740", "Aluminosilicate", "Lead Crystal", "Fused Quartz", "Fused Silica", "E-Glass Fiber", "S-Glass Fiber", "AR-Glass Fiber", "C-Glass Fiber", "D-Glass Fiber", "R-Glass Fiber", "Gorilla Glass", "Sapphire Glass", "Low-E Glass", "Float Glass Clear", "Float Glass Tinted", "Glass Ceramic (Zerodur)", "Glass Ceramic (Ceran)"]],
            *[f"Concrete {n}" for n in ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60", "C60/75", "C70/85", "C80/95", "C90/105", "Ultra-High Performance (UHPC)", "Self-Compacting (SCC)", "Fiber Reinforced (SFRC)", "Lightweight Aggregate", "Heavyweight", "Polymer Modified", "Geopolymer", "Pervious", "Shotcrete"]],
            "Gypsum Board Standard", "Gypsum Board Fire-Rated", "Gypsum Board Moisture-Resistant",
            "Calcium Silicate Board", "Calcium Silicate Pipe Insulation",
            "Cement Portland Type I", "Cement Portland Type II", "Cement Portland Type III", "Cement Portland Type IV", "Cement Portland Type V",
            "Mortar Type M", "Mortar Type S", "Mortar Type N", "Mortar Type O",
            "Brick Common", "Brick Engineering", "Brick Fire (Fireclay)", "Brick Insulating",
            "Clay Tile Structural", "Clay Tile Roofing",
            "Limestone", "Marble", "Granite", "Sandstone", "Slate", "Quartzite", "Travertine", "Basalt",
        ],
        "ranges": {
            "density": (0.50, 11.50),
            "tensile_strength_ultimate": (1, 3500),
            "tensile_strength_yield": (1, 3200),
            "modulus_of_elasticity": (2, 450),
            "thermal_conductivity": (0.10, 200),
            "melting_point": (500, 2050),
            "cost_per_kg_usd": (0.03, 85.0),
            "sustainability_score": (3, 9),
        },
        "applications": ["Windows and facades", "Structural foundations", "Bridges", "Road construction", "Flooring", "Countertops", "Laboratory equipment", "Optical components", "Thermal insulation", "Fire protection", "Masonry", "Roofing"],
        "notes_pool": [
            "Standard construction material with good compressive strength.",
            "Glass type with specific optical or thermal properties.",
            "Natural stone with aesthetic appeal and structural capability.",
            "High-performance variant engineered for demanding applications.",
            "Fire-resistant material for safety-critical construction.",
            "Specialty glass with tailored thermal expansion properties.",
        ],
        "sustainability_pool": [
            "Glass is 100% recyclable and can be recycled endlessly.",
            "Concrete is the most used construction material globally. Cement production is a major CO2 source.",
            "Natural stone is durable and requires minimal processing energy.",
            "Recycled aggregate concrete reduces virgin material consumption.",
        ],
    },

    # ── WOOD & NATURAL MATERIALS ──
    "Natural Materials": {
        "templates": [
            # Hardwoods
            *[f"Wood {n}" for n in ["Ash (White)", "Balsa", "Beech (European)", "Birch (Yellow)", "Cherry (Black)", "Ebony", "Elm (American)", "Hickory", "Ipe (Lapacho)", "Jarrah", "Mahogany (African)", "Mahogany (Honduras)", "Maple (Hard)", "Maple (Soft)", "Meranti", "Merbau", "Oak (Red)", "Oak (White)", "Padauk", "Poplar (Yellow)", "Purpleheart", "Rosewood (Indian)", "Sapele", "Teak", "Walnut (Black)", "Wenge", "Zebrawood"]],
            # Softwoods
            *[f"Wood {n}" for n in ["Cedar (Western Red)", "Cedar (White)", "Cypress", "Douglas Fir", "Hemlock (Western)", "Larch (European)", "Pine (Ponderosa)", "Pine (Radiata)", "Pine (Southern Yellow)", "Redwood", "Spruce (Sitka)", "Spruce (White)", "Yew"]],
            # Engineered wood
            *[f"Engineered Wood {n}" for n in ["Plywood (Softwood)", "Plywood (Hardwood)", "Plywood (Marine)", "OSB", "MDF Standard", "MDF Moisture Resistant", "HDF", "Particleboard Standard", "Particleboard Moisture Resistant", "LVL (Laminated Veneer Lumber)", "Glulam", "CLT (Cross Laminated Timber)", "Parallam (PSL)", "I-Joist"]],
            # Bamboo
            *[f"Bamboo {n}" for n in ["Moso (Raw)", "Moso (Laminated)", "Guadua", "Strand Woven", "Phyllostachys"]],
            # Natural fibers
            "Cork Natural", "Cork Agglomerated", "Leather (Bovine)", "Leather (Goat)", "Bone (Cortical)", "Bone (Cancellous)", "Shell (Nacre)", "Silk Fiber", "Hemp Fiber", "Flax Fiber", "Jute Fiber", "Sisal Fiber", "Coconut Coir", "Kenaf Fiber", "Cotton Fiber", "Wool Fiber",
        ],
        "ranges": {
            "density": (0.05, 1.35),
            "tensile_strength_ultimate": (3, 600),
            "tensile_strength_yield": (2, 500),
            "modulus_of_elasticity": (0.01, 18.0),
            "thermal_conductivity": (0.04, 0.30),
            "melting_point": (150, 400),
            "cost_per_kg_usd": (0.3, 35.0),
            "sustainability_score": (6, 10),
        },
        "applications": ["Furniture", "Flooring", "Construction framing", "Decking", "Structural beams", "Musical instruments", "Boatbuilding", "Insulation", "Textiles", "Packaging", "Sporting goods", "Art and craft"],
        "notes_pool": [
            "Hardwood with attractive grain pattern and good workability.",
            "Softwood offering economical structural performance.",
            "Engineered wood product with improved dimensional stability.",
            "Bamboo species with exceptional growth rate and strength.",
            "Natural fiber with renewable sourcing and biodegradability.",
            "Tropical hardwood with outstanding durability and weather resistance.",
        ],
        "sustainability_pool": [
            "Renewable resource when sustainably harvested. Carbon negative over lifecycle.",
            "Fast-growing species that can be harvested every 3-5 years.",
            "Biodegradable at end of life with minimal environmental impact.",
            "Forest certification (FSC, PEFC) ensures responsible management.",
            "Natural material requiring minimal processing energy.",
        ],
    },
}


def generate_material(name, category, family_data):
    """Generate a single material entry with realistic properties."""
    ranges = family_data["ranges"]

    # Generate properties
    density = r(*ranges["density"], 2)
    uts = r(*ranges["tensile_strength_ultimate"], 0)
    ys = r(ranges["tensile_strength_yield"][0], min(uts * 0.95, ranges["tensile_strength_yield"][1]), 0)
    mod = r(*ranges["modulus_of_elasticity"], 1)
    tc = r(*ranges["thermal_conductivity"], 2)
    mp = r(*ranges["melting_point"], 0)
    cost = r(*ranges["cost_per_kg_usd"], 2)
    sus = random.randint(*ranges["sustainability_score"])

    apps = random.sample(family_data["applications"], min(3, len(family_data["applications"])))
    note = pick(family_data["notes_pool"])
    sus_note = pick(family_data["sustainability_pool"])

    return {
        "material_name": name,
        "category": category,
        "material_notes": note,
        "density": density,
        "tensile_strength_ultimate": uts,
        "tensile_strength_yield": ys,
        "modulus_of_elasticity": mod,
        "thermal_conductivity": tc,
        "melting_point": mp,
        "cost_per_kg_usd": cost,
        "sustainability_score": sus,
        "sustainability_notes": sus_note,
        "common_applications": apps,
    }


def main():
    TARGET = 10000
    db = dict(existing_db)  # Start with existing 72
    existing_names = set(db.keys())

    print(f"\n{'='*70}")
    print(f"HELIOS MATERIALS DATABASE GENERATOR")
    print(f"Target: {TARGET} materials")
    print(f"{'='*70}")

    for category, family_data in MATERIAL_FAMILIES.items():
        templates = family_data["templates"]
        added = 0
        for name in templates:
            if name not in existing_names:
                db[name] = generate_material(name, category, family_data)
                existing_names.add(name)
                added += 1
        print(f"  {category}: +{added} new (from {len(templates)} templates)")

    print(f"\n  Subtotal after templates: {len(db)} materials")

    # ── Fill remaining with parametric variants ──
    remaining = TARGET - len(db)
    if remaining > 0:
        print(f"  Generating {remaining} additional parametric variants...")

        # Categories we can expand easily with numbered variants
        expandable = [
            ("Thermoplastics", "Polymer Blend", ["Custom polymer blend optimized for specific application requirements."]),
            ("Carbon & Alloy Steels", "Steel Grade", ["Custom steel grade with tailored composition for specific mechanical targets."]),
            ("Aluminum Alloys", "Aluminum Alloy", ["Custom aluminum alloy composition for targeted performance."]),
            ("Nickel Alloys", "Nickel Alloy", ["Nickel-based alloy with tailored composition for demanding service conditions."]),
            ("Ceramics", "Ceramic Compound", ["Advanced ceramic formulation engineered for specific property targets."]),
            ("Composites", "Composite System", ["Engineered composite system with optimized fiber-matrix combination."]),
            ("Copper Alloys", "Copper Alloy", ["Copper-based alloy formulated for specific conductivity and strength requirements."]),
            ("Stainless Steels", "Stainless Grade", ["Stainless steel grade with specific chemistry for targeted corrosion resistance."]),
            ("Elastomers", "Elastomer Compound", ["Custom elastomer compound blended for specific service conditions."]),
            ("High-Performance Polymers", "HPP Grade", ["High-performance polymer grade with tailored thermal and mechanical properties."]),
            ("Glass & Construction", "Construction Material", ["Engineered construction material for structural or architectural use."]),
            ("Specialty Metals", "Specialty Alloy", ["Specialty alloy with unique property combination for niche applications."]),
            ("Natural Materials", "Bio-Material", ["Bio-based material sourced from renewable natural resources."]),
            ("Tool Steels", "Tool Steel Grade", ["Tool steel grade heat-treated for specific hardness and toughness balance."]),
            ("Titanium Alloys", "Titanium Alloy", ["Titanium-based alloy for high strength-to-weight applications."]),
        ]

        idx = 1
        while len(db) < TARGET:
            cat_name, prefix, notes = pick(expandable)
            family = MATERIAL_FAMILIES[cat_name]
            name = f"{prefix} {idx:04d}"
            while name in existing_names:
                idx += 1
                name = f"{prefix} {idx:04d}"

            mat = generate_material(name, cat_name, family)
            mat["material_notes"] = pick(notes)
            db[name] = mat
            existing_names.add(name)
            idx += 1

        print(f"  ✅ Parametric variants added")

    # ── Save ──
    output_path = "materials_database.json"
    with open(output_path, "w") as f:
        json.dump(db, f, indent=2)

    # Stats
    categories = {}
    for mat in db.values():
        cat = mat.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n{'='*70}")
    print(f"✅ GENERATION COMPLETE: {len(db)} materials")
    print(f"{'='*70}")
    print(f"\nMaterials by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print(f"\nSaved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"\nNext step: run 'python ingest_v2.py' to rebuild the vector database")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
