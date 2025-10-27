"""
Helios Evaluation Framework
Tests retrieval accuracy and answer quality across different query types.
"""

import json
import time
from typing import List, Dict
import requests

# Test cases organized by query type
TEST_CASES = [
    # Category 1: Property-based queries (should use filters)
    {
        "query": "materials with yield strength over 800 MPa",
        "expected_materials": ["Inconel 718", "Titanium Ti-6Al-4V", "Stainless Steel 17-4 PH", "Tool Steel D2"],
        "min_expected": 2,
        "category": "property_filter"
    },
    {
        "query": "materials with density less than 2 g/cc",
        "expected_materials": ["Magnesium AZ31B", "PEEK", "Nylon 6/6", "Polypropylene"],
        "min_expected": 3,
        "category": "property_filter"
    },
    {
        "query": "materials under $5 per kg",
        "expected_materials": ["Carbon Steel 1045", "Aluminum 6061-T6", "Polypropylene"],
        "min_expected": 3,
        "category": "cost_filter"
    },
    
    # Category 2: Semantic/fuzzy queries
    {
        "query": "lightweight corrosion resistant metal",
        "expected_materials": ["Aluminum 6061-T6", "Titanium Ti-6Al-4V", "Magnesium AZ31B"],
        "min_expected": 2,
        "category": "semantic"
    },
    {
        "query": "high temperature resistant materials",
        "expected_materials": ["Inconel 718", "PEEK", "Tungsten", "Ceramic (Alumina Al2O3)"],
        "min_expected": 2,
        "category": "semantic"
    },
    {
        "query": "materials for aerospace applications",
        "expected_materials": ["Titanium Ti-6Al-4V", "Aluminum 7075-T6", "Carbon Fiber Composite"],
        "min_expected": 2,
        "category": "application"
    },
    
    # Category 3: Exact matches (keyword search should excel)
    {
        "query": "density 2.70 g/cc",
        "expected_materials": ["Aluminum 6061-T6"],
        "min_expected": 1,
        "category": "exact_match"
    },
    {
        "query": "Aluminum 6061-T6 properties",
        "expected_materials": ["Aluminum 6061-T6"],
        "min_expected": 1,
        "category": "exact_match"
    },
    
    # Category 4: Comparison queries
    {
        "query": "compare stainless steel 304 and 316",
        "expected_materials": ["Stainless Steel 304", "Stainless Steel 316"],
        "min_expected": 2,
        "category": "comparison"
    },
    {
        "query": "titanium vs aluminum for aerospace",
        "expected_materials": ["Titanium Ti-6Al-4V", "Aluminum 6061-T6", "Aluminum 7075-T6"],
        "min_expected": 2,
        "category": "comparison"
    },
    
    # Category 5: Sustainability queries
    {
        "query": "most sustainable materials",
        "expected_materials": ["Bamboo", "Wood (Oak)", "Natural Rubber"],
        "min_expected": 2,
        "category": "sustainability"
    },
    {
        "query": "recyclable metals",
        "expected_materials": ["Aluminum 6061-T6", "Stainless Steel 304", "Copper C110"],
        "min_expected": 2,
        "category": "sustainability"
    },
    
    # Category 6: Category-based queries
    {
        "query": "show me aluminum alloys",
        "expected_materials": ["Aluminum 6061-T6", "Aluminum 7075-T6", "Aluminum 2024-T3"],
        "min_expected": 2,
        "category": "category_search"
    },
    {
        "query": "list ceramics",
        "expected_materials": ["Ceramic (Alumina Al2O3)", "Silicon Carbide", "Zirconia"],
        "min_expected": 2,
        "category": "category_search"
    },
    
    # Category 7: Complex multi-constraint
    {
        "query": "strong lightweight affordable metal",
        "expected_materials": ["Aluminum 7075-T6", "Carbon Steel 1045"],
        "min_expected": 1,
        "category": "multi_constraint"
    },
]


def run_single_test(test_case: Dict, api_url: str = "http://127.0.0.1:8000/query", 
                   use_hybrid: bool = True) -> Dict:
    """
    Run a single test case and evaluate results.
    """
    query = test_case["query"]
    expected_materials = test_case["expected_materials"]
    min_expected = test_case["min_expected"]
    
    # Send query to API
    start_time = time.time()
    response = requests.post(api_url, json={
        "question": query,
        "chat_history": [],
        "use_hybrid": use_hybrid
    })
    response_time = time.time() - start_time
    
    if response.status_code != 200:
        return {
            "query": query,
            "success": False,
            "error": f"API returned {response.status_code}",
            "response_time": response_time
        }
    
    data = response.json()
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    
    # Extract material names from sources
    retrieved_materials = [src["source"].replace("Materials Database - ", "") 
                          for src in sources]
    
    # Check how many expected materials were found
    found_materials = [mat for mat in expected_materials 
                      if any(mat.lower() in retrieved.lower() 
                            for retrieved in retrieved_materials)]
    
    # Calculate metrics
    precision = len(found_materials) / len(retrieved_materials) if retrieved_materials else 0
    recall = len(found_materials) / len(expected_materials) if expected_materials else 0
    passed = len(found_materials) >= min_expected
    
    return {
        "query": query,
        "category": test_case["category"],
        "success": True,
        "passed": passed,
        "expected_materials": expected_materials,
        "found_materials": found_materials,
        "retrieved_materials": retrieved_materials,
        "precision": precision,
        "recall": recall,
        "response_time": response_time,
        "retrieval_method": data.get("retrieval_method", "unknown")
    }


def run_evaluation_suite(api_url: str = "http://127.0.0.1:8000/query",
                        use_hybrid: bool = True,
                        save_results: bool = True) -> Dict:
    """
    Run full evaluation suite and return results.
    """
    print("=" * 80)
    print(f"HELIOS EVALUATION SUITE")
    print(f"Method: {'Hybrid Retrieval' if use_hybrid else 'Semantic Only'}")
    print("=" * 80)
    
    results = []
    category_stats = {}
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] Testing: {test_case['query'][:60]}...")
        
        result = run_single_test(test_case, api_url, use_hybrid)
        results.append(result)
        
        # Track by category
        category = result.get("category", "unknown")
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1
        if result.get("passed", False):
            category_stats[category]["passed"] += 1
        
        # Print result
        status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
        print(f"   {status} - Found {len(result.get('found_materials', []))} of {len(result.get('expected_materials', []))} expected")
        print(f"   Response time: {result.get('response_time', 0):.2f}s")
    
    # Calculate overall stats
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("passed", False))
    avg_precision = sum(r.get("precision", 0) for r in results) / total_tests
    avg_recall = sum(r.get("recall", 0) for r in results) / total_tests
    avg_response_time = sum(r.get("response_time", 0) for r in results) / total_tests
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"\nAverage Precision: {avg_precision:.2%}")
    print(f"Average Recall: {avg_recall:.2%}")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    
    print(f"\nResults by Category:")
    for category, stats in sorted(category_stats.items()):
        pass_rate = stats["passed"] / stats["total"] * 100
        print(f"  {category:20s}: {stats['passed']}/{stats['total']} ({pass_rate:.0f}%)")
    
    # Save results
    if save_results:
        method_name = "hybrid" if use_hybrid else "semantic"
        filename = f"evaluation/results_{method_name}.json"
        with open(filename, 'w') as f:
            json.dump({
                "method": method_name,
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "pass_rate": passed_tests / total_tests,
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "avg_response_time": avg_response_time
                },
                "category_stats": category_stats,
                "detailed_results": results
            }, f, indent=2)
        print(f"\n✅ Results saved to {filename}")
    
    print("=" * 80)
    
    return {
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_response_time": avg_response_time
        },
        "category_stats": category_stats,
        "results": results
    }


def compare_methods():
    """
    Compare hybrid vs semantic-only retrieval.
    """
    print("\n" + "=" * 80)
    print("COMPARING RETRIEVAL METHODS")
    print("=" * 80)
    
    print("\n1️⃣  Running with HYBRID retrieval...")
    hybrid_results = run_evaluation_suite(use_hybrid=True, save_results=True)
    
    print("\n2️⃣  Running with SEMANTIC ONLY retrieval...")
    semantic_results = run_evaluation_suite(use_hybrid=False, save_results=True)
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    hybrid_pass_rate = hybrid_results["summary"]["pass_rate"]
    semantic_pass_rate = semantic_results["summary"]["pass_rate"]
    improvement = (hybrid_pass_rate - semantic_pass_rate) * 100
    
    print(f"\nPass Rate:")
    print(f"  Hybrid:   {hybrid_pass_rate:.1%}")
    print(f"  Semantic: {semantic_pass_rate:.1%}")
    print(f"  Improvement: {improvement:+.1f} percentage points")
    
    print(f"\nPrecision:")
    print(f"  Hybrid:   {hybrid_results['summary']['avg_precision']:.2%}")
    print(f"  Semantic: {semantic_results['summary']['avg_precision']:.2%}")
    
    print(f"\nRecall:")
    print(f"  Hybrid:   {hybrid_results['summary']['avg_recall']:.2%}")
    print(f"  Semantic: {semantic_results['summary']['avg_recall']:.2%}")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    # Create evaluation directory if it doesn't exist
    import os
    os.makedirs("evaluation", exist_ok=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_methods()
    else:
        run_evaluation_suite(use_hybrid=True, save_results=True)
