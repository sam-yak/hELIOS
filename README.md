# Helios ☀️ Engineering Assistant

> An AI-powered materials selection assistant using advanced RAG (Retrieval-Augmented Generation) with hybrid retrieval, comprehensive materials database, and production-grade observability.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.27-orange.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

Helios is a conversational AI assistant designed to help engineers select appropriate materials for their projects. It combines semantic search with keyword matching, leverages a comprehensive materials database, and provides intelligent recommendations based on multiple constraints.

### Key Features

- 🔍 **Hybrid Retrieval**: Combines semantic embeddings (ChromaDB) with keyword search (BM25)
- 📊 **72 Engineering Materials**: Comprehensive database with complete technical, economic, and sustainability data
- 💬 **Conversational Interface**: Natural language queries with context-aware responses
- 📈 **Production Observability**: LangSmith integration for full distributed tracing
- ⚡ **Fast Response Times**: Average 2-3 seconds per query
- 📁 **Data Export**: Export material data in JSON, CSV, or TXT formats
- 🔐 **Environment-based Config**: Secure credential management with .env files

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Database](#database)
- [API Documentation](#api-documentation)
- [Evaluation](#evaluation)
- [Development](#development)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           Frontend (HTML/JS)             │
│  - Query input & chat history           │
│  - Results display with sources         │
│  - Export functionality                 │
└────────┬────────────────────────────────┘
         │ POST /query
         ▼
┌─────────────────────────────────────────┐
│      FastAPI Backend (main.py)          │
│  - Request validation                   │
│  - Logging & metrics                    │
│  - Response formatting                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       Hybrid Retrieval System           │
│  ┌────────────────┬─────────────────┐  │
│  │ Semantic       │ Keyword (BM25)  │  │
│  │ (ChromaDB)     │                 │  │
│  │ 60% weight     │ 40% weight      │  │
│  └────────┬───────┴────────┬────────┘  │
│           │                 │           │
│           └────────┬────────┘           │
│                    │                    │
│              Ensemble Merge             │
└────────┬────────────────────────────────┘
         │ Top 5 Documents
         ▼
┌─────────────────────────────────────────┐
│         Materials Database              │
│  - 72 materials across 27 categories   │
│  - Physical, mechanical, thermal props  │
│  - Cost, sustainability, applications   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       OpenAI GPT-4 (LLM)                │
│  - Context: Retrieved documents         │
│  - Chat history for context             │
│  - Precise engineering responses        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Response with Sources              │
│  - Natural language answer              │
│  - Source documents cited               │
│  - Material metadata                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     LangSmith (Observability)           │
│  - Distributed tracing                  │
│  - Performance monitoring               │
│  - Query analytics                      │
└─────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- **FastAPI**: High-performance web framework
- **LangChain**: Orchestration for LLM applications
- **OpenAI GPT-4**: Language model for answer generation
- **ChromaDB**: Vector database for semantic search
- **BM25**: Keyword-based retrieval algorithm

**Frontend:**
- Vanilla JavaScript with Markdown rendering
- Responsive design with clean UI

**Observability:**
- LangSmith for distributed tracing
- Custom logging with structured logs
- Performance metrics tracking

**Data:**
- 72 materials with complete properties
- Unified JSON database
- 27 material categories

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- LangSmith account (optional, for tracing)
- 2GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/helios.git
cd helios
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=sk-proj-your-key-here

# LangSmith Configuration (OPTIONAL)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_your-key-here
LANGCHAIN_PROJECT=helios-prod

# Application Configuration
HELIOS_ENV=development
LOG_LEVEL=INFO
```

**Getting API Keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **LangSmith**: https://smith.langchain.com/ (free tier available)

### Step 5: Build the Vector Database

```bash
# This creates the ChromaDB database from materials_database.json
python ingest_v2.py
```

Expected output:
```
======================================================================
HELIOS MATERIALS DATABASE INGESTION v2.0
======================================================================
✅ Created 72 documents with rich metadata.
✅ Split into 72 chunks.
✅ Embedding model loaded.
✅ Vector store created and persisted.
```

### Step 6: Start the Server

```bash
python run.py
```

Server will start at: **http://127.0.0.1:8000**

---

## 🚀 Quick Start

### Basic Usage

1. **Open your browser** to http://127.0.0.1:8000
2. **Enter a query** in the text box, for example:
   - "lightweight metal for aerospace"
   - "compare titanium and aluminum"
   - "materials with yield strength over 800 MPa"
3. **View results** with source citations
4. **Export data** if a specific material is detected

### Example Queries

**Property-Based:**
```
Find materials with density less than 3 g/cc
```

**Application-Based:**
```
What materials are suitable for high temperature applications?
```

**Comparison:**
```
Compare stainless steel 304 and 316 for marine use
```

**Category:**
```
Show me all aluminum alloys in the database
```

---

## 💡 Usage

### Command Line Interface

#### Start the Server
```bash
python run.py
```

#### Run Database Ingestion
```bash
python ingest_v2.py
```

#### Run Evaluation Suite
```bash
# Run hybrid retrieval tests
python evaluation/test_suite.py

# Compare hybrid vs semantic-only
python evaluation/test_suite.py compare

# Run realistic tests (only materials in DB)
python evaluation/test_suite_realistic.py compare
```

#### Verify Data
```bash
# Check materials count
python migrate_data.py
```

### API Usage

#### Query Endpoint

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "lightweight corrosion resistant metal",
    "chat_history": [],
    "use_hybrid": true
  }'
```

Response:
```json
{
  "answer": "Based on the materials in the database...",
  "sources": [
    {
      "source": "Materials Database - Aluminum 6061-T6",
      "content": "Material: Aluminum 6061-T6\n..."
    }
  ],
  "detected_material": "Aluminum 6061-T6",
  "retrieval_method": "hybrid"
}
```

#### Compare Retrieval Methods

```bash
curl -X POST http://127.0.0.1:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"query": "strong lightweight metal", "k": 5}'
```

#### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "features": ["hybrid_retrieval", "langsmith_tracing"],
  "materials_count": 72,
  "environment": "development",
  "langsmith_enabled": true
}
```

---

## 🗄️ Database

### Materials Database Structure

The `materials_database.json` contains 72 materials with complete data:

```json
{
  "Material Name": {
    "material_name": "Aluminum 6061-T6",
    "category": "Aluminum Alloys",
    "material_notes": "Description...",
    "density": 2.70,
    "tensile_strength_ultimate": 310,
    "tensile_strength_yield": 276,
    "modulus_of_elasticity": 68.9,
    "thermal_conductivity": 167,
    "melting_point": 617,
    "cost_per_kg_usd": 3.5,
    "sustainability_score": 7,
    "sustainability_notes": "...",
    "common_applications": [
      "Aircraft fittings",
      "Marine hardware"
    ]
  }
}
```

### Material Categories (27 total)

- Aluminum Alloys (4)
- Stainless Steels (4)
- Carbon Steels (3)
- Tool Steels (2)
- Titanium Alloys (3)
- Nickel Superalloys (3)
- Nickel Alloys (2)
- Copper Alloys (4)
- Refractory Metals (4)
- Thermoplastics (10)
- High-Performance Polymers (2)
- Fluoropolymers (1)
- Elastomers (5)
- Composites (3)
- Ceramics (4)
- Glass (2)
- Natural Materials (3)
- Precious Metals (3)
- Cast Irons (2)
- And more...

### Adding New Materials

1. Edit `materials_database.json`
2. Add new material following the structure above
3. Run: `python ingest_v2.py` to rebuild the database
4. Restart server: `python run.py`

### Database Maintenance

**Rebuild from scratch:**
```bash
rm -rf db/
python ingest_v2.py
```

**Verify integrity:**
```bash
python migrate_data.py
```

---

## 📖 API Documentation

### Endpoints

#### `GET /`
Serves the frontend HTML interface.

#### `POST /query`
Main query endpoint for material searches.

**Request Body:**
```json
{
  "question": "string",
  "chat_history": [
    {"role": "user", "content": "..."},
    {"role": "ai", "content": "..."}
  ],
  "use_hybrid": true
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {"source": "string", "content": "string"}
  ],
  "detected_material": "string | null",
  "retrieval_method": "hybrid | semantic_only"
}
```

#### `POST /compare`
Compare different retrieval methods.

**Request:**
```json
{
  "query": "string",
  "k": 5
}
```

**Response:**
```json
{
  "query": "string",
  "results": {
    "semantic": ["Material A", "Material B"],
    "keyword": ["Material C", "Material D"],
    "hybrid": ["Material A", "Material C"]
  },
  "analysis": {
    "semantic_count": 2,
    "keyword_count": 2,
    "hybrid_count": 2
  }
}
```

#### `POST /export`
Export material data.

**Request:**
```json
{
  "material_name": "string",
  "export_format": "json | csv | txt"
}
```

**Response:** File download

#### `GET /health`
Health check endpoint.

#### `GET /stats`
Database statistics.

---

## 🧪 Evaluation

### Running Tests

The evaluation framework tests retrieval accuracy across multiple query types.

```bash
# Run hybrid retrieval evaluation
python evaluation/test_suite.py

# Compare hybrid vs semantic-only
python evaluation/test_suite.py compare

# Run realistic tests (recommended)
python evaluation/test_suite_realistic.py compare
```

### Test Categories

1. **Exact Match**: Material name queries
2. **Comparison**: Compare two materials
3. **Category Search**: Find materials by category
4. **Application**: Materials for specific uses
5. **Semantic Property**: Fuzzy property descriptions
6. **Material Type**: General material classes

### Metrics

- **Pass Rate**: Percentage of tests where minimum materials found
- **Precision**: Relevant materials / Total retrieved
- **Recall**: Found expected / Total expected
- **Response Time**: Average query latency

### Sample Results

```
================================================================================
EVALUATION SUMMARY
================================================================================
Total Tests: 20
Passed: 16 (80.0%)

Average Precision: 72.5%
Average Recall: 85.3%
Average Response Time: 2.34s

Results by Category:
  exact_match        : 3/3 (100%)
  comparison         : 3/3 (100%)
  category_search    : 4/4 (100%)
  application        : 3/4 (75%)
  semantic_property  : 2/3 (67%)
  material_type      : 1/3 (33%)
```

---

## 🛠️ Development

### Project Structure

```
hELIOS/
├── main.py                      # FastAPI application
├── run.py                       # Server startup script
├── ingest_v2.py                 # Database ingestion
├── migrate_data.py              # Data migration utility
├── materials_database.json      # Unified materials database
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
├── .gitignore                   # Git ignore rules
│
├── frontend/
│   └── index.html              # Web interface
│
├── utils/
│   ├── __init__.py
│   └── logger.py               # Production logging
│
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py     # Hybrid retrieval logic
│   └── document_loader.py      # Document loading for BM25
│
├── evaluation/
│   ├── __init__.py
│   ├── test_suite.py           # Full evaluation suite
│   ├── test_suite_realistic.py # Realistic tests
│   ├── results_hybrid.json     # Test results
│   └── results_semantic.json   # Comparison results
│
├── logs/
│   ├── helios.log              # Detailed system logs
│   └── metrics.log             # Performance metrics (CSV)
│
├── db/                         # ChromaDB vector database
│   ├── chroma.sqlite3
│   └── [embeddings data]
│
└── docs/
    ├── README.md               # This file
    ├── WEEK_1_SUMMARY.md      # Week 1 documentation
    └── WEEK_2_RESULTS.md      # Week 2 results
```

### Code Architecture

**Main Components:**

1. **main.py**: FastAPI application with endpoints
2. **hybrid_retriever.py**: Ensemble retrieval (semantic + BM25)
3. **document_loader.py**: Loads documents for BM25 indexing
4. **logger.py**: Structured logging with metrics
5. **test_suite.py**: Automated evaluation framework

### Development Workflow

```bash
# 1. Make code changes
vim main.py

# 2. Server auto-reloads (if running with run.py)

# 3. Test changes
curl -X POST http://127.0.0.1:8000/query -d '...'

# 4. Run evaluation
python evaluation/test_suite_realistic.py

# 5. Check logs
tail -f logs/helios.log
```

### Adding New Features

1. **New Retrieval Method:**
   - Add to `retrieval/` directory
   - Integrate in `main.py`
   - Update evaluation tests

2. **New Material Properties:**
   - Update `materials_database.json` structure
   - Modify `document_loader.py` to extract new properties
   - Update metadata in `ingest_v2.py`
   - Rebuild database

3. **New API Endpoint:**
   - Add route in `main.py`
   - Add logging
   - Update API documentation

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | - | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `helios-prod` | LangSmith project name |
| `HELIOS_ENV` | No | `development` | Environment name |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### Retrieval Configuration

Edit `retrieval/hybrid_retriever.py`:

```python
def create_hybrid_retriever(...):
    return HybridMaterialRetriever(
        vector_store=vector_store,
        documents=documents,
        semantic_weight=0.6,  # Adjust weight
        keyword_weight=0.4    # Adjust weight
    )
```

### Logging Configuration

Edit `utils/logger.py`:

```python
def setup_logger(name, log_file, level=logging.INFO):
    # Customize log format, handlers, etc.
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "OPENAI_API_KEY not found"
**Solution:**
```bash
# Check .env file exists
cat .env

# Verify key is set
echo $OPENAI_API_KEY

# Re-run with explicit export
export OPENAI_API_KEY='sk-proj-...'
python run.py
```

#### 2. "Database not found" or "db/ directory missing"
**Solution:**
```bash
# Rebuild database
python ingest_v2.py
```

#### 3. "Module not found" errors
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install rank-bm25
```

#### 4. Slow response times
**Possible causes:**
- OpenAI API rate limits
- Large number of materials
- Network latency

**Solutions:**
- Check OpenAI API status
- Reduce `k` parameter in retrieval
- Use caching (future feature)

#### 5. Import errors with ChromaDB
**Solution:**
```bash
# ChromaDB might have deprecation warnings
# Install updated version (optional)
pip install langchain-chroma

# Or ignore warnings (they're non-fatal)
```

### Debug Mode

Enable detailed logging:

```bash
# In .env
LOG_LEVEL=DEBUG

# Restart server
python run.py
```

Check logs:
```bash
tail -f logs/helios.log
```

### Performance Issues

Monitor metrics:
```bash
# View performance data
cat logs/metrics.log

# Analyze response times
grep "Response Time" logs/helios.log
```

---

## 📈 Performance

### Benchmarks

**System Specifications:**
- MacBook Air M1
- 8GB RAM
- Python 3.13

**Performance Metrics:**
- Average Response Time: 2.3 seconds
- P50 Response Time: 2.1 seconds
- P95 Response Time: 3.5 seconds
- P99 Response Time: 7.2 seconds

**Throughput:**
- ~25 queries per minute (sequential)
- Database size: 17MB (72 materials)
- Memory usage: ~500MB

### Optimization Tips

1. **Reduce retrieval size:**
   ```python
   # In main.py, reduce k value
   retrieved_docs = hybrid_retriever.retrieve(query, k=3)  # Default: 5
   ```

2. **Cache embeddings** (future feature)

3. **Use lighter LLM:**
   ```python
   llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Faster, cheaper
   ```

4. **Batch processing** for multiple queries

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests:**
   ```bash
   python evaluation/test_suite_realistic.py
   ```
5. **Commit your changes:**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to branch:**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Update tests for new features
- Update README with new functionality
- Add logging for debugging

---

## 🗺️ Roadmap

### Week 3 (Planned)
- [ ] Multi-step agentic workflows
- [ ] Tool-based reasoning
- [ ] Multi-constraint optimization (Pareto analysis)
- [ ] Material compatibility graph

### Week 4 (Planned)
- [ ] Fine-tuned embeddings for materials domain
- [ ] Advanced filtering with self-query improvements
- [ ] Cost estimation tools
- [ ] Supplier integration

### Future Enhancements
- [ ] Expand to 500+ materials
- [ ] Real-time supplier data integration
- [ ] Material comparison visualizations
- [ ] Mobile-responsive frontend
- [ ] Multi-language support
- [ ] PDF report generation
- [ ] Integration with CAD tools
- [ ] Collaborative material selection
- [ ] Material lifecycle analysis

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 API
- **LangChain** for RAG framework
- **ChromaDB** for vector storage
- **FastAPI** for web framework
- **HuggingFace** for embedding models
- **MatWeb** for materials data inspiration

---

## 📧 Contact

**Project Maintainer:** [Your Name]
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

**Project Link:** https://github.com/yourusername/helios

---

## 📊 Project Stats

- **Version:** 2.1.0
- **Materials:** 72
- **Categories:** 27
- **Retrieval Methods:** 2 (Semantic + Keyword)
- **Supported Formats:** JSON, CSV, TXT
- **Response Time:** ~2.3s average
- **Test Coverage:** 20+ test cases
- **Lines of Code:** ~2,500+

---

## 🎓 For Acqui-hire / Job Applications

This project demonstrates:

✅ **Production RAG Implementation**
- Hybrid retrieval combining semantic + keyword search
- Proper prompt engineering and chain orchestration
- Context-aware conversational interface

✅ **Data Engineering**
- Designed unified data schema
- ETL pipeline for materials ingestion
- Metadata extraction and normalization

✅ **Full-Stack Development**
- FastAPI backend with proper error handling
- Interactive frontend with real-time updates
- RESTful API design

✅ **MLOps Best Practices**
- Structured logging and monitoring
- Performance metrics tracking
- Automated evaluation framework
- Distributed tracing with LangSmith

✅ **Software Engineering**
- Clean code architecture
- Comprehensive documentation
- Environment-based configuration
- Modular, testable design

✅ **Problem Solving**
- Identified and documented system limitations
- Iterative improvement approach
- Honest assessment of trade-offs

---

**Built with ❤️ for engineers, by engineers.**