import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Check if API key is loaded
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("=" * 70)
        print("❌ ERROR: ANTHROPIC_API_KEY not found!")
        print("=" * 70)
        print("\nPlease create a .env file in the project root with:")
        print("ANTHROPIC_API_KEY=sk-ant-your-key-here")
        print("\nOr export it in your terminal:")
        print("export ANTHROPIC_API_KEY='sk-ant-your-key-here'")
        print("=" * 70)
        exit(1)
    
    print("=" * 70)
    print("🚀 Starting Helios Engineering Assistant")
    print("=" * 70)
    print(f"Environment: {os.getenv('HELIOS_ENV', 'development')}")
    print(f"Anthropic API Key: {'✅ Loaded' if os.getenv('ANTHROPIC_API_KEY') else '❌ Missing'}")
    print(f"LangSmith Tracing: {'✅ Enabled' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else 'ℹ️  Disabled'}")
    print("=" * 70)
    print()

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["."]
    )
