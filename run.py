import uvicorn
import os

if __name__ == "__main__":
    # Ensure the API key is set if you have a .env file,
    # otherwise, you'll still need to set it in the terminal.
    # This script doesn't handle .env loading by itself.
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        # You might want to exit here or load from a .env file
        # For now, we'll let it proceed and potentially fail in main.py

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["."]  # Watch only the current directory
    )
