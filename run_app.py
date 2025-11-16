"""
Simple script to run the Streamlit app with better output visibility
"""
import subprocess
import sys

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Streamlit RAG Pipeline App...")
    print("=" * 50)
    print("\nThe app will open in your default browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_rag_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\nApp stopped by user.")
    except Exception as e:
        print(f"\nError running app: {e}")
        print("\nMake sure Streamlit is installed: pip install streamlit")

