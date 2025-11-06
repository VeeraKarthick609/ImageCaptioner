#!/usr/bin/env python3
"""
Simple script to run the Streamlit Image Caption Generator app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    print("🚀 Starting Image Caption Generator...")
    print("📸 Upload images and get AI-generated captions!")
    print("-" * 50)
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        print("\n💡 Make sure you have installed the requirements:")
        print("pip install -r requirements_streamlit.txt")
    except FileNotFoundError:
        print("❌ Streamlit not found!")
        print("\n💡 Install streamlit first:")
        print("pip install streamlit")

if __name__ == "__main__":
    main()
