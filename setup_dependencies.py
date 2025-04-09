import os
import subprocess

def install_dependencies():
    # Install Python dependencies
    print("Installing Python dependencies...")
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Install Ollama
    print("Installing Ollama...")
    subprocess.check_call(["brew", "install", "ollama"])

    # Pull the required model for Ollama
    print("Pulling the required model for Ollama...")
    subprocess.check_call(["ollama", "pull", "deepseek-r1:32b-qwen-distill-q4_K_M"])

if __name__ == "__main__":
    try:
        install_dependencies()
        print("All dependencies installed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")