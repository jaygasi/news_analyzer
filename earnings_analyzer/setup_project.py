#!/usr/bin/env python3
"""
Quick Setup Script for Earnings News Trader
Handles installation, configuration, and initial testing
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_step(step_num, description):
    """Print a step description"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def run_command(command, description=""):
    """Run a command and handle errors"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected (compatible)")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("Creating virtual environment...")
    
    if os.path.exists("venv"):
        print("Virtual environment already exists")
        return True
    
    if not run_command("python -m venv venv", "creating virtual environment"):
        return False
    
    print("✅ Virtual environment created successfully")
    return True

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Linux/Mac
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "installing requirements"):
        return False
    
    print("✅ Dependencies installed successfully")
    return True

def setup_environment_file():
    """Setup .env configuration file"""
    print("Setting up environment configuration...")
    
    if os.path.exists(".env"):
        print("⚠️ .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Keeping existing .env file")
            return True
    
    # Copy template
    if os.path.exists(".env.template"):
        shutil.copy(".env.template", ".env")
        print("✅ Created .env file from template")
    else:
        # Create basic .env file
        env_content = """# Earnings News Trader Configuration
# Fill in your API keys below

# Google Gemini API (Recommended - Free tier available)
GOOGLE_API_KEY=your_google_api_key_here
USE_GEMINI=true

# Gmail for notifications
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password_here

# Finnhub for earnings data (Free tier available)
FINNHUB_API_KEY=your_finnhub_api_key_here

# System settings
LOG_LEVEL=INFO
MIN_CONFIDENCE_SCORE=0.7
MAX_POSITION_SIZE=0.05
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created basic .env file")
    
    print("\n🔧 Next: Edit the .env file and add your API keys")
    print("   - Get Google Gemini API key: https://makersuite.google.com/app/apikey")
    print("   - Get Finnhub API key: https://finnhub.io/register")
    print("   - Setup Gmail App Password: https://support.google.com/accounts/answer/185833")
    
    return True

def test_configuration():
    """Test the configuration"""
    print("Testing configuration...")
    
    # Test basic imports
    try:
        from config.config import Config
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test database connection
    try:
        from src.database.connection import test_connection, create_tables
        if test_connection():
            print("✅ Database connection successful")
            create_tables()
            print("✅ Database tables created")
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    return True

def show_next_steps():
    """Show what to do next"""
    print_header("🎉 Setup Complete!")
    
    print("""
🚀 Your Earnings News Trader is ready! Here's what to do next:

1. 📝 Configure your API keys in the .env file:
   - Edit .env and add your actual API keys
   - At minimum, you need Google Gemini and Gmail credentials

2. 🧪 Test the system:
   python main.py test-earnings    # Test earnings data collection
   python main.py test-news        # Test news collection

3. 🏃 Run the system:
   python main.py run              # Single collection cycle
   python main.py schedule         # Continuous monitoring

4. 📊 Check status:
   python main.py status           # View system status

📚 Documentation:
   - Check README.md for detailed setup instructions
   - See logs/earnings_trader.log for system logs
   - Visit the GitHub repository for updates

⚠️ Important:
   - This is for educational purposes only
   - Always do your own research before trading
   - Start with paper trading to test strategies
""")

def main():
    """Main setup process"""
    print_header("🚀 Earnings News Trader - Quick Setup")
    
    print("""
This script will help you set up the Earnings News Trader system.
The setup process includes:

1. Check Python version compatibility
2. Create virtual environment
3. Install required dependencies
4. Setup configuration files
5. Test basic functionality

Press Enter to continue or Ctrl+C to cancel...
""")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create virtual environment
    print_step(2, "Creating virtual environment")
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Step 3: Install dependencies
    print_step(3, "Installing dependencies")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        print("💡 Try running: pip install -r requirements.txt manually")
        sys.exit(1)
    
    # Step 4: Setup environment file
    print_step(4, "Setting up configuration")
    if not setup_environment_file():
        sys.exit(1)
    
    # Step 5: Test configuration
    print_step(5, "Testing basic configuration")
    if not test_configuration():
        print("⚠️ Basic configuration test failed")
        print("💡 This might be due to missing API keys in .env file")
        print("💡 Configure your .env file and run: python main.py test")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
