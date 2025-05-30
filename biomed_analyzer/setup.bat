@echo off
REM filepath: c:\Users\JayHy\OneDrive\Projects\news_analyzer\biomed_analyzer\setup.bat
setlocal enabledelayedexpansion

echo Setting up Biomedical News Analyzer...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH! Please install Python 3.13.3 or later.
    exit /b 1
)

REM Check Python version
python -c "import sys; sys.exit(0 if sys.version_info >= (3,13,3) else 1)"
if %ERRORLEVEL% neq 0 (
    echo Python 3.13.3 or later is required!
    exit /b 1
)

REM Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists...
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    (
        echo FINNHUB_API_KEY=your_finnhub_key_here
        echo GEMINI_API_KEY=your_gemini_key_here
        echo FMP_API_KEY=your_fmp_key_here
        echo DISCORD_WEBHOOK_URL=your_discord_webhook_here
        echo DATABASE_URL=sqlite:///./biomedical_analyzer.db
    ) > .env
    echo Please update the .env file with your API keys
)

REM Create necessary directories
mkdir logs 2>nul
mkdir cache 2>nul

REM Initialize database
echo Initializing database...
python -c "from biomed_analyzer.database.models import Base; from biomed_analyzer.database.session import engine; Base.metadata.create_all(bind=engine)" 2>nul || echo Database initialization skipped (optional component)

echo.
echo Setup complete! Please follow these steps:
echo 1. Edit the .env file with your API keys:
echo    - FINNHUB_API_KEY: Get from https://finnhub.io/
echo    - FMP_API_KEY: Get from https://financialmodelingprep.com/
echo    - GEMINI_API_KEY: Get from Google AI Studio
echo 2. Run 'venv\Scripts\activate.bat' to activate the virtual environment
echo 3. Run 'python main.py --help' to see available options
echo 4. You can disable either API with --disable-finnhub or --disable-fmp if needed
echo.

pause