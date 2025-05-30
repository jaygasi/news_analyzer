# biomed_analyzer/diagnose.py
"""
Production diagnostic tool for the Biomedical News Analyzer
Run this FIRST to diagnose any configuration or connection issues
"""

import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def check_imports() -> Tuple[bool, List[str]]:
    """Check if all required packages are importable"""
    print("🔍 Checking package imports...")
    results = []
    all_good = True
    
    packages = [
        ("python-dotenv", "dotenv"),
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        ("google-generativeai", "google.generativeai"),
        ("finnhub-python", "finnhub")
    ]
    
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            results.append(f"  ✓ {package_name}")
        except ImportError as e:
            results.append(f"  ✗ {package_name} - {e}")
            all_good = False
    
    return all_good, results

def check_environment() -> Tuple[bool, List[str]]:
    """Check environment variables and .env file"""
    print("🔍 Checking environment configuration...")
    results = []
    all_good = True
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        results.append(f"  ✓ .env file found")
        
        # Load and check contents
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check required API keys
            api_keys = {
                "GEMINI_API_KEY": "Required for LLM analysis",
                "FINNHUB_API_KEY": "Optional - for news and FDA calendar", 
                "FMP_API_KEY": "Optional - for additional news coverage"
            }
            
            found_keys = 0
            for key, description in api_keys.items():
                value = os.getenv(key)
                if value and value.strip():
                    if len(value.strip()) < 10:
                        results.append(f"  ⚠ {key} - Too short (likely invalid)")
                        if key == "GEMINI_API_KEY":
                            all_good = False
                    else:
                        results.append(f"  ✓ {key} - Present")
                        found_keys += 1
                else:
                    results.append(f"  ✗ {key} - Missing ({description})")
                    if key == "GEMINI_API_KEY":
                        all_good = False
            
            # Check if at least one news API is available
            has_news_api = (os.getenv("FINNHUB_API_KEY") and len(os.getenv("FINNHUB_API_KEY").strip()) >= 10) or \
                          (os.getenv("FMP_API_KEY") and len(os.getenv("FMP_API_KEY").strip()) >= 10)
            
            if not has_news_api:
                results.append("  ✗ No valid news API key found (need either Finnhub or FMP)")
                all_good = False
            else:
                results.append("  ✓ At least one news API key is available")
                
        except Exception as e:
            results.append(f"  ✗ Error loading .env file: {e}")
            all_good = False
    else:
        results.append(f"  ✗ .env file not found")
        all_good = False
    
    return all_good, results

def test_api_connections() -> Tuple[bool, List[str]]:
    """Test actual API connections"""
    print("🔍 Testing API connections...")
    results = []
    all_good = True
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Test Gemini API
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and gemini_key.strip():
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                # Quick test
                response = model.generate_content("Test", request_options={'timeout': 10})
                if response and hasattr(response, 'text'):
                    results.append("  ✓ Gemini API - Connection successful")
                else:
                    results.append("  ✗ Gemini API - Invalid response")
                    all_good = False
            except Exception as e:
                results.append(f"  ✗ Gemini API - Connection failed: {str(e)[:100]}")
                all_good = False
        else:
            results.append("  ✗ Gemini API - No key provided")
            all_good = False
        
        # Test Finnhub API
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        if finnhub_key and finnhub_key.strip():
            try:
                import finnhub
                client = finnhub.Client(api_key=finnhub_key)
                result = client.company_profile2(symbol='AAPL')
                if result and isinstance(result, dict) and 'name' in result:
                    results.append(f"  ✓ Finnhub API - Connection successful (test: {result['name']})")
                else:
                    results.append("  ✗ Finnhub API - Invalid response format")
                    all_good = False
            except Exception as e:
                error_str = str(e)
                if "Invalid Response" in error_str and "<!DOCTYPE html>" in error_str:
                    results.append("  ✗ Finnhub API - Invalid API key (receiving HTML)")
                else:
                    results.append(f"  ✗ Finnhub API - Connection failed: {error_str[:100]}")
                all_good = False
        else:
            results.append("  ⚠ Finnhub API - No key provided (optional)")
        
        # Test FMP API
        fmp_key = os.getenv("FMP_API_KEY")
        if fmp_key and fmp_key.strip():
            try:
                import requests
                response = requests.get(
                    f"https://financialmodelingprep.com/api/v3/profile/AAPL",
                    params={'apikey': fmp_key},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        results.append(f"  ✓ FMP API - Connection successful (test: {data[0].get('companyName', 'Unknown')})")
                    else:
                        results.append("  ✗ FMP API - Empty or invalid response")
                        all_good = False
                elif response.status_code == 401:
                    results.append("  ✗ FMP API - Invalid API key (401 Unauthorized)")
                    all_good = False
                else:
                    results.append(f"  ✗ FMP API - HTTP {response.status_code}")
                    all_good = False
            except Exception as e:
                results.append(f"  ✗ FMP API - Connection failed: {str(e)[:100]}")
                all_good = False
        else:
            results.append("  ⚠ FMP API - No key provided (optional)")
    
    except Exception as e:
        results.append(f"  ✗ Error during API testing: {e}")
        all_good = False
    
    return all_good, results

def test_application_modules() -> Tuple[bool, List[str]]:
    """Test if application modules can be imported"""
    print("🔍 Testing application modules...")
    results = []
    all_good = True
    
    modules = [
        "analyzer.config",
        "analyzer.utils", 
        "analyzer.models",
        "analyzer.llm_services",
    ]
    
    for module in modules:
        try:
            __import__(module)
            results.append(f"  ✓ {module}")
        except Exception as e:
            results.append(f"  ✗ {module} - {str(e)[:100]}")
            all_good = False
    
    return all_good, results

def main():
    """Run complete diagnostic"""
    print("=" * 60)
    print("🏥 BIOMEDICAL NEWS ANALYZER - DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_checks_passed = True
    
    # Run all checks
    checks = [
        ("Package Imports", check_imports),
        ("Environment Configuration", check_environment), 
        ("Application Modules", test_application_modules),
        ("API Connections", test_api_connections),
    ]
    
    for check_name, check_func in checks:
        try:
            passed, results = check_func()
            all_checks_passed = all_checks_passed and passed
            
            for result in results:
                print(result)
            print()
        except Exception as e:
            print(f"  ✗ {check_name} - Diagnostic failed: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            all_checks_passed = False
            print()
    
    # Final summary
    print("=" * 60)
    if all_checks_passed:
        print("🎉 ALL DIAGNOSTICS PASSED!")
        print("Your application should be ready to run.")
        print("Try: python main.py --help")
    else:
        print("❌ SOME DIAGNOSTICS FAILED")
        print()
        print("Common fixes:")
        print("1. Missing packages: pip install -r requirements.txt")
        print("2. Missing .env file: Copy .env.example to .env and add your API keys")
        print("3. Invalid API keys: Check your keys at:")
        print("   - Finnhub: https://finnhub.io/dashboard")
        print("   - FMP: https://financialmodelingprep.com/developer/docs")
        print("   - Gemini: https://ai.google.dev/")
        print("4. Module errors: Check file paths and imports")
    print("=" * 60)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())