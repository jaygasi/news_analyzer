import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import schedule with error handling
try:
    import schedule
except ImportError:
    print("❌ Missing dependency: schedule")
    print("💡 Install with: pip install schedule")
    sys.exit(1)

# Import main components with better error handling
try:
    from config.config import Config
    print("✅ Configuration loaded")
except ImportError as e:
    print(f"❌ Configuration import error: {e}")
    print("💡 Make sure config/config.py exists and is properly formatted")
    sys.exit(1)

try:
    from src.database.connection import create_tables, test_connection, get_db_stats, initialize_database
    print("✅ Database connection module loaded")
except ImportError as e:
    print(f"❌ Database connection import error: {e}")
    print("💡 Make sure src/database/connection.py exists")
    sys.exit(1)

try:
    from src.collectors.news_collector import NewsCollector
    print("✅ News collector loaded")
except ImportError as e:
    print(f"❌ News collector import error: {e}")
    print("💡 Make sure src/collectors/news_collector.py exists")
    sys.exit(1)

try:
    from src.analyzers.sentiment_analyzer import SentimentAnalyzer
    print("✅ Sentiment analyzer loaded")
except ImportError as e:
    print(f"❌ Sentiment analyzer import error: {e}")
    print("💡 Make sure src/analyzers/sentiment_analyzer.py exists")
    sys.exit(1)

try:
    from src.analyzers.signal_generator import TradingSignalGenerator
    print("✅ Signal generator loaded")
except ImportError as e:
    print(f"❌ Signal generator import error: {e}")
    print("💡 Make sure src/analyzers/signal_generator.py exists")
    sys.exit(1)

try:
    from src.notifications.gmail_notifier import GmailNotifier
    print("✅ Gmail notifier loaded")
except ImportError as e:
    print(f"❌ Gmail notifier import error: {e}")
    print("💡 Make sure src/notifications/gmail_notifier.py exists")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration with proper error handling."""
    try:
        log_level_str = Config.LOG_LEVEL.upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        if not isinstance(log_level, int):
            log_level = logging.INFO
            print(f"⚠️ Invalid LOG_LEVEL '{log_level_str}' in config. Using INFO.")

        # Create logs directory
        Path("logs").mkdir(exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/earnings_trader.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        # Configure console encoding if needed
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                if sys.stdout.encoding != 'utf-8':
                    sys.stdout.reconfigure(encoding='utf-8')
            except Exception as e:
                logging.warning(f"Could not reconfigure stdout encoding: {e}")

        print("✅ Logging configured successfully")
        
    except Exception as e:
        print(f"❌ Error setting up logging: {e}")
        raise

class EarningsNewsTrader:
    """Main application class"""
    
    # Scheduling constants
    SCAN_START_HOUR = 10
    SCAN_END_HOUR = 16
    DAILY_COLLECTION_TIME = "06:00"
    MARKET_OPEN_SCAN_TIME = "09:30"
    MIDDAY_SCAN_TIME = "12:00"
    MARKET_CLOSE_SCAN_TIME = "15:30"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.news_collector = NewsCollector()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.signal_generator = TradingSignalGenerator()
            self.notifier = GmailNotifier()
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}", exc_info=True)
            raise

    def validate_setup(self):
        """Validate configuration and connections"""
        self.logger.info("🔍 Validating system setup...")
        
        try:
            # Validate configuration
            validation_result = Config.validate()
            if not validation_result['valid']:
                self.logger.error("❌ Configuration validation failed:")
                for issue in validation_result['issues']:
                    self.logger.error(f"  - {issue}")
                return False
            
            self.logger.info("✅ Configuration validated")

            # Test database connection
            if not test_connection():
                self.logger.error("❌ Database connection failed")
                return False
            
            self.logger.info("✅ Database connection successful")

            # Initialize database
            initialize_database()
            self.logger.info("✅ Database initialized")

            # Test Gmail connection
            if not self.notifier.test_connection():
                self.logger.error("❌ Gmail connection failed")
                return False
            
            self.logger.info("✅ Gmail connection successful")

            # Check LLM providers
            available_providers = self.sentiment_analyzer.get_available_llm_providers()
            current_provider = self.sentiment_analyzer.get_current_llm_provider()

            if available_providers:
                self.logger.info(f"✅ LLM providers available: {', '.join(available_providers)}")
                self.logger.info(f"✅ Primary LLM provider: {current_provider}")
            else:
                self.logger.warning("⚠️ No LLM providers available - using traditional sentiment analysis only")

            # Get database stats
            stats = get_db_stats()
            self.logger.info(f"📊 Database stats: {stats}")
            
            return True

        except Exception as e:
            self.logger.error(f"❌ Setup validation failed: {e}", exc_info=True)
            return False

    def run_daily_collection(self):
        """Run daily earnings-focused news collection and analysis"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 Starting daily collection cycle")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Collect earnings-focused news
            self.logger.info("📈 Step 1: Collecting earnings-focused news...")
            collection_stats = self.news_collector.run_earnings_focused_collection()

            self.logger.info("📊 Collection Results:")
            self.logger.info(f"   - Earnings events found: {collection_stats.get('earnings_events', 0)}")
            self.logger.info(f"   - Target tickers: {collection_stats.get('target_tickers', 0)}")
            self.logger.info(f"   - Articles collected: {collection_stats.get('articles_collected', 0)}")
            self.logger.info(f"   - Articles stored: {collection_stats.get('articles_stored', 0)}")

            # Step 2: Analyze sentiment and generate signals
            signals = []
            if collection_stats.get('articles_stored', 0) > 0:
                self.logger.info("🤖 Step 2: Analyzing sentiment and generating signals...")
                self.signal_generator.process_unanalyzed_articles()
                
                self.logger.info("📊 Step 3: Preparing daily report...")
                signals = self.signal_generator.get_daily_signals()
                
            else:
                self.logger.info("⚠️ No new articles to analyze")

            # Step 3: Send notifications
            if collection_stats.get('earnings_events', 0) > 0 or signals:
                self.logger.info("📧 Step 4: Sending daily report...")
                report_stats = self.news_collector.get_collection_stats()
                report_stats.update(collection_stats)
                self.notifier.send_daily_report(signals, report_stats)
            else:
                self.logger.info("📧 Step 4: Skipping report - no significant data")

            # Summary
            self.logger.info("=" * 60)
            self.logger.info("✅ Daily collection cycle completed!")
            self.logger.info(f"📊 Summary: {collection_stats.get('earnings_events', 0)} earnings events, "
                           f"{collection_stats.get('articles_stored', 0)} articles stored, "
                           f"{len(signals)} signals generated")
            self.logger.info("=" * 60)
            
            return collection_stats

        except Exception as e:
            self.logger.error(f"❌ Error in daily collection: {e}", exc_info=True)
            try:
                self.notifier.send_error_notification(f"Daily Collection Error: {e}")
            except Exception as notify_error:
                self.logger.error(f"Failed to send error notification: {notify_error}")
            raise

    def run_earnings_scan(self):
        """Run earnings-focused scan for urgent signals"""
        self.logger.info("🎯 Running earnings-focused scan...")
        
        try:
            # Collect new earnings-related news
            collection_stats = self.news_collector.run_earnings_focused_collection()
            
            if collection_stats.get('articles_stored', 0) == 0:
                self.logger.info("⚠️ No new articles found - skipping signal processing")
                return collection_stats

            # Process articles for urgent signals
            self.logger.info("⚙️ Processing articles for urgent signals...")
            self.signal_generator.process_unanalyzed_articles()

            # Check for urgent signals
            urgent_signals = self.signal_generator.get_urgent_signals(
                min_confidence=Config.URGENT_SIGNAL_CONFIDENCE_THRESHOLD
            )
            
            # Send urgent alerts
            urgent_count = 0
            for signal in urgent_signals:
                try:
                    self.notifier.send_urgent_alert(
                        ticker=signal['ticker'],
                        signal_type=signal['signal_type'],
                        confidence=signal['confidence_score'],
                        reasoning=signal['reasoning']
                    )
                    urgent_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to send urgent alert for {signal['ticker']}: {e}")

            if urgent_count > 0:
                self.logger.info(f"🚨 Sent {urgent_count} urgent alerts")
            else:
                self.logger.info("✅ No urgent signals found")

            collection_stats['urgent_signals_sent'] = urgent_count
            return collection_stats

        except Exception as e:
            self.logger.error(f"❌ Error in earnings scan: {e}", exc_info=True)
            raise

    def run_quick_scan(self):
        """Quick scan for urgent signals from existing data"""
        self.logger.info("⚡ Running quick scan...")
        
        try:
            # Process any unanalyzed articles
            self.signal_generator.process_unanalyzed_articles()

            # Check for urgent signals
            urgent_signals = self.signal_generator.get_urgent_signals(
                min_confidence=Config.URGENT_SIGNAL_CONFIDENCE_THRESHOLD
            )

            # Send alerts
            alert_count = 0
            for signal in urgent_signals:
                try:
                    self.notifier.send_urgent_alert(
                        ticker=signal['ticker'],
                        signal_type=signal['signal_type'],
                        confidence=signal['confidence_score'],
                        reasoning=signal['reasoning']
                    )
                    alert_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to send urgent alert: {e}")

            if alert_count > 0:
                self.logger.info(f"🚨 Quick scan: Sent {alert_count} urgent alerts")
            else:
                self.logger.info("✅ Quick scan: No urgent signals")

            return alert_count

        except Exception as e:
            self.logger.error(f"❌ Error in quick scan: {e}", exc_info=True)
            return 0

    def run_scheduler(self):
        """Run the scheduler for continuous operation"""
        self.logger.info("⏰ Starting Earnings News Trader Scheduler")
        self.logger.info("Press Ctrl+C to stop")

        # Schedule daily collection
        schedule.every().day.at(self.DAILY_COLLECTION_TIME).do(self.run_daily_collection)
        
        # Schedule earnings scans
        schedule.every().day.at(self.MARKET_OPEN_SCAN_TIME).do(self.run_earnings_scan)
        schedule.every().day.at(self.MIDDAY_SCAN_TIME).do(self.run_earnings_scan)
        schedule.every().day.at(self.MARKET_CLOSE_SCAN_TIME).do(self.run_earnings_scan)

        # Schedule quick scans during market hours
        for hour in range(self.SCAN_START_HOUR, self.SCAN_END_HOUR):
            schedule.every().day.at(f"{hour:02d}:00").do(self.run_quick_scan)

        self.logger.info("📅 Scheduled tasks:")
        for job in schedule.get_jobs():
            self.logger.info(f"  - {job}")

        try:
            while True:
                schedule.run_pending()
                time.sleep(Config.SCHEDULER_SLEEP_INTERVAL_SECONDS)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"💥 Scheduler crashed: {e}", exc_info=True)
            try:
                self.notifier.send_error_notification(f"Scheduler crashed: {e}")
            except Exception:
                pass
        finally:
            self.logger.info("Scheduler shutting down")

    def get_status(self):
        """Get comprehensive system status"""
        try:
            news_stats = self.news_collector.get_collection_stats()
            signal_stats = self.signal_generator.get_signal_stats()
            db_stats = get_db_stats()

            available_providers = self.sentiment_analyzer.get_available_llm_providers()
            current_provider = self.sentiment_analyzer.get_current_llm_provider()

            return {
                "system_status": "operational",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "news_collection": news_stats,
                "signal_generation": signal_stats,
                "database_stats": db_stats,
                "llm_config": {
                    "available_providers": available_providers,
                    "current_provider": current_provider,
                },
                "scheduled_jobs": len(schedule.jobs) if schedule.jobs else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}", exc_info=True)
            return {
                "system_status": "error",
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def test_earnings_apis(self, days_ahead: int = 7):
        """Test earnings API functionality"""
        self.logger.info(f"🧪 Testing earnings APIs for next {days_ahead} days...")
        
        try:
            calendar_events = self.news_collector.fetch_comprehensive_earnings_calendar(days_ahead)
            
            if calendar_events:
                self.logger.info(f"✅ Found {len(calendar_events)} earnings events")
                for event in calendar_events[:Config.MAX_TEST_EARNINGS_TO_DISPLAY]:
                    self.logger.info(f"  - {event.ticker:6} on {event.date.strftime('%Y-%m-%d')} "
                                   f"({event.time or 'N/A'}) EPS: {event.eps_estimate or 'N/A'}")
                
                if len(calendar_events) > Config.MAX_TEST_EARNINGS_TO_DISPLAY:
                    self.logger.info(f"  ... and {len(calendar_events) - Config.MAX_TEST_EARNINGS_TO_DISPLAY} more")
                    
                return calendar_events
            else:
                self.logger.warning("⚠️ No earnings events found")
                return []

        except Exception as e:
            self.logger.error(f"❌ Earnings API test failed: {e}", exc_info=True)
            raise

    def test_news_collection(self, test_tickers: list = None):
        """Test news collection functionality"""
        self.logger.info("🧪 Testing news collection...")
        
        if test_tickers is None:
            test_tickers = Config.DEFAULT_TEST_TICKERS

        try:
            # Test RSS feeds first
            working_feeds, failed_feeds = self.news_collector.test_rss_feeds()
            
            if not working_feeds:
                self.logger.error("❌ No working RSS feeds found!")
                return None

            self.logger.info(f"✅ {len(working_feeds)} RSS feeds working")
            if failed_feeds:
                self.logger.warning(f"⚠️ {len(failed_feeds)} feeds failed")

            # Test news collection
            self.logger.info(f"Testing news collection for: {', '.join(test_tickers)}")
            articles_map = self.news_collector.collect_targeted_news(test_tickers)
            
            total_articles = sum(len(articles) for articles in articles_map.values())
            
            self.logger.info("📊 News collection results:")
            for ticker, articles in articles_map.items():
                if articles:
                    self.logger.info(f"  {ticker}: {len(articles)} articles")
                    for article in articles[:Config.MAX_TEST_ARTICLES_TO_DISPLAY_PER_TICKER]:
                        self.logger.info(f"    • {article.get('title', 'N/A')[:80]}...")

            self.logger.info(f"Total articles found: {total_articles}")
            return articles_map

        except Exception as e:
            self.logger.error(f"❌ News collection test failed: {e}", exc_info=True)
            raise

def handle_command(trader, command: str):
    """Handle CLI commands"""
    try:
        if command == "run":
            print("🚀 Running daily collection cycle...")
            trader.run_daily_collection()
            print("✅ Daily collection completed")
            
        elif command == "earnings":
            print("🎯 Running earnings scan...")
            stats = trader.run_earnings_scan()
            print(f"✅ Earnings scan completed: {stats}")
            
        elif command == "scan":
            print("⚡ Running quick scan...")
            alerts = trader.run_quick_scan()
            print(f"✅ Quick scan completed: {alerts} alerts sent")
            
        elif command == "schedule":
            trader.run_scheduler()
            
        elif command == "status":
            status = trader.get_status()
            print("\n📊 System Status:")
            print("=" * 50)
            for key, value in status.items():
                print(f"{key}: {value}")
            print("=" * 50)
            
        elif command == "test":
            print("✅ System validation completed at startup")
            
        elif command == "test-earnings":
            print("🧪 Testing earnings APIs...")
            events = trader.test_earnings_apis()
            if events:
                print(f"✅ Found {len(events)} earnings events")
            else:
                print("⚠️ No earnings events found")
                
        elif command == "test-news":
            print("🧪 Testing news collection...")
            articles = trader.test_news_collection()
            if articles:
                total = sum(len(v) for v in articles.values())
                print(f"✅ Collected {total} articles")
            else:
                print("⚠️ News collection failed")
                
        elif command.startswith("switch-llm="):
            provider = command.split("=", 1)[1]
            if trader.sentiment_analyzer.switch_llm_provider(provider):
                print(f"✅ Switched to LLM provider: {provider}")
            else:
                print(f"❌ Failed to switch to provider: {provider}")
                
        else:
            print(f"❓ Unknown command: {command}")
            print_usage()
            
    except Exception as e:
        print(f"❌ Command failed: {e}")
        logging.getLogger(__name__).error(f"Command '{command}' failed: {e}", exc_info=True)

def print_usage():
    """Print usage information"""
    print("""
📋 Earnings News Trader - Usage

Production Commands:
  python main.py schedule      # Start continuous monitoring (production mode)
  python main.py run          # Single collection and analysis cycle
  python main.py earnings     # Earnings-focused scan with urgent alerts
  python main.py scan         # Quick scan for urgent signals

Testing Commands:
  python main.py test         # Validate system setup and configuration
  python main.py test-earnings # Test earnings calendar APIs
  python main.py test-news    # Test news collection from RSS feeds
  python main.py status       # Show detailed system status

Configuration:
  python main.py switch-llm=gemini    # Switch primary LLM provider
  
Examples:
  python main.py test-earnings        # Test if earnings APIs work
  python main.py run                  # Run once to see how it works
  python main.py schedule             # Start for continuous operation

💡 Tip: Start with 'test-earnings' and 'test-news' to verify your setup!
""")

def main():
    """Main application entry point"""
    print("🚀 Earnings News Trader - Starting Up...")
    
    # Setup logging first
    try:
        setup_logging()
    except Exception as e:
        print(f"❌ Failed to setup logging: {e}")
        sys.exit(1)

    logger = logging.getLogger(__name__)
    
    # Initialize trader
    try:
        trader = EarningsNewsTrader()
    except Exception as e:
        logger.critical(f"💥 Failed to initialize trader: {e}")
        print(f"❌ Initialization failed: {e}")
        print("💡 Check logs/earnings_trader.log for details")
        sys.exit(1)

    # Get command
    command = sys.argv[1].lower() if len(sys.argv) > 1 else None

    # Handle special commands that don't need full validation
    if command in ["test-earnings", "test-news", "status"]:
        handle_command(trader, command)
        return

    # For most commands, validate setup first
    logger.info("🔍 Validating system setup...")
    if not trader.validate_setup():
        logger.error("❌ System validation failed")
        print("❌ Setup validation failed - check logs for details")
        print("💡 Common issues:")
        print("   - Missing API keys in .env file")
        print("   - Invalid Gmail credentials")
        print("   - Network connectivity problems")
        sys.exit(1)

    logger.info("✅ System validation completed")

    # Execute command or show usage
    if command:
        handle_command(trader, command)
    else:
        print("No command provided - use 'python main.py schedule' for continuous operation")
        print_usage()

if __name__ == "__main__":
    main()
