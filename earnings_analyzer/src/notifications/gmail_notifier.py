import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date
from typing import List, Dict
from config.config import Config
import logging

logger = logging.getLogger(__name__)

class GmailNotifier:
    """Gmail notification system using SMTP"""
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.port = 587  # For starttls
        self.sender_email = Config.GMAIL_EMAIL
        self.password = Config.GMAIL_APP_PASSWORD
        
    def create_daily_report_html(self, signals: List[Dict], stats: Dict = None) -> str:
        """Create HTML email report with fixed statistics"""
        
        # Fix statistics calculation
        if stats is None:
            stats = {}
        
        # Calculate correct statistics from signals
        today_signals = len(signals)
        high_confidence_signals = len([s for s in signals if s.get('confidence_score', 0) >= 0.8])
        long_signals = len([s for s in signals if s.get('signal_type') == 'LONG'])
        short_signals = len([s for s in signals if s.get('signal_type') == 'SHORT'])
        
        # Use actual stats from database if available, otherwise use calculated values
        today_signals_stat = stats.get('today_signals', today_signals)
        high_confidence_stat = stats.get('high_confidence', high_confidence_signals)
        today_articles_stat = stats.get('articles_added_today_utc', stats.get('today_articles', 0))
        long_signals_stat = stats.get('today_long_signals', long_signals)
        short_signals_stat = stats.get('today_short_signals', short_signals)
        
        logger.info(f"📊 Email stats: Signals={today_signals_stat}, High Conf={high_confidence_stat}, "
                   f"Articles={today_articles_stat}, Long/Short={long_signals_stat}/{short_signals_stat}")
        
        # Email styling
        css_style = """
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 20px; text-align: center; border-radius: 10px; }
            .stats { display: flex; justify-content: space-around; margin: 20px 0; flex-wrap: wrap; }
            .stat-box { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; 
                       margin: 5px; min-width: 120px; border: 1px solid #e9ecef; }
            .signals-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .signals-table th, .signals-table td { 
                border: 1px solid #ddd; padding: 12px; text-align: left; 
            }
            .signals-table th { background-color: #f2f2f2; font-weight: bold; }
            .long-signal { background-color: #d4edda; color: #155724; }
            .short-signal { background-color: #f8d7da; color: #721c24; }
            .neutral-signal { background-color: #fff3cd; color: #856404; }
            .confidence-high { font-weight: bold; color: #28a745; }
            .confidence-medium { color: #ffc107; }
            .confidence-low { color: #dc3545; }
            .footer { margin-top: 30px; padding: 20px; background: #f8f9fa; 
                     border-radius: 8px; font-size: 12px; color: #666; }
            .llm-badge { 
                display: inline-block; 
                padding: 2px 6px; 
                border-radius: 12px; 
                font-size: 10px; 
                font-weight: bold; 
                text-transform: uppercase;
                margin-left: 5px;
            }
            .claude { background: #FF6B35; color: white; }
            .openai { background: #00A67E; color: white; }
            .gemini { background: #4285F4; color: white; }
            .grok { background: #1DA1F2; color: white; }
            .none { background: #6c757d; color: white; }
            @media (max-width: 600px) {
                .stats { flex-direction: column; }
                .stat-box { margin: 5px 0; }
                .signals-table { font-size: 12px; }
            }
        </style>
        """
        
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Earnings News Trading Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            {css_style}
        </head>
        <body>
            <div class="header">
                <h1>📊 Daily Earnings Trading Report</h1>
                <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
            </div>
        """
        
        # Add statistics with corrected values
        html_content += f"""
        <div class="stats">
            <div class="stat-box">
                <h3>{today_signals_stat}</h3>
                <p>Today's Signals</p>
            </div>
            <div class="stat-box">
                <h3>{high_confidence_stat}</h3>
                <p>High Confidence</p>
            </div>
            <div class="stat-box">
                <h3>{today_articles_stat}</h3>
                <p>Articles Processed</p>
            </div>
            <div class="stat-box">
                <h3>{long_signals_stat}/{short_signals_stat}</h3>
                <p>Long/Short</p>
            </div>
        </div>
        """
        
        # Show LLM provider usage if available
        provider_usage = stats.get('provider_usage', {})
        if provider_usage:
            provider_info = []
            for provider, count in provider_usage.items():
                if provider and provider != 'None':
                    provider_info.append(f"{provider}: {count}")
            
            if provider_info:
                html_content += f"""
                <div style="text-align: center; margin: 10px 0; font-size: 12px; color: #666;">
                    🤖 AI Providers Used: {', '.join(provider_info)}
                </div>
                """
        
        # Add signals table
        if signals:
            html_content += """
            <h2>🎯 Trading Signals</h2>
            <table class="signals-table">
                <tr>
                    <th>Ticker</th>
                    <th>Signal</th>
                    <th>Confidence</th>
                    <th>Current Price</th>
                    <th>30D Change</th>
                    <th>Position Size</th>
                    <th>Stop/Target</th>
                    <th>Reasoning</th>
                </tr>
            """
            
            for signal in signals:
                stock_ctx = signal.get('stock_context', {})
                
                # Signal styling
                signal_class = f"{signal['signal_type'].lower()}-signal"
                
                # Confidence styling - FIXED: Use actual confidence score
                confidence = signal['confidence_score']
                if confidence >= 0.8:
                    conf_class = "confidence-high"
                elif confidence >= 0.6:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"
                
                # LLM provider badge
                llm_provider = signal.get('llm_provider_used', 'None')
                llm_badge = ""
                if llm_provider and llm_provider != 'None':
                    provider_clean = llm_provider.replace('Provider', '').lower()
                    llm_badge = f'<span class="llm-badge {provider_clean}">{provider_clean}</span>'
                else:
                    llm_badge = f'<span class="llm-badge none">traditional</span>'
                
                # Position size formatting
                position_size = signal.get('suggested_position_size', 0)
                position_str = f"{position_size:.1%}" if position_size else "N/A"
                
                # Stop loss and take profit
                stop_loss = signal.get('stop_loss_price', 0)
                take_profit = signal.get('take_profit_price', 0)
                stop_target = f"${stop_loss:.2f} / ${take_profit:.2f}" if stop_loss and take_profit else "N/A"
                
                # Current price - try multiple sources
                current_price = (stock_ctx.get('current_price') or 
                               signal.get('signal_stock_price') or 
                               signal.get('stock_price', 0))
                price_str = f"${current_price:.2f}" if current_price else "N/A"
                
                # 30D change
                price_change = (stock_ctx.get('30d_change') or 
                              signal.get('price_change_30d', 0))
                change_str = f"{price_change:+.1f}%" if price_change else "N/A"
                
                html_content += f"""
                <tr class="{signal_class}">
                    <td><strong>{signal['ticker']}</strong></td>
                    <td><strong>{signal['signal_type']}</strong>{llm_badge}</td>
                    <td class="{conf_class}"><strong>{confidence:.2f}</strong></td>
                    <td>{price_str}</td>
                    <td>{change_str}</td>
                    <td>{position_str}</td>
                    <td style="font-size: 11px;">{stop_target}</td>
                    <td>{signal['reasoning'][:150]}{'...' if len(signal['reasoning']) > 150 else ''}</td>
                </tr>
                """
            
            html_content += "</table>"
            
            # Add analysis breakdown for first few signals
            if len(signals) > 0:
                html_content += """
                <h3>📈 Analysis Breakdown (Top Signals)</h3>
                <div style="font-size: 12px; color: #666; margin: 10px 0;">
                """
                
                for i, signal in enumerate(signals[:3]):  # Show top 3 signals
                    textblob = signal.get('textblob_score', 0)
                    vader = signal.get('vader_score', 0) 
                    llm = signal.get('llm_score', 0)
                    combined = signal.get('combined_sentiment', 0)
                    
                    html_content += f"""
                    <p><strong>{signal['ticker']}:</strong> 
                    TextBlob: {textblob:.2f} | VADER: {vader:.2f} | LLM: {llm:.2f} | Combined: {combined:.2f}</p>
                    """
                
                html_content += "</div>"
        else:
            html_content += """
            <div style="text-align: center; padding: 40px;">
                <h2>📭 No Trading Signals Today</h2>
                <p>No high-confidence trading signals were generated today. 
                   The system is actively monitoring earnings news and will alert you 
                   when opportunities arise.</p>
                
                <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 8px;">
                    <p><strong>🔍 System Status:</strong> All components operational</p>
                    <p><strong>📰 News Monitoring:</strong> Active</p>
                    <p><strong>🤖 AI Analysis:</strong> Ready</p>
                </div>
            </div>
            """
        
        # Add footer
        html_content += f"""
            <div class="footer">
                <h3>⚠️ Important Disclaimer</h3>
                <p><strong>This is for educational purposes only.</strong> These signals are generated by automated 
                analysis and should not be considered as financial advice. Always conduct your own research 
                and consider your risk tolerance before making any trading decisions.</p>
                
                <p><strong>Risk Management:</strong> Never risk more than you can afford to lose. 
                Use proper position sizing and stop-loss orders. The suggested position sizes and stop-loss levels 
                are calculated automatically but should be adjusted based on your personal risk tolerance.</p>
                
                <p><strong>System Info:</strong> Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} 
                using multi-source news analysis and AI sentiment evaluation.</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _format_market_cap(self, market_cap: float) -> str:
        """Format market cap in readable format"""
        if not market_cap or market_cap == 0:
            return "N/A"
        
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.1f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.1f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.1f}M"
        else:
            return f"${market_cap:.0f}"
    
    def send_daily_report(self, signals: List[Dict], stats: Dict = None):
        """Send daily trading report via Gmail with fixed statistics"""
        try:
            # Log what we're sending for debugging
            signal_count = len(signals)
            logger.info(f"📧 Preparing daily report: {signal_count} signals")
            
            if signals:
                for signal in signals:
                    logger.info(f"  Signal: {signal['ticker']} {signal['signal_type']} "
                               f"(confidence: {signal['confidence_score']:.2f})")
            
            # Create message
            date_str = datetime.now().strftime('%m/%d/%Y')
            
            subject = f"📊 Earnings Trading Report - {signal_count} Signal{'s' if signal_count != 1 else ''} - {date_str}"
            
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = Config.ALERT_EMAIL
            
            # Create HTML content
            html_content = self.create_daily_report_html(signals, stats)
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, Config.ALERT_EMAIL, message.as_string())
            
            logger.info(f"✅ Daily report sent successfully to {Config.ALERT_EMAIL} ({signal_count} signals)")
            
        except Exception as e:
            logger.error(f"❌ Error sending daily report: {e}")
            raise
    
    def send_urgent_alert(self, ticker: str, signal_type: str, confidence: float, reasoning: str):
        """Send urgent trading alert for high-confidence signals"""
        try:
            subject = f"🚨 URGENT: {signal_type} Signal - {ticker} (Confidence: {confidence:.2f})"
            
            html_content = f"""
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
                <div style="background: #ff6b6b; color: white; padding: 20px; text-align: center; border-radius: 10px;">
                    <h1>🚨 URGENT TRADING ALERT</h1>
                    <h2>{ticker} - {signal_type}</h2>
                    <p style="font-size: 18px; font-weight: bold;">Confidence: {confidence:.2f}</p>
                </div>
                
                <div style="padding: 20px; background: #f8f9fa; margin-top: 10px; border-radius: 8px;">
                    <h3>📊 Analysis:</h3>
                    <p>{reasoning}</p>
                    
                    <p><strong>⏰ Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    
                    <div style="background: #fff; padding: 15px; border-left: 4px solid #ff6b6b; margin-top: 15px;">
                        <p><strong>🎯 Next Steps:</strong></p>
                        <ul>
                            <li>Review the analysis and reasoning</li>
                            <li>Check current market conditions</li>
                            <li>Determine appropriate position size</li>
                            <li>Set stop-loss and take-profit levels</li>
                        </ul>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; margin-top: 20px; border-radius: 8px; font-size: 12px; color: #666;">
                    <p><strong>⚠️ Disclaimer:</strong> This is an automated alert for educational purposes only. 
                    Always conduct your own research and consider your risk tolerance before making trading decisions.</p>
                </div>
            </body>
            </html>
            """
            
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = Config.ALERT_EMAIL
            
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, Config.ALERT_EMAIL, message.as_string())
            
            logger.info(f"✅ Urgent alert sent for {ticker} {signal_type} signal (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error sending urgent alert: {e}")
            raise
    
    def send_error_notification(self, error_message: str):
        """Send error notification"""
        try:
            subject = f"🚨 Earnings Trader Error - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background: #dc3545; color: white; padding: 20px; text-align: center; border-radius: 10px;">
                    <h1>🚨 System Error Alert</h1>
                    <p>Earnings News Trader encountered an error</p>
                </div>
                
                <div style="padding: 20px;">
                    <h3>❌ Error Details:</h3>
                    <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #dc3545; font-family: monospace;">
                        {error_message}
                    </div>
                    
                    <p><strong>⏰ Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    
                    <h3>🔧 Recommended Actions:</h3>
                    <ul>
                        <li>Check the application logs in the logs/ directory</li>
                        <li>Verify API key configurations in .env file</li>
                        <li>Ensure internet connectivity</li>
                        <li>Check if any APIs are down or rate-limited</li>
                        <li>Restart the application if needed</li>
                    </ul>
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; margin-top: 20px; font-size: 12px;">
                    <p>This is an automated error notification from your Earnings News Trader system.</p>
                </div>
            </body>
            </html>
            """
            
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = Config.ALERT_EMAIL
            
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, Config.ALERT_EMAIL, message.as_string())
            
            logger.info("✅ Error notification sent successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to send error notification: {e}")
    
    def test_connection(self) -> bool:
        """Test Gmail SMTP connection"""
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.password)
            
            logger.info("Gmail connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Gmail connection test failed: {e}")
            return False
