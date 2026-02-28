"""
Streaming Data Service - Real-time market data and price updates
Handles WebSocket connections, API streaming, and data cache management
"""

import json
import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Supported data sources"""
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    SIMULATION = "simulation"  # For testing


@dataclass
class PriceUpdate:
    """Real-time price update"""
    symbol: str
    timestamp: str
    price: float
    bid: float
    ask: float
    volume: int
    change_percent: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DataStreamEvent:
    """Base event from data stream"""
    event_type: str  # 'price_update', 'alert_trigger', 'stream_status'
    timestamp: str
    data: Dict
    
    def to_dict(self):
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'data': self.data
        }


class StreamingDataService:
    """
    Manages real-time market data streaming with multiple source support.
    Simulates WebSocket-like behavior for testing and live feeds for production.
    """
    
    def __init__(self, data_source: DataSourceType = DataSourceType.SIMULATION):
        self.data_source = data_source
        self.subscribed_symbols: Dict[str, List[Callable]] = {}
        self.price_cache: Dict[str, PriceUpdate] = {}
        self.is_running = False
        self.update_frequency = 2  # seconds between updates
        self.stream_thread = None
        self.event_queue = queue.Queue()
        
    def subscribe(self, symbol: str, callback: Callable[[PriceUpdate], None]):
        """
        Subscribe to price updates for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            callback: Function called with PriceUpdate when price changes
        """
        if symbol not in self.subscribed_symbols:
            self.subscribed_symbols[symbol] = []
        
        self.subscribed_symbols[symbol].append(callback)
        logger.info(f"Subscribed to {symbol} (now {len(self.subscribed_symbols[symbol])} subscribers)")
    
    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from price updates."""
        if symbol in self.subscribed_symbols:
            try:
                self.subscribed_symbols[symbol].remove(callback)
                logger.info(f"Unsubscribed from {symbol}")
            except ValueError:
                pass
    
    def start(self):
        """Start the streaming service."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        logger.info("Streaming service started")
    
    def stop(self):
        """Stop the streaming service."""
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        logger.info("Streaming service stopped")
    
    def _stream_loop(self):
        """Main streaming loop - runs in background thread."""
        while self.is_running:
            try:
                # Get latest prices from configured source
                updates = self._fetch_latest_prices()
                
                # Process each update and notify subscribers
                for symbol, update in updates.items():
                    self.price_cache[symbol] = update
                    
                    # Notify all subscribers for this symbol
                    if symbol in self.subscribed_symbols:
                        for callback in self.subscribed_symbols[symbol]:
                            try:
                                callback(update)
                            except Exception as e:
                                logger.error(f"Error in callback for {symbol}: {e}")
                
                # Queue events for dashboard
                if updates:
                    self.event_queue.put(
                        DataStreamEvent(
                            event_type='price_update_batch',
                            timestamp=datetime.now().isoformat(),
                            data={s: u.to_dict() for s, u in updates.items()}
                        )
                    )
                
                # Sleep before next update
                time.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in stream loop: {e}")
                time.sleep(1)
    
    def _fetch_latest_prices(self) -> Dict[str, PriceUpdate]:
        """
        Fetch latest prices from configured data source.
        Returns dict of symbol -> PriceUpdate
        """
        if self.data_source == DataSourceType.YAHOO_FINANCE:
            return self._fetch_yahoo_finance()
        elif self.data_source == DataSourceType.SIMULATION:
            return self._fetch_simulated_prices()
        else:
            return {}
    
    def _fetch_yahoo_finance(self) -> Dict[str, PriceUpdate]:
        """Fetch prices from Yahoo Finance API (requires yfinance package)."""
        try:
            import yfinance as yf
            
            updates = {}
            if not self.subscribed_symbols:
                return updates
            
            for symbol in self.subscribed_symbols.keys():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d')
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        current_price = latest['Close']
                        
                        # Calculate change
                        if len(data) > 1:
                            previous = data.iloc[-2]
                            prev_price = previous['Close']
                            change_percent = ((current_price - prev_price) / prev_price) * 100
                        else:
                            change_percent = 0
                        
                        # Get current bid/ask (approximate)
                        bid = current_price * 0.999
                        ask = current_price * 1.001
                        
                        updates[symbol] = PriceUpdate(
                            symbol=symbol,
                            timestamp=datetime.now().isoformat(),
                            price=float(current_price),
                            bid=float(bid),
                            ask=float(ask),
                            volume=int(latest['Volume']),
                            change_percent=float(change_percent)
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} from Yahoo Finance: {e}")
            
            return updates
        except ImportError:
            logger.warning("yfinance not installed, falling back to simulation")
            return self._fetch_simulated_prices()
    
    def _fetch_simulated_prices(self) -> Dict[str, PriceUpdate]:
        """Generate simulated price updates for testing (with realistic variations)."""
        import random
        
        updates = {}
        now = datetime.now()
        
        for symbol in self.subscribed_symbols.keys():
            # Get or initialize base price
            if symbol in self.price_cache:
                last_price = self.price_cache[symbol].price
            else:
                # Default prices for common symbols
                base_prices = {
                    'AAPL': 180.50, 'MSFT': 420.30, 'GOOGL': 140.25,
                    'AMZN': 170.80, 'TSLA': 210.45
                }
                last_price = base_prices.get(symbol, 100.0)
            
            # Small random change (0.1% to 0.5%)
            change_percent = random.uniform(-0.5, 0.5)
            new_price = last_price * (1 + change_percent / 100)
            
            updates[symbol] = PriceUpdate(
                symbol=symbol,
                timestamp=now.isoformat(),
                price=round(new_price, 2),
                bid=round(new_price * 0.999, 2),
                ask=round(new_price * 1.001, 2),
                volume=random.randint(50000, 5000000),
                change_percent=round(change_percent, 2)
            )
        
        return updates
    
    def get_latest_price(self, symbol: str) -> Optional[PriceUpdate]:
        """Get the latest cached price for a symbol."""
        return self.price_cache.get(symbol)
    
    def get_all_prices(self) -> Dict[str, PriceUpdate]:
        """Get all cached prices."""
        return self.price_cache.copy()
    
    def get_next_event(self, timeout: float = None) -> Optional[DataStreamEvent]:
        """Get next streaming event from queue."""
        try:
            return self.event_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def set_update_frequency(self, seconds: int):
        """Set frequency of price updates (seconds)."""
        self.update_frequency = max(1, seconds)
        logger.info(f"Update frequency set to {seconds} seconds")


class StreamingConnection:
    """
    Simulates a WebSocket-like connection for dashboard integration.
    Provides async-like behavior with threading for real event streaming.
    """
    
    def __init__(self, streaming_service: StreamingDataService):
        self.service = streaming_service
        self.connected = False
        self.client_id = f"client_{int(time.time() * 1000)}"
    
    def connect(self):
        """Establish streaming connection."""
        self.service.start()
        self.connected = True
        logger.info(f"Connection {self.client_id} established")
    
    def disconnect(self):
        """Close streaming connection."""
        self.connected = False
        logger.info(f"Connection {self.client_id} disconnected")
    
    def send_subscription(self, symbols: List[str], callback: Callable):
        """Subscribe to multiple symbols with callback."""
        for symbol in symbols:
            self.service.subscribe(symbol, callback)
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self.connected


# Global streaming service instance
_streaming_service = None


def get_streaming_service(data_source: DataSourceType = DataSourceType.SIMULATION) -> StreamingDataService:
    """Get or create the global streaming service."""
    global _streaming_service
    
    if _streaming_service is None:
        _streaming_service = StreamingDataService(data_source=data_source)
    
    return _streaming_service


def reset_streaming_service():
    """Reset the global streaming service (for testing)."""
    global _streaming_service
    
    if _streaming_service:
        _streaming_service.stop()
    
    _streaming_service = None


if __name__ == "__main__":
    # Demo usage
    service = get_streaming_service(DataSourceType.SIMULATION)
    
    # Subscribe to symbols
    def on_price_update(update: PriceUpdate):
        print(f"[{update.timestamp}] {update.symbol}: ${update.price} ({update.change_percent:+.2f}%)")
    
    service.subscribe('AAPL', on_price_update)
    service.subscribe('MSFT', on_price_update)
    
    # Start streaming
    service.start()
    
    try:
        print("Streaming for 30 seconds...")
        time.sleep(30)
    finally:
        service.stop()
        print("Done!")
