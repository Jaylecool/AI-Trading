"""
Alpaca Broker Integration — Live Trading
Wraps the alpaca-py SDK to plug into the existing TradingEngine.

Usage:
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.
    Set ALPACA_PAPER=true  for paper trading (default),
        ALPACA_PAPER=false for live trading.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List

import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import so the rest of the system still runs if alpaca-py isn't installed
# ---------------------------------------------------------------------------
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning(
        "alpaca-py not installed. Run: pip install alpaca-py\n"
        "Live trading will be unavailable until then."
    )


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _is_paper() -> bool:
    return cfg.ALPACA_PAPER


def _check_available():
    if not _ALPACA_AVAILABLE:
        raise RuntimeError(
            "alpaca-py is not installed. Install it with: pip install alpaca-py"
        )
    if not cfg.ALPACA_API_KEY or not cfg.ALPACA_SECRET_KEY:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in your .env file."
        )


# ---------------------------------------------------------------------------
# AlpacaBroker
# ---------------------------------------------------------------------------

class AlpacaBroker:
    """
    Thin wrapper around the Alpaca Trading API.

    Paper mode is used by default so you can test without risking real money.
    Switch to live by setting ALPACA_PAPER=false in .env.
    """

    def __init__(self):
        _check_available()
        self._paper = _is_paper()
        self._client = TradingClient(
            api_key=cfg.ALPACA_API_KEY,
            secret_key=cfg.ALPACA_SECRET_KEY,
            paper=self._paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=cfg.ALPACA_API_KEY,
            secret_key=cfg.ALPACA_SECRET_KEY,
        )
        mode = "PAPER" if self._paper else "LIVE"
        logger.info(f"AlpacaBroker initialised in {mode} mode.")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        """Return key account metrics as a plain dict."""
        acct = self._client.get_account()
        return {
            "account_number": acct.account_number,
            "status": acct.status,
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "last_equity": float(acct.last_equity),
            "pnl_today": float(acct.equity) - float(acct.last_equity),
            "pnl_today_pct": (
                (float(acct.equity) - float(acct.last_equity))
                / float(acct.last_equity)
                * 100
                if float(acct.last_equity) > 0
                else 0.0
            ),
            "paper": self._paper,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """Return all open positions."""
        positions = self._client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "side": p.side,
            })
        return result

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return a single open position, or None if not held."""
        try:
            p = self._client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
                "side": p.side,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_latest_price(self, symbol: str) -> float:
        """Fetch the latest trade price for a symbol."""
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trades = self._data_client.get_stock_latest_trade(req)
        return float(trades[symbol].price)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,           # "buy" or "sell"
        time_in_force: str = "day",
    ) -> Dict:
        """Submit a market order. Returns the order as a dict."""
        _side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        _tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=_side,
            time_in_force=_tif,
        )
        order = self._client.submit_order(req)
        logger.info(f"Market order submitted: {side.upper()} {qty} {symbol} — id={order.id}")
        return self._order_to_dict(order)

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Dict:
        """Submit a limit order. Returns the order as a dict."""
        _side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        _tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=_side,
            limit_price=limit_price,
            time_in_force=_tif,
        )
        order = self._client.submit_order(req)
        logger.info(f"Limit order submitted: {side.upper()} {qty} {symbol} @ {limit_price} — id={order.id}")
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID. Returns True on success."""
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled.")
            return True
        except Exception as exc:
            logger.error(f"Failed to cancel order {order_id}: {exc}")
            return False

    def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict]:
        """Retrieve recent orders."""
        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }
        req = GetOrdersRequest(
            status=status_map.get(status.lower(), QueryOrderStatus.ALL),
            limit=limit,
        )
        orders = self._client.get_orders(req)
        return [self._order_to_dict(o) for o in orders]

    def close_position(self, symbol: str) -> Dict:
        """Market-sell / close the entire position for a symbol."""
        order = self._client.close_position(symbol)
        logger.info(f"Position closed for {symbol}")
        return self._order_to_dict(order)

    def close_all_positions(self) -> List[Dict]:
        """Close all open positions at market price."""
        responses = self._client.close_all_positions(cancel_orders=True)
        results = []
        for r in responses:
            try:
                results.append(self._order_to_dict(r.body))
            except Exception:
                results.append({"raw": str(r)})
        logger.info("All positions closed.")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _order_to_dict(order) -> Dict:
        return {
            "id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "symbol": order.symbol,
            "side": order.side,
            "type": order.type,
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "status": order.status,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        }


# ---------------------------------------------------------------------------
# Module-level singleton — lazily created
# ---------------------------------------------------------------------------
_broker_instance: Optional["AlpacaBroker"] = None


def get_broker() -> "AlpacaBroker":
    """Return the shared AlpacaBroker instance, creating it on first call."""
    global _broker_instance
    if _broker_instance is None:
        _broker_instance = AlpacaBroker()
    return _broker_instance


def is_available() -> bool:
    """True if alpaca-py is installed and API keys are configured."""
    return (
        _ALPACA_AVAILABLE
        and bool(cfg.ALPACA_API_KEY)
        and bool(cfg.ALPACA_SECRET_KEY)
    )
