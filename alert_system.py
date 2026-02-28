"""
Alert System - Define and manage alert rules and event triggers
Supports customizable alert conditions based on price, predictions, risk, and portfolio metrics
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts the system can generate"""
    PRICE_ALERT = "price_alert"  # Price crosses threshold
    PREDICTION_ALERT = "prediction_alert"  # Prediction confidence changes
    STOP_LOSS_ALERT = "stop_loss_alert"  # Stop loss breached
    PORTFOLIO_ALERT = "portfolio_alert"  # Portfolio drawdown, equity change
    RISK_ALERT = "risk_alert"  # Risk metrics exceed limits
    PERFORMANCE_ALERT = "performance_alert"  # Win rate, profit changes


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ComparisonOperator(Enum):
    """Comparison operators for alert conditions"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_OR_EQUAL = ">="
    LESS_OR_EQUAL = "<="
    PERCENT_INCREASE = "increase%"  # Increase by X percent
    PERCENT_DECREASE = "decrease%"  # Decrease by X percent


@dataclass
class AlertRule:
    """
    User-defined alert rule
    
    Example:
    - Alert me if AAPL price drops below $150 (price_alert)
    - Alert me if prediction confidence falls below 70% (prediction_alert)
    - Alert me when portfolio drawdown exceeds 10% (portfolio_alert)
    """
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    alert_type: AlertType = AlertType.PRICE_ALERT
    symbol: Optional[str] = None  # For symbol-specific alerts
    metric_field: str = ""  # e.g., 'price', 'confidence', 'drawdown'
    operator: ComparisonOperator = ComparisonOperator.LESS_THAN
    threshold_value: float = 0.0
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True
    
    # Notifications
    notify_popup: bool = True
    notify_sound: bool = True
    notify_email: bool = False
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_triggered: Optional[str] = None
    trigger_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'alert_type': self.alert_type.value,
            'symbol': self.symbol,
            'metric_field': self.metric_field,
            'operator': self.operator.value,
            'threshold_value': self.threshold_value,
            'severity': self.severity.name,
            'enabled': self.enabled,
            'notify_popup': self.notify_popup,
            'notify_sound': self.notify_sound,
            'notify_email': self.notify_email,
            'created_at': self.created_at,
            'last_triggered': self.last_triggered,
            'trigger_count': self.trigger_count
        }


@dataclass
class AlertEvent:
    """
    An alert event triggered when rule conditions are met
    """
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    alert_type: AlertType = AlertType.PRICE_ALERT
    severity: AlertSeverity = AlertSeverity.MEDIUM
    symbol: Optional[str] = None
    
    # Trigger details
    message: str = ""
    metric_field: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    operator: str = ""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'alert_type': self.alert_type.value,
            'severity': self.severity.name,
            'symbol': self.symbol,
            'message': self.message,
            'metric_field': self.metric_field,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'operator': self.operator,
            'timestamp': self.timestamp,
            'is_acknowledged': self.is_acknowledged
        }


class AlertSystem:
    """
    Manages alert rules, evaluates conditions, and triggers alerts
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}  # alert_id -> AlertEvent
        self.alert_history: List[AlertEvent] = []
        self.callbacks: List[Callable[[AlertEvent], None]] = []
        self.max_history_size = 1000
    
    def create_rule(self, 
                   name: str,
                   alert_type: AlertType,
                   metric_field: str,
                   operator: ComparisonOperator,
                   threshold_value: float,
                   symbol: Optional[str] = None,
                   severity: AlertSeverity = AlertSeverity.MEDIUM) -> AlertRule:
        """
        Create a new alert rule.
        
        Args:
            name: User-friendly rule name
            alert_type: Type of alert
            metric_field: Field to monitor (e.g., 'price', 'confidence', 'drawdown')
            operator: Comparison operator
            threshold_value: Threshold to trigger alert
            symbol: Optional symbol for symbol-specific alerts
            severity: Alert severity level
            
        Returns:
            Created AlertRule
        """
        rule = AlertRule(
            name=name,
            alert_type=alert_type,
            metric_field=metric_field,
            operator=operator,
            threshold_value=threshold_value,
            symbol=symbol,
            severity=severity
        )
        
        self.rules[rule.rule_id] = rule
        logger.info(f"Created alert rule: {name} (ID: {rule.rule_id})")
        
        return rule
    
    def update_rule(self, rule_id: str, **kwargs) -> bool:
        """Update an alert rule."""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        
        # Update allowed fields
        allowed_fields = {
            'name', 'enabled', 'threshold_value', 'severity',
            'notify_popup', 'notify_sound', 'notify_email'
        }
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                setattr(rule, key, value)
        
        logger.info(f"Updated alert rule: {rule_id}")
        return True
    
    def delete_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Deleted alert rule: {rule_id}")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get a specific rule."""
        return self.rules.get(rule_id)
    
    def get_all_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self.rules.values())
    
    def get_rules_for_symbol(self, symbol: str) -> List[AlertRule]:
        """Get all rules for a specific symbol."""
        return [r for r in self.rules.values() if r.symbol == symbol or r.symbol is None]
    
    def subscribe_to_alerts(self, callback: Callable[[AlertEvent], None]):
        """Subscribe to alert events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unsubscribe_from_alerts(self, callback: Callable[[AlertEvent], None]):
        """Unsubscribe from alert events."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def evaluate(self, symbol: str, data: Dict[str, Any]) -> List[AlertEvent]:
        """
        Evaluate all rules against provided data and trigger alerts if conditions met.
        
        Args:
            symbol: Symbol being evaluated
            data: Dictionary with metric values (e.g., {'price': 150.5, 'confidence': 0.75})
            
        Returns:
            List of triggered AlertEvents
        """
        triggered_alerts = []
        
        for rule in self.rules.values():
            # Skip disabled rules
            if not rule.enabled:
                continue
            
            # Skip if rule is for different symbol
            if rule.symbol and rule.symbol != symbol:
                continue
            
            # Check if metric is in data
            if rule.metric_field not in data:
                continue
            
            # Evaluate condition
            current_value = data[rule.metric_field]
            threshold = rule.threshold_value
            operator = rule.operator
            
            should_trigger = self._evaluate_condition(
                current_value, threshold, operator
            )
            
            if should_trigger:
                alert = self._create_alert(rule, symbol, current_value, threshold)
                triggered_alerts.append(alert)
                
                # Update rule metadata
                rule.last_triggered = datetime.now().isoformat()
                rule.trigger_count += 1
                
                # Store and notify
                self._register_alert(alert)
        
        return triggered_alerts
    
    def _evaluate_condition(self, current: float, threshold: float, 
                          operator: ComparisonOperator) -> bool:
        """Evaluate if condition is met."""
        if operator == ComparisonOperator.EQUALS:
            return current == threshold
        elif operator == ComparisonOperator.NOT_EQUALS:
            return current != threshold
        elif operator == ComparisonOperator.GREATER_THAN:
            return current > threshold
        elif operator == ComparisonOperator.LESS_THAN:
            return current < threshold
        elif operator == ComparisonOperator.GREATER_OR_EQUAL:
            return current >= threshold
        elif operator == ComparisonOperator.LESS_OR_EQUAL:
            return current <= threshold
        elif operator == ComparisonOperator.PERCENT_INCREASE:
            # Assuming threshold is baseline and current is new value
            return current > threshold * (1 + threshold * 0.01)  # 1% increase
        elif operator == ComparisonOperator.PERCENT_DECREASE:
            return current < threshold * (1 - threshold * 0.01)  # 1% decrease
        else:
            return False
    
    def _create_alert(self, rule: AlertRule, symbol: str, 
                     current_value: float, threshold_value: float) -> AlertEvent:
        """Create an alert event from a triggered rule."""
        # Generate compelling message
        message = self._generate_alert_message(
            rule, symbol, current_value, threshold_value
        )
        
        alert = AlertEvent(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            alert_type=rule.alert_type,
            severity=rule.severity,
            symbol=symbol,
            message=message,
            metric_field=rule.metric_field,
            current_value=current_value,
            threshold_value=threshold_value,
            operator=rule.operator.value
        )
        
        return alert
    
    def _generate_alert_message(self, rule: AlertRule, symbol: str,
                               current: float, threshold: float) -> str:
        """Generate human-readable alert message."""
        symbol_str = f"{symbol} " if symbol else ""
        
        if rule.alert_type == AlertType.PRICE_ALERT:
            return f"Price Alert: {symbol_str}is now ${current:.2f} ({rule.operator.value} ${threshold:.2f})"
        elif rule.alert_type == AlertType.STOP_LOSS_ALERT:
            return f"Stop Loss Triggered: {symbol_str}hit stop level at ${current:.2f}"
        elif rule.alert_type == AlertType.PREDICTION_ALERT:
            return f"Prediction Alert: Confidence is {current:.1%} ({rule.operator.value} {threshold:.1%})"
        elif rule.alert_type == AlertType.PORTFOLIO_ALERT:
            return f"Portfolio Alert: {rule.metric_field} is now {current:.2f}% ({rule.operator.value} {threshold:.2f}%)"
        elif rule.alert_type == AlertType.RISK_ALERT:
            return f"Risk Alert: {rule.metric_field} reached {current:.2f} ({rule.operator.value} {threshold:.2f})"
        elif rule.alert_type == AlertType.PERFORMANCE_ALERT:
            return f"Performance Alert: {rule.metric_field} changed to {current:.2f}% ({rule.operator.value} {threshold:.2f}%)"
        else:
            return f"Alert: {rule.name} triggered (current: {current}, threshold: {threshold})"
    
    def _register_alert(self, alert: AlertEvent):
        """Register and broadcast an alert."""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Trim history if too large
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # Notify all subscribers
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].is_acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Remove an active alert."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert dismissed: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active (unacknowledged) alerts."""
        return [a for a in self.active_alerts.values() if not a.is_acknowledged]
    
    def get_alert_history(self, limit: int = 100) -> List[AlertEvent]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[AlertEvent]:
        """Get all alerts of specific severity."""
        return [a for a in self.active_alerts.values() if a.severity == severity]
    
    def get_critical_alerts(self) -> List[AlertEvent]:
        """Get all critical severity alerts."""
        return self.get_alerts_by_severity(AlertSeverity.CRITICAL)
    
    def export_rules(self) -> str:
        """Export all rules as JSON."""
        rules_data = [r.to_dict() for r in self.rules.values()]
        return json.dumps(rules_data, indent=2)
    
    def import_rules(self, rules_json: str):
        """Import rules from JSON."""
        try:
            rules_data = json.loads(rules_json)
            for rule_data in rules_data:
                # Reconstruct alert rule from JSON
                rule = AlertRule(
                    rule_id=rule_data.get('rule_id', str(uuid.uuid4())),
                    name=rule_data['name'],
                    alert_type=AlertType(rule_data['alert_type']),
                    symbol=rule_data.get('symbol'),
                    metric_field=rule_data['metric_field'],
                    operator=ComparisonOperator(rule_data['operator']),
                    threshold_value=rule_data['threshold_value'],
                    severity=AlertSeverity[rule_data['severity']],
                    enabled=rule_data['enabled'],
                    notify_popup=rule_data['notify_popup'],
                    notify_sound=rule_data['notify_sound'],
                    notify_email=rule_data['notify_email'],
                    created_at=rule_data['created_at'],
                    last_triggered=rule_data.get('last_triggered'),
                    trigger_count=rule_data['trigger_count']
                )
                self.rules[rule.rule_id] = rule
            logger.info(f"Imported {len(rules_data)} alert rules")
            return True
        except Exception as e:
            logger.error(f"Error importing rules: {e}")
            return False


# Global alert system instance
_alert_system = None


def get_alert_system() -> AlertSystem:
    """Get or create the global alert system."""
    global _alert_system
    
    if _alert_system is None:
        _alert_system = AlertSystem()
    
    return _alert_system


if __name__ == "__main__":
    # Demo usage
    system = get_alert_system()
    
    # Create some alert rules
    rule1 = system.create_rule(
        name="Apple stock drops below $150",
        alert_type=AlertType.PRICE_ALERT,
        metric_field='price',
        operator=ComparisonOperator.LESS_THAN,
        threshold_value=150.0,
        symbol='AAPL',
        severity=AlertSeverity.HIGH
    )
    
    rule2 = system.create_rule(
        name="Portfolio loses more than 10%",
        alert_type=AlertType.PORTFOLIO_ALERT,
        metric_field='drawdown',
        operator=ComparisonOperator.GREATER_THAN,
        threshold_value=10.0,
        severity=AlertSeverity.CRITICAL
    )
    
    # Subscribe to alerts
    def on_alert(alert):
        print(f"[{alert.severity.name}] {alert.rule_name}: {alert.message}")
    
    system.subscribe_to_alerts(on_alert)
    
    # Trigger alerts by evaluating data
    print("Testing evaluation...")
    alerts = system.evaluate('AAPL', {'price': 145.5})
    print(f"Triggered {len(alerts)} alert(s)")
    
    alerts = system.evaluate('PORTFOLIO', {'drawdown': 15.0})
    print(f"Triggered {len(alerts)} alert(s)")
