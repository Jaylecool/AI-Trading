"""
Notification Service - Handle alert notifications (pop-ups, sounds, emails, etc.)
Manages notification preferences and delivery mechanisms
"""

import json
import smtplib
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification delivery channels"""
    POPUP = "popup"  # In-app pop-up/toast
    SOUND = "sound"  # Audio alert
    BROWSER_PUSH = "browser_push"  # Browser push notification
    EMAIL = "email"  # Email notification
    DASHBOARD = "dashboard"  # Dashboard notification banner
    LOG = "log"  # Internal logging


@dataclass
class NotificationPreference:
    """User notification preferences"""
    user_id: str = "default"
    
    # Channel enable/disable
    enable_popup: bool = True
    enable_sound: bool = True
    enable_browser_push: bool = False
    enable_email: bool = False
    enable_dashboard: bool = True
    
    # Email settings
    email_address: Optional[str] = None
    
    # Sound settings
    sound_volume: float = 0.7  # 0.0 to 1.0
    sound_file: str = "alert.mp3"  # Default sound
    
    # Sound by severity
    critical_sound: str = "critical.mp3"
    high_sound: str = "high.mp3"
    medium_sound: str = "medium.mp3"
    low_sound: str = "low.mp3"
    
    # Quiet hours
    quiet_hours_enabled: bool = False
    quiet_hours_start: str = "22:00"  # HH:MM format
    quiet_hours_end: str = "08:00"
    
    # Notification grouping
    group_similar_alerts: bool = True
    grouping_window_minutes: int = 5  # Group alerts within 5 minutes
    
    # Created/updated timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Notification:
    """A single notification to be delivered"""
    notification_id: str = ""
    alert_id: str = ""
    rule_id: str = ""
    
    title: str = ""
    message: str = ""
    severity: str = "MEDIUM"
    
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.POPUP])
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_sent: bool = False
    delivery_times: Dict[str, str] = field(default_factory=dict)  # channel -> timestamp
    
    def to_dict(self) -> Dict:
        return {
            'notification_id': self.notification_id,
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity,
            'channels': [c.value for c in self.channels],
            'timestamp': self.timestamp,
            'is_sent': self.is_sent,
            'delivery_times': self.delivery_times
        }


class NotificationService:
    """
    Manages notification delivery across multiple channels.
    Handles preferences, scheduling, and delivery tracking.
    """
    
    def __init__(self):
        self.preferences: Dict[str, NotificationPreference] = {}
        self.notifications: Dict[str, Notification] = {}
        self.notification_history: List[Notification] = []
        self.callbacks: List[Callable[[Notification], None]] = []
        self.max_history_size = 1000
        
        # Email service settings
        self.smtp_configured = False
        self.smtp_server = None
        self.smtp_port = None
        self.smtp_sender = None
        self.smtp_password = None
    
    def set_user_preferences(self, user_id: str, 
                           preferences: NotificationPreference) -> bool:
        """Set notification preferences for a user."""
        self.preferences[user_id] = preferences
        preferences.updated_at = datetime.now().isoformat()
        logger.info(f"Updated preferences for user: {user_id}")
        return True
    
    def get_user_preferences(self, user_id: str = "default") -> NotificationPreference:
        """Get notification preferences for a user."""
        if user_id not in self.preferences:
            self.preferences[user_id] = NotificationPreference(user_id=user_id)
        return self.preferences[user_id]
    
    def update_preference(self, user_id: str, **kwargs) -> bool:
        """Update specific preference fields."""
        prefs = self.get_user_preferences(user_id)
        
        allowed_fields = {
            'enable_popup', 'enable_sound', 'enable_browser_push',
            'enable_email', 'enable_dashboard', 'email_address',
            'sound_volume', 'sound_file', 'quiet_hours_enabled',
            'quiet_hours_start', 'quiet_hours_end', 'group_similar_alerts'
        }
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                setattr(prefs, key, value)
        
        prefs.updated_at = datetime.now().isoformat()
        return True
    
    def configure_email(self, smtp_server: str, port: int, 
                       sender_email: str, password: str) -> bool:
        """Configure SMTP for email notifications."""
        try:
            self.smtp_server = smtp_server
            self.smtp_port = port
            self.smtp_sender = sender_email
            self.smtp_password = password
            self.smtp_configured = True
            logger.info(f"Configured SMTP: {smtp_server}:{port}")
            return True
        except Exception as e:
            logger.error(f"Error configuring email: {e}")
            return False
    
    def subscribe_to_notifications(self, callback: Callable[[Notification], None]):
        """Subscribe to notification events."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unsubscribe_from_notifications(self, callback: Callable[[Notification], None]):
        """Unsubscribe from notification events."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def send_notification(self, notification: Notification, 
                         user_id: str = "default") -> bool:
        """
        Send a notification through configured channels.
        
        Args:
            notification: Notification object to send
            user_id: Target user ID
            
        Returns:
            True if sent successfully
        """
        prefs = self.get_user_preferences(user_id)
        
        # Check quiet hours
        if self._in_quiet_hours(prefs):
            logger.info(f"Notification suppressed during quiet hours: {notification.notification_id}")
            return False
        
        # Determine active channels based on preferences
        active_channels = self._get_active_channels(notification.channels, prefs)
        
        if not active_channels:
            logger.warning(f"No active channels for notification: {notification.notification_id}")
            return False
        
        success = True
        for channel in active_channels:
            try:
                if channel == NotificationChannel.POPUP:
                    self._send_popup(notification)
                elif channel == NotificationChannel.SOUND:
                    self._send_sound(notification, prefs)
                elif channel == NotificationChannel.BROWSER_PUSH:
                    self._send_browser_push(notification)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email(notification, prefs)
                elif channel == NotificationChannel.DASHBOARD:
                    self._send_dashboard(notification)
                elif channel == NotificationChannel.LOG:
                    self._send_log(notification)
                
                # Record delivery time
                notification.delivery_times[channel.value] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Error sending via {channel.value}: {e}")
                success = False
        
        # Mark as sent and store
        notification.is_sent = True
        self.notifications[notification.notification_id] = notification
        self.notification_history.append(notification)
        
        # Trim history if too large
        if len(self.notification_history) > self.max_history_size:
            self.notification_history = self.notification_history[-self.max_history_size:]
        
        # Notify subscribers
        for callback in self.callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
        
        logger.info(f"Notification sent: {notification.notification_id} via {[c.value for c in active_channels]}")
        return success
    
    def _get_active_channels(self, requested_channels: List[NotificationChannel],
                            prefs: NotificationPreference) -> List[NotificationChannel]:
        """Filter channels based on user preferences."""
        active = []
        
        for channel in requested_channels:
            if channel == NotificationChannel.POPUP and prefs.enable_popup:
                active.append(channel)
            elif channel == NotificationChannel.SOUND and prefs.enable_sound:
                active.append(channel)
            elif channel == NotificationChannel.BROWSER_PUSH and prefs.enable_browser_push:
                active.append(channel)
            elif channel == NotificationChannel.EMAIL and prefs.enable_email:
                active.append(channel)
            elif channel == NotificationChannel.DASHBOARD and prefs.enable_dashboard:
                active.append(channel)
            elif channel == NotificationChannel.LOG:
                active.append(channel)  # Always log
        
        return active
    
    def _in_quiet_hours(self, prefs: NotificationPreference) -> bool:
        """Check if current time is within quiet hours."""
        if not prefs.quiet_hours_enabled:
            return False
        
        now = datetime.now().time()
        start = datetime.strptime(prefs.quiet_hours_start, "%H:%M").time()
        end = datetime.strptime(prefs.quiet_hours_end, "%H:%M").time()
        
        if start <= end:
            return start <= now <= end
        else:  # Quiet hours span midnight
            return now >= start or now <= end
    
    def _send_popup(self, notification: Notification):
        """Queue notification for popup display."""
        logger.info(f"POPUP: {notification.title} - {notification.message}")
    
    def _send_sound(self, notification: Notification, 
                   prefs: NotificationPreference):
        """Play sound notification."""
        # Select sound based on severity
        sound_file = {
            'CRITICAL': prefs.critical_sound,
            'HIGH': prefs.high_sound,
            'MEDIUM': prefs.medium_sound,
            'LOW': prefs.low_sound
        }.get(notification.severity, prefs.sound_file)
        
        logger.info(f"SOUND: Playing {sound_file} (volume: {prefs.sound_volume})")
        
        # In production, would use: playsound, pygame, or similar
        # Example: playsound.playsound(sound_file)
    
    def _send_browser_push(self, notification: Notification):
        """Send browser push notification."""
        logger.info(f"BROWSER_PUSH: {notification.title}")
        # In production, would send to browser push service
    
    def _send_email(self, notification: Notification, 
                   prefs: NotificationPreference):
        """Send email notification."""
        if not self.smtp_configured or not prefs.email_address:
            logger.warning("Email not configured or no email address set")
            return False
        
        try:
            # Construct email
            subject = f"[{notification.severity}] {notification.title}"
            body = f"{notification.message}\n\nTimestamp: {notification.timestamp}"
            
            # In production, would send via SMTP
            # server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            # server.login(self.smtp_sender, self.smtp_password)
            # server.send_message(...)
            
            logger.info(f"EMAIL: Sent to {prefs.email_address} - {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _send_dashboard(self, notification: Notification):
        """Queue notification for dashboard display."""
        logger.info(f"DASHBOARD: {notification.title}")
    
    def _send_log(self, notification: Notification):
        """Log notification."""
        logger.info(f"[{notification.severity}] {notification.title}: {notification.message}")
    
    def get_active_notifications(self) -> List[Notification]:
        """Get all unsent or pending notifications."""
        return [n for n in self.notifications.values() if not n.is_sent]
    
    def get_notification_history(self, limit: int = 100) -> List[Notification]:
        """Get recent notification history."""
        return self.notification_history[-limit:]
    
    def clear_old_notifications(self, minutes: int = 60):
        """Clear notifications older than specified minutes."""
        cutoff = datetime.now().timestamp()
        cutoff -= minutes * 60
        
        to_remove = []
        for notif_id, notif in self.notifications.items():
            notif_time = datetime.fromisoformat(notif.timestamp).timestamp()
            if notif_time < cutoff:
                to_remove.append(notif_id)
        
        for notif_id in to_remove:
            del self.notifications[notif_id]
        
        logger.info(f"Cleared {len(to_remove)} old notifications")
    
    def export_preferences(self, user_id: str = "default") -> str:
        """Export user preferences as JSON."""
        prefs = self.get_user_preferences(user_id)
        return json.dumps(prefs.to_dict(), indent=2)
    
    def import_preferences(self, prefs_json: str, user_id: str = "default") -> bool:
        """Import user preferences from JSON."""
        try:
            prefs_data = json.loads(prefs_json)
            prefs = NotificationPreference(
                user_id=user_id,
                enable_popup=prefs_data.get('enable_popup', True),
                enable_sound=prefs_data.get('enable_sound', True),
                enable_browser_push=prefs_data.get('enable_browser_push', False),
                enable_email=prefs_data.get('enable_email', False),
                enable_dashboard=prefs_data.get('enable_dashboard', True),
                email_address=prefs_data.get('email_address'),
                sound_volume=prefs_data.get('sound_volume', 0.7),
                quiet_hours_enabled=prefs_data.get('quiet_hours_enabled', False),
                quiet_hours_start=prefs_data.get('quiet_hours_start', '22:00'),
                quiet_hours_end=prefs_data.get('quiet_hours_end', '08:00'),
                group_similar_alerts=prefs_data.get('group_similar_alerts', True)
            )
            self.set_user_preferences(user_id, prefs)
            return True
        except Exception as e:
            logger.error(f"Error importing preferences: {e}")
            return False


# Global notification service instance
_notification_service = None


def get_notification_service() -> NotificationService:
    """Get or create the global notification service."""
    global _notification_service
    
    if _notification_service is None:
        _notification_service = NotificationService()
    
    return _notification_service


if __name__ == "__main__":
    # Demo usage
    service = get_notification_service()
    
    # Set user preferences
    prefs = NotificationPreference(
        user_id="user1",
        enable_popup=True,
        enable_sound=True,
        enable_email=False,
        sound_volume=0.8
    )
    service.set_user_preferences("user1", prefs)
    
    # Create and send notification
    notif = Notification(
        notification_id="notif_001",
        alert_id="alert_001",
        title="Price Alert",
        message="AAPL dropped below $150",
        severity="HIGH",
        channels=[NotificationChannel.POPUP, NotificationChannel.SOUND]
    )
    
    service.send_notification(notif, user_id="user1")
    
    print("Notification sent!")
