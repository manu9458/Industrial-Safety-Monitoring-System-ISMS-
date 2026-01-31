import requests
import threading
import time
import cv2

class TelegramService:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_alert_time = 0
        self.cooldown = 15

    def send_alert(self, message):
        """Sends a text message notification."""
        threading.Thread(target=self._send_text_task, args=(message,), daemon=True).start()

    def send_snapshot(self, frame, caption=None):
        """Sends a visual snapshot of the event."""
        if time.time() - self.last_alert_time < self.cooldown:
            return

        self.last_alert_time = time.time()
        threading.Thread(target=self._send_photo_task, args=(frame.copy(), caption), daemon=True).start()

    def _send_text_task(self, message):
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            pass # Fail silently for network issues to avoid clutter

    def _send_photo_task(self, frame, caption):
        try:
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                return
            
            url = f"{self.base_url}/sendPhoto"
            files = {'photo': ('alert.jpg', buffer.tobytes(), 'image/jpeg')}
            data = {'chat_id': self.chat_id}
            if caption:
                data['caption'] = caption
            
            requests.post(url, data=data, files=files, timeout=10)
        except Exception:
            pass
