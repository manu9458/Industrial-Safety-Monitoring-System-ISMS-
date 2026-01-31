import requests
import threading
import cv2
import time

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_alert_time = 0
        self.alert_cooldown = 15  # Seconds between Telegram alerts to avoid spam

    def send_message(self, message):
        """Sends a text message asynchronously."""
        def _task():
            try:
                url = f"{self.base_url}/sendMessage"
                payload = {"chat_id": self.chat_id, "text": message}
                requests.post(url, json=payload, timeout=5)
            except Exception as e:
                print(f"Telegram Error: {e}")
        
        threading.Thread(target=_task, daemon=True).start()
    
    def send_frame(self, frame, caption=None):
        """Sends an OpenCV frame as a photo asynchronously."""
        if time.time() - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = time.time()
        
        # Copy frame to avoid thread conflicts if the frame is modified elsewhere
        frame_copy = frame.copy()

        def _task():
            try:
                # Convert frame to bytes
                is_success, buffer = cv2.imencode(".jpg", frame_copy)
                if not is_success:
                    return
                
                url = f"{self.base_url}/sendPhoto"
                files = {'photo': ('alert.jpg', buffer.tobytes(), 'image/jpeg')}
                data = {'chat_id': self.chat_id}
                if caption:
                    data['caption'] = caption
                
                response = requests.post(url, data=data, files=files, timeout=10)
                # print(f"Telegram Status: {response.status_code}")
            except Exception as e:
                print(f"Telegram Photo Error: {e}")
                
        threading.Thread(target=_task, daemon=True).start()
