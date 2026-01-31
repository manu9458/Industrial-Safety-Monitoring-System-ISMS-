import cv2
import threading
import time

class ThreadedCamera:
    """
    Optimized camera reader that captures frames in a separate thread
    to prevent I/O blocking in the main processing loop.
    """
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        self.force_stop = False
        self.frame = None
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.fps_limit = 1/30
        self.thread = None

    def start(self):
        if not self.capture.isOpened():
            return None
            
        # Read the first frame to ensure connection
        ret, self.frame = self.capture.read()
        if not ret:
            return None

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        while not self.force_stop:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                self.force_stop = True
            
            time.sleep(self.fps_limit)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.force_stop = True
        if self.thread:
            self.thread.join()
        self.capture.release()
