import cv2
import threading
import time

class ThreadedCamera:
    """
    Performance Optimization:
    Separates the camera frame reading into a dedicated thread.
    This prevents I/O blocking from slowing down the image processing loop.
    """
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        # Set Resolution to HD (1280x720) for larger window
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # FPS calculation
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        
        # Thread threading
        self.thread = None
        self.status = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        if self.capture.isOpened():
            self.status = True
            _, self.frame = self.capture.read()
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True # Daemon thread exits when main program exits
            self.thread.start()
            return self
        else:
            print("Error: Could not open video source.")
            return None

    def update(self):
        while self.status:
            ret, frame = self.capture.read()
            with self.lock:
                if ret:
                    self.frame = frame
                else:
                    self.status = False
            time.sleep(self.FPS)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.status = False
        if self.thread is not None:
            self.thread.join()
        self.capture.release()
