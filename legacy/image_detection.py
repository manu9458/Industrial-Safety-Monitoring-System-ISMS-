import cv2
import numpy as np
import time
import datetime
import csv
import os
import threading
from collections import deque

class ThreadedCamera:
    """
    Performance Optimization:
    Separates the camera frame reading into a dedicated thread.
    This prevents I/O blocking from slowing down the image processing loop.
    """
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
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

class SurveillanceSystem:
    def __init__(self):
        # 1. Background Subtraction
        # Using MOG2 (Mixture of Gaussians) for better shadow handling and lighting adaptation
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        
        # 3. Visual Buffer (Trajectory)
        # deque stores the last 64 center points of movement to draw a tail
        self.trajectory_points = deque(maxlen=64)
        
        # Logging Setup
        self.log_file = 'activity_log.csv'
        self.init_logger()
        
        # Logic Variables
        self.consecutive_frames = 0
        self.alarm_trigger_frames = 15  # 4. Confidence Logic threshold
        self.is_alarm_active = False
        self.min_contour_area = 1000

    def init_logger(self):
        """Initializes the CSV log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Motion Magnitude (Area)", "Status"])

    def log_activity(self, area, status):
        """2. Activity Logging: Saves event data to CSV."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, area, status])

    def process_frame(self, frame):
        if frame is None:
            return frame

        # Apply Background Subtraction
        # This returns a binary mask where white = motion, black = background
        fg_mask = self.back_sub.apply(frame)
        
        # Remove shadows (gray pixels) from the mask by simple thresholding
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_found_this_frame = False
        max_area = 0
        center = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                motion_found_this_frame = True
                max_area = max(max_area, area)
                
                # Bounding Box
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate center for trajectory
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 4. Confidence Logic
        if motion_found_this_frame:
            self.consecutive_frames += 1
            if center:
                self.trajectory_points.appendleft(center)
        else:
            self.consecutive_frames = 0
            if len(self.trajectory_points) > 0:
                self.trajectory_points.pop() # Slowly fade tail if no motion

        # Trigger Alarm if confidence threshold met
        status_text = "Status: Idle"
        status_color = (0, 255, 0)
        
        if self.consecutive_frames > self.alarm_trigger_frames:
            self.is_alarm_active = True
            status_text = "Status: ALARM TRIGGERED"
            status_color = (0, 0, 255)
            
            # Log significant events (limit logging frequency in real app, here per frame for demo)
            if self.consecutive_frames % 30 == 0: # Log once every ~second during alarm
                self.log_activity(max_area, "ALARM")
        else:
            self.is_alarm_active = False

        # Draw Trajectory (Visual Buffer)
        for i in range(1, len(self.trajectory_points)):
            if self.trajectory_points[i - 1] is None or self.trajectory_points[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, self.trajectory_points[i - 1], self.trajectory_points[i], (0, 0, 255), thickness)

        # Draw UI
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        return frame, fg_mask

def main():
    # 5. Performance Optimization: Start Threaded Camera
    camera = ThreadedCamera()
    camera_stream = camera.start()
    
    system = SurveillanceSystem()
    
    print("Industrial Monitoring System Started...")
    print("Press 'ESC' or 'q' to exit.")

    while True:
        try:
            frame = camera_stream.get_frame()
            if frame is None:
                continue

            processed_frame, mask = system.process_frame(frame)

            # Show feeds
            cv2.imshow("Live Feed", processed_frame)
            cv2.imshow("Mask (Debug)", mask)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
                
        except AttributeError:
            pass

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()