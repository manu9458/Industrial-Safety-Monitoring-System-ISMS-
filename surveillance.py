import cv2
import time
from logger import ActivityLogger
from ultralytics import YOLO
from ai_assistant import SmartAssistant

class SurveillanceSystem:
    def __init__(self):
        # 1. Person Detection (Human Movement)
        print("Loading Human Detection Model...")
        self.person_model = YOLO("yolov8n.pt") 
        
        # 2. PPE Detection (Safety Helmet)
        try:
            print("Loading PPE Detection Model...")
            self.ppe_model = YOLO("hardhat.pt")
            self.ppe_active = True
        except Exception:
            print("Warning: 'hardhat.pt' missing. Using fallback logic.")
            self.ppe_active = False
        

        
        # 4. AI Assistant (Gemini)
        self.ai = SmartAssistant()
        self.last_routine_check = time.time()
        
        self.logger = ActivityLogger()
        self.frame_count = 0
        
        # Alert Thresholds
        self.violation_count = 0
        self.alert_limit = 10 

    def process_frame(self, frame):
        if frame is None:
            return frame, None

        self.frame_count += 1
        annotated_frame = frame.copy()
        
        # --- PHASE 1: Human Detection ---
        person_results = self.person_model(frame, classes=[0], stream=True, conf=0.5, verbose=False)
        
        persons = [] 
        for r in person_results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                persons.append(coords)
                
                # Visual
                x1, y1, x2, y2 = coords
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                




        # --- PHASE 3: Safety Helmet Verification ---
        helmets = []
        if self.ppe_active:
            ppe_results = self.ppe_model(frame, stream=True, conf=0.5, verbose=False)
            for r in ppe_results:
                for box in r.boxes:
                    cls_name = self.ppe_model.names[int(box.cls[0])]
                    if "hat" in cls_name.lower() or "helmet" in cls_name.lower():
                        helmets.append(list(map(int, box.xyxy[0])))

        # --- PHASE 4: Logic & Alerts ---
        violation_found = False
        
        for px1, py1, px2, py2 in persons:
            has_helmet = False
            head_y_limit = py1 + (py2 - py1) // 3
            
            for hx1, hy1, hx2, hy2 in helmets:
                h_center_x = (hx1 + hx2) // 2
                h_center_y = (hy1 + hy2) // 2
                
                if (px1 < h_center_x < px2) and (py1 - 50 < h_center_y < head_y_limit):
                    has_helmet = True
                    cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                    break
            
            if has_helmet:
                cv2.putText(annotated_frame, "SAFE", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            else:
                # UNSAFE
                violation_found = True
                cv2.putText(annotated_frame, "NO HELMET", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

        # Violation Logic
        # Violation Logic
        if not persons:
            self.violation_count = 0
            
        elif violation_found:
            self.violation_count = min(30, self.violation_count + 1) # Cap at 30 to prevent stuck alerts
        else:
            self.violation_count = max(0, self.violation_count - 2) # Recover faster (decrement by 2)

        status_text = "Status: Monitoring"
        status_color = (0, 255, 0)

        if self.violation_count > self.alert_limit:
            status_text = "ALERT: SAFETY VIOLATION"
            status_color = (0, 0, 255)
            
            # 1. Log to CSV
            if self.frame_count % 60 == 0:
                self.logger.log(len(persons), "No Helmet Detected")
            
            # 2. TRIGGER GEMINI AI (Context-Aware Voice Warning)
            # We send the RAW frame (without boxes) so AI detects better
            self.ai.analyze_scene(frame, trigger_reason="High Priority: Worker detected without helmet")

        # Complex Hazard Check (Routine Scan every 15 seconds)
        # This handles "Smoking", "Fire", "Blocked Path" which YOLO might miss
        if time.time() - self.last_routine_check > 15.0:
            self.ai.analyze_scene(frame, trigger_reason="Routine General Hazard Scan")
            self.last_routine_check = time.time()

        # UI
        cv2.putText(annotated_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Cleanup


        return annotated_frame, None

