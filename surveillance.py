import cv2
import time
import numpy as np
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

        h_img, w_img = frame.shape[:2]
        
        # --- PHASE 0: Draw Forbidden Zone (ROI) ---
        # Define a dynamic ROI (e.g., Right 25% of the screen)
        # In a real app, this would be user-configurable
        roi_points = np.array([
            [int(w_img * 0.75), 0], 
            [w_img, 0], 
            [w_img, h_img], 
            [int(w_img * 0.75), h_img]
        ], np.int32)

        # Draw transparent overlay for the zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_points], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        cv2.polylines(frame, [roi_points], True, (0, 0, 255), 2)
        cv2.putText(frame, "RESTRICTED ZONE", (int(w_img * 0.76), 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.frame_count += 1
        annotated_frame = frame.copy()
        
        # h_img, w_img calculation moved up for ROI logic
        
        # --- PHASE 1: Human Detection ---
        person_results = self.person_model(frame, classes=[0], stream=True, conf=0.5, verbose=False)
        
        persons = [] 
        for r in person_results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                x1, y1, x2, y2 = coords
                
                # Filter out small detections (e.g. hands, arms) that are not full bodies
                # Person must be at least 25% of screen height
                if (y2 - y1) < (h_img * 0.25):
                    continue

                persons.append(coords)
                
                # Visual
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                




        # --- PHASE 3: Safety Helmet Verification ---
        helmets = []
        if self.ppe_active:
            # High confidence to reduce false positives
            ppe_results = self.ppe_model(frame, stream=True, conf=0.7, verbose=False)
            for r in ppe_results:
                for box in r.boxes:
                    cls_name = self.ppe_model.names[int(box.cls[0])]
                    if "hat" in cls_name.lower() or "helmet" in cls_name.lower():
                        helmets.append(list(map(int, box.xyxy[0])))

        # --- PHASE 4: Logic & Alerts ---
        violation_found = False
        roi_violation = False
        
        # 4a. Match Helmets to Persons (Greedy Assignment)
        # Avoids one helmet validating multiple people in a crowd
        matches = [] # (person_idx, helmet_idx, distance)
        
        for p_i, (px1, py1, px2, py2) in enumerate(persons):
            # Check ROI Violation
            feet_point = (int((px1 + px2) / 2), py2)
            if cv2.pointPolygonTest(roi_points, feet_point, False) >= 0:
                roi_violation = True
                cv2.circle(annotated_frame, feet_point, 5, (0, 0, 255), -1)

            # Helmet Matching Prep
            p_center_x = (px1 + px2) // 2
            p_head_y = py1 # Top of bounding box
            
            for h_i, (hx1, hy1, hx2, hy2) in enumerate(helmets):
                 h_center_x = (hx1 + hx2) // 2
                 h_center_y = (hy1 + hy2) // 2
                 
                 # Check if helmet is roughly within the person's head region
                 # Expanded range slightly for stability
                 if (px1 - 20 < h_center_x < px2 + 20) and \
                    (py1 - 60 < h_center_y < py1 + (py2 - py1)//3):
                     
                     # Calculate distance (prefer closer matches)
                     dist = abs(p_center_x - h_center_x) + abs(p_head_y - h_center_y)
                     matches.append((p_i, h_i, dist))
        
        # Sort matches by distance (closest pairs first)
        matches.sort(key=lambda x: x[2])
        
        person_has_helmet = [False] * len(persons)
        used_helmets = set()
        
        for p_i, h_i, dist in matches:
            if not person_has_helmet[p_i] and h_i not in used_helmets:
                person_has_helmet[p_i] = True
                used_helmets.add(h_i)
                
                # Draw Helmet Box (Green)
                hx1, hy1, hx2, hy2 = helmets[h_i]
                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

        # 4b. Draw Person Status
        for i, (px1, py1, px2, py2) in enumerate(persons):
            if person_has_helmet[i]:
                cv2.putText(annotated_frame, "SAFE", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            else:
                violation_found = True
                cv2.putText(annotated_frame, "NO HELMET", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

        # Violation Logic
        
        # Priority: ROI Violation > No Helmet
        if roi_violation:
            self.violation_count = min(30, self.violation_count + 2)
            
        elif not persons:
            self.violation_count = 0
            
        elif violation_found:
            self.violation_count = min(30, self.violation_count + 1)
        else:
            self.violation_count = max(0, self.violation_count - 2)
        
        # Status Text Logic (Support Multiple Messages)
        status_lines = []
        
        if roi_violation:
             status_lines.append(("ALERT: RESTRICTED ZONE", (0, 0, 255)))
             
        if self.violation_count > self.alert_limit:
             status_lines.append(("ALERT: SAFETY VIOLATION", (0, 0, 255)))
             
        if not status_lines:
             status_lines.append(("Status: Monitoring", (0, 255, 0)))

        # Trigger Actions
        if self.violation_count > self.alert_limit or roi_violation:
            violation_type = "Restricted Zone Violation" if roi_violation else "No Helmet Detected"

            # 1. Log to CSV
            if self.frame_count % 60 == 0:
                self.logger.log(len(persons), violation_type)
            
            # 2. TRIGGER GEMINI AI
            trigger_msg = f"High Priority: {violation_type}"
            self.ai.analyze_scene(frame, trigger_reason=trigger_msg)

        # Complex Hazard Check
        if time.time() - self.last_routine_check > 15.0:
            self.ai.analyze_scene(frame, trigger_reason="Routine General Hazard Scan")
            self.last_routine_check = time.time()

        # UI - Draw Multiline Status
        start_y = 40
        for text, color in status_lines:
            cv2.putText(annotated_frame, text, (20, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            start_y += 40
        
        # Cleanup


        return annotated_frame, None

