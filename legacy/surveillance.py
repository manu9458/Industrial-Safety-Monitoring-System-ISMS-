import cv2
import time
import numpy as np
from logger import ActivityLogger
from ultralytics import YOLO
from ai_assistant import SmartAssistant
from telegram_notifier import TelegramNotifier

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
        
        # 5. Telegram Bot
        self.telegram = TelegramNotifier(
            token="8503256297:AAEgqw4YG5UbZbvR4DJSU_EGbxcgvMhjXJo",
            chat_id="7718614749"
        )
        
        self.logger = ActivityLogger()
        self.frame_count = 0
        
        # Alert Thresholds
        self.violation_count = 0
        self.alert_limit = 12 # Increased for stability against motion blur

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
        person_confs = []
        for r in person_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                person_confs.append(conf)
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
        helmet_confs = []
        yolo_helmet_count = 0
        if self.ppe_active:
            # Balanced confidence for speed and accuracy
            ppe_results = self.ppe_model(frame, stream=True, conf=0.6, verbose=False)
            for r in ppe_results:
                for box in r.boxes:
                    cls_name = self.ppe_model.names[int(box.cls[0])]
                    if "hat" in cls_name.lower() or "helmet" in cls_name.lower():
                        helmet_coords = list(map(int, box.xyxy[0]))
                        helmets.append(helmet_coords)
                        helmet_confs.append(float(box.conf[0]))
                        yolo_helmet_count += 1
                        # Draw all detected helmets in yellow for debugging
                        hx1, hy1, hx2, hy2 = helmet_coords
                        cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 1)
        
        # --- PHASE 3.5: Color-Based Yellow Helmet Detection (Fallback) ---
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV (EXPANDED for various yellow shades)
        # Covers bright yellow, pale yellow, and orange-yellow
        lower_yellow = np.array([15, 80, 80])   # Expanded range
        upper_yellow = np.array([35, 255, 255])
        
        # Create mask for yellow color
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of yellow regions
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color_helmet_count = 0
        total_yellow_contours = len(contours)
        
        # Filter contours to find helmet-sized yellow objects
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 20000:  # Lowered minimum, increased maximum
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (helmets are roughly square/circular)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.4 < aspect_ratio < 2.5:  # More forgiving
                    # Add to helmets list if not already detected by YOLO
                    color_helmet = [x, y, x+w, y+h]
                    
                    # Check if this overlaps with existing YOLO detections
                    is_duplicate = False
                    for existing in helmets:
                        ex1, ey1, ex2, ey2 = existing
                        # Check for significant overlap
                        if not (x+w < ex1 or x > ex2 or y+h < ey1 or y > ey2):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        helmets.append(color_helmet)
                        color_helmet_count += 1
                        # Draw color-detected helmets in cyan for debugging
                        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
        
        # Debug output
        if self.frame_count % 30 == 0:  # Print every 30 frames (~1 second)
            print(f"[DEBUG] Helmets: YOLO={yolo_helmet_count}, Color={color_helmet_count}, Total={len(helmets)}")
            
            if helmet_confs:
                avg_conf = sum(helmet_confs) / len(helmet_confs)
                print(f"        Helmet Model Accuracy: {avg_conf:.2%} (Range: {min(helmet_confs):.2%} - {max(helmet_confs):.2%})")
            
            if person_confs:
                avg_p_conf = sum(person_confs) / len(person_confs)
                print(f"        Person Model Accuracy: {avg_p_conf:.2%}")

            print(f"        Yellow contours found: {total_yellow_contours} (filtered to {color_helmet_count} helmets)")
            if color_helmet_count > 0 and yolo_helmet_count == 0:
                print("  ⚠️  YOLO failed - Color detection active")
            elif yolo_helmet_count == 0 and color_helmet_count == 0 and len(persons) > 0:
                print("  ⚠️  NO HELMETS DETECTED - Check lighting/color range")

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
                cv2.putText(annotated_frame, "RESTRICTED", (px1, py1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Helmet Matching Prep
            p_center_x = (px1 + px2) // 2
            p_head_y = py1 # Top of bounding box
            p_width = px2 - px1
            p_height = py2 - py1
            
            for h_i, (hx1, hy1, hx2, hy2) in enumerate(helmets):
                 h_center_x = (hx1 + hx2) // 2
                 h_center_y = (hy1 + hy2) // 2
                 h_width = hx2 - hx1
                 h_height = hy2 - hy1
                 
                 # Stricter validation: helmet must be in upper portion of person box
                 # and reasonably sized
                 if (px1 - 40 < h_center_x < px2 + 40) and \
                    (py1 - 60 < h_center_y < py1 + (py2 - py1)//3):
                     
                     # Validate helmet size (should be reasonable relative to person)
                     if h_width > p_width * 1.5 or h_height > p_height * 0.6:
                         continue  # Skip unreasonably large helmets
                     
                     if h_width < 20 or h_height < 20:
                         continue  # Skip tiny detections
                     
                     # CRITICAL: Check if helmet overlaps with OTHER people's head regions
                     # This prevents cross-person helmet assignment when people are close
                     overlaps_other_person = False
                     for other_i, (ox1, oy1, ox2, oy2) in enumerate(persons):
                         if other_i != p_i:  # Don't check against self
                             other_head_region_y = oy1 + (oy2 - oy1)//3
                             # Check if helmet is in another person's head region
                             if (ox1 - 20 < h_center_x < ox2 + 20) and \
                                (oy1 - 30 < h_center_y < other_head_region_y):
                                 overlaps_other_person = True
                                 if self.frame_count % 30 == 0:
                                     print(f"  ⚠️  Helmet overlap detected - rejecting cross-person match")
                                 break
                     
                     if overlaps_other_person:
                         continue  # Skip this helmet for this person
                     
                     # Calculate distance with weighted vertical alignment
                     horizontal_dist = abs(p_center_x - h_center_x)
                     vertical_dist = abs(p_head_y - h_center_y)
                     dist = horizontal_dist + (vertical_dist * 2)
                     
                     # Stricter threshold
                     max_acceptable_dist = (p_width * 2) + 100
                     if dist < max_acceptable_dist:
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
            self.violation_count = min(30, self.violation_count + 10) # Immediate Alert (2 frames)
            
        elif not persons:
            self.violation_count = 0
            
        elif violation_found:
            self.violation_count = min(30, self.violation_count + 2) # Smoother to prevent flicker (buffer)
        else:
            self.violation_count = max(0, self.violation_count - 4) # Fast Recovery
        
        # Status Text Logic (Support Multiple Messages)
        status_lines = []
        
        if roi_violation:
             status_lines.append(("CRITICAL: UNAUTHORIZED ACCESS", (0, 0, 255)))
             
        if self.violation_count > self.alert_limit:
             status_lines.append(("WARNING: NO HELMET DETECTED", (0, 0, 255)))
             
        if not status_lines:
             status_lines.append(("Status: Monitoring", (0, 255, 0)))

        # Trigger Actions
        if self.violation_count > self.alert_limit or roi_violation:
            parts = []
            if roi_violation: parts.append("Restricted Zone Violation")
            if self.violation_count > self.alert_limit: parts.append("No Helmet Detected")
            
            violation_type = " + ".join(parts)

            # 1. Log to CSV
            if self.frame_count % 60 == 0:
                self.logger.log(len(persons), violation_type)
            
            # 2. TRIGGER GEMINI AI
            trigger_msg = f"High Priority: {violation_type}"
            self.ai.analyze_scene(frame, trigger_reason=trigger_msg)
            
            # 3. TRIGGER TELEGRAM ALERT
            self.telegram.send_frame(frame, caption=f"🚨 ALERT: {violation_type}")

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

