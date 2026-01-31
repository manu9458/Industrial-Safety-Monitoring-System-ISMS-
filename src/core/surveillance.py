
import cv2
import numpy as np
import time
from ultralytics import YOLO

from src.config.settings import Config
from src.utils.logger import ActivityLogger
from src.services.telegram import TelegramService

class SurveillanceSystem:
    def __init__(self):
        self.logger = ActivityLogger(Config.LOG_FILE)
        self.logger.info("Initializing Surveillance System...")

        # Initialize Models
        self._load_models()
        
        # Services
        self.telegram = TelegramService(Config.TELEGRAM_TOKEN, Config.TELEGRAM_CHAT_ID)
        
        # State
        self.frame_count = 0
        self.violation_counter = 0
        self.last_routine_scan = 0

    def _load_models(self):
        try:
            self.model_person = YOLO(Config.MODEL_PERSON)
            self.logger.info(f"Loaded {Config.MODEL_PERSON}")
        except Exception as e:
            self.logger.error(f"Failed to load person model: {e}")
            raise e

        try:
            self.model_appe = YOLO(Config.MODEL_PPE)
            self.ppe_active = True
            self.logger.info(f"Loaded {Config.MODEL_PPE}")
        except Exception:
            self.logger.warning("PPE Model not found. Running in limited mode.")
            self.ppe_active = False

    def process_frame(self, frame):
        if frame is None:
            return frame

        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Define Restricted Zone (Right 25%)
        zone_x = int(width * 0.75)
        roi_poly = np.array([
            [zone_x, 0], [width, 0], 
            [width, height], [zone_x, height]
        ], np.int32)

        # Draw Zone
        self._draw_zone(frame, roi_poly)

        # 1. Detect Persons
        persons, p_confs = self._detect_persons(frame, height)
        
        # 2. Detect Helmets
        helmets, h_confs = self._detect_helmets(frame)

        # 3. Analyze Safety & Violations
        matches, violations, safe_persons = self._match_ppe(persons, helmets)
        
        # 4. Check Zone Violations
        zone_violations = self._check_zone_access(matches + violations, roi_poly)

        # Visualization
        self._draw_detections(frame, safe_persons, violations, zone_violations)
        
        # Alert Logic
        status_text = self._handle_alerts(frame, len(violations), len(zone_violations))
        
        # Render Status
        self._draw_status(frame, status_text)
        
        # Debug Logs (Model Accuracy)
        if self.frame_count % 30 == 0:
            self._log_debug_stats(p_confs, h_confs)

        return frame

    def _draw_zone(self, frame, poly):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
        cv2.putText(frame, "RESTRICTED AREA", (poly[0][0] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _detect_persons(self, frame, img_height):
        # Run YOLO inference
        results = self.model_person(frame, classes=[0], stream=True, 
                                  conf=Config.CONF_PERSON, verbose=False)
        
        boxes = []
        confs = []
        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                conf = float(box.conf[0])
                
                # Filter small detections (e.g. erratic artifacts)
                h = coords[3] - coords[1]
                if h > img_height * 0.2: # Min 20% height
                    boxes.append(coords)
                    confs.append(conf)
        
        return boxes, confs

    def _detect_helmets(self, frame):
        boxes = []
        confs = []
        
        if not self.ppe_active:
            # Fallback: Color based (Simple Yellow)
            # Keeping it simple for this implementation refactor
            return boxes, confs

        results = self.model_appe(frame, stream=True, conf=Config.CONF_HELMET, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_name = self.model_appe.names[int(box.cls[0])]
                if "hat" in cls_name.lower() or "helmet" in cls_name.lower():
                    boxes.append(list(map(int, box.xyxy[0])))
                    confs.append(float(box.conf[0]))
        
        return boxes, confs

    def _match_ppe(self, persons, helmets):
        # Greedy matching based on head position
        # Returns: matches (indices), violations (indices), safe_persons
        
        safe = []
        violations = []
        
        used_helmets = set()
        
        for p in persons:
            px1, py1, px2, py2 = p
            p_head_y = py1
            p_center_x = (px1 + px2) // 2
            
            has_helmet = False
            
            # Find best helmet match
            best_h = -1
            min_dist = float('inf')
            
            for i, h in enumerate(helmets):
                if i in used_helmets:
                    continue
                    
                hx1, hy1, hx2, hy2 = h
                h_center_x = (hx1 + hx2) // 2
                h_center_y = (hy1 + hy2) // 2
                
                # Logic: Helmet must be near top of person box
                if (px1 < h_center_x < px2) and (py1 - 50 < h_center_y < py1 + (py2-py1)//3):
                     dist = abs(p_center_x - h_center_x) + abs(p_head_y - h_center_y)
                     if dist < min_dist:
                         min_dist = dist
                         best_h = i
            
            if best_h != -1:
                person_width = px2 - px1
                if min_dist < person_width: # Reasonable proximity
                    used_helmets.add(best_h)
                    has_helmet = True
            
            if has_helmet:
                safe.append(p)
            else:
                violations.append(p)
                
        return [], violations, safe

    def _check_zone_access(self, persons, zone_poly):
        violators = []
        for p in persons:
            x1, y1, x2, y2 = p
            feet_point = (int((x1 + x2) / 2), y2)
            if cv2.pointPolygonTest(zone_poly, feet_point, False) >= 0:
                violators.append(p)
        return violators

    def _draw_detections(self, frame, safe, violations, zone_violations):
        for p in safe:
            x1, y1, x2, y2 = p
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "SAFE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for p in violations:
            x1, y1, x2, y2 = p
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "NO HELMET", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for p in zone_violations:
             x1, y1, x2, y2 = p
             cv2.circle(frame, (int((x1+x2)/2), y2), 5, (0, 0, 255), -1)

    def _handle_alerts(self, frame, violation_count, zone_count):
        status = "Status: Nominal"
        
        is_violation = (violation_count > 0 or zone_count > 0)
        
        if is_violation:
            self.violation_counter += 1
        else:
            self.violation_counter = max(0, self.violation_counter - 1)
            
        if self.violation_counter > Config.ALERT_COOLDOWN: # Using cooldown as a frame buffer roughly
            msg = []
            if zone_count > 0: msg.append("Restricted Zone Access")
            if violation_count > 0: msg.append("PPE Violation")
            
            alert_msg = " + ".join(msg)
            status = f"ALERT: {alert_msg}"
            
            # Trigger External Services
            self.telegram.send_snapshot(frame, f"🚨 {alert_msg}")
            
            if self.frame_count % 60 == 0:
                self.logger.log_event(violation_count + zone_count, "VIOLATION", alert_msg)

        return status

    def _draw_status(self, frame, text):
        color = (0, 0, 255) if "ALERT" in text else (0, 255, 0)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def _log_debug_stats(self, p_confs, h_confs):
        if p_confs:
            avg = sum(p_confs)/len(p_confs)
            print(f"[DEBUG] Person Accuracy: {avg:.2%}")
        if h_confs:
            avg = sum(h_confs)/len(h_confs)
            print(f"[DEBUG] Helmet Accuracy: {avg:.2%}")
