
import cv2
import time
import sys
from src.core.camera import ThreadedCamera
from src.core.surveillance import SurveillanceSystem
from src.config.settings import Config

def main():
    print("Starting Industrial Monitoring System...")
    print("Initializing components...")

    # Initialize Camera
    camera = ThreadedCamera(Config.CAMERA_SOURCE)
    if not camera.start():
        print("Error: Could not access camera.")
        sys.exit(1)

    # Initialize System
    try:
        system = SurveillanceSystem()
    except Exception as e:
        print(f"Critical Error during initialization: {e}")
        camera.stop()
        sys.exit(1)

    print("System Active. Press 'Q' or 'ESC' to exit.")

    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                continue

            # Process
            processed_frame = system.process_frame(frame)

            # Display
            cv2.imshow("Industrial Monitor", processed_frame)

            # Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # ESC
                break
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Runtime Error: {e}")
            break

    # Cleanup
    print("Shutting down...")
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
