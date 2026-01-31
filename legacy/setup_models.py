import requests
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        print("Please manually download it or check your internet connection.")

def main():
    # Standard YOLOv8n (Person Detection)
    # Ultralytics auto-downloads this, but we can be explicit
    # download_file("https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt", "yolov8n.pt")
    
    # PPE Detection Model (Hard Hat)
    # Source: https://huggingface.co/keremberke/yolov8n-hard-hat-detection
    # This model detects: ['hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']
    ppe_url = "https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt"
    download_file(ppe_url, "hardhat.pt")

if __name__ == "__main__":
    main()
