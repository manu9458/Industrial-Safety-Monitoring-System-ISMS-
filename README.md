# 🛡️ Industrial Safety Monitoring System (ISMS)

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-yellow.svg)

A production-grade, computer vision-based surveillance system designed for industrial environments. This system detects personnel, verifies Personal Protective Equipment (PPE) compliance (specifically hard hats), and monitors restricted zones in real-time.

## 🚀 Key Features

*   **Real-time Person Detection**: Utilizes YOLOv8 for accurate human detection.
*   **PPE Compliance Verification**: Detects and validates usage of safety helmets/hard hats.
*   **Restricted Zone Monitoring**: configurable virtual geofencing to alert on unauthorized access.
*   **Intelligent Alerting**: 
    *   Visual indicators (Green=Safe, Red=Violation).
    *   Telegram Integration for instant snapshot alerts to safety supervisors.
*   **Performance Optimized**: Threaded capture methodology to ensure high FPS execution.
*   **Robust Logging**: Detailed CSV logs of all safety violations and system events.
*   **Environment Ready**: Configurable for both Development and Production environments.

## 📂 Project Structure

```
├── src/
│   ├── config/         # Configuration and Environment Management
│   ├── core/           # Main Surveillance Logic & Camera Handling
│   ├── services/       # External Integrations (Telegram, etc.)
│   └── utils/          # Utilities (Logging, Helpers)
├── models/             # YOLO Weights (yolov8n.pt, hardhat.pt)
├── logs/               # Activity Logs (Gitignored)
├── .env                # Secrets & Config (Gitignored)
├── main.py             # Entry Point
└── requirements.txt    # Dependencies
```

## 🛠️ Installation

### Prerequisites
*   Python 3.9 or higher
*   Webcam or IP Camera source

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/industrial-safety-monitor.git
cd industrial-safety-monitor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```ini
# Environment (development / production)
APP_ENV=development

# Telegram Alerts (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

## 🖥️ Usage

Run the application:
```bash
python main.py
```

*   **To Exit**: Press `Q` or `ESC`.

## ⚙️ Logic & Thresholds

*   **Violations**: An alert is triggered if a person is detected without a helmet OR entering the restricted zone.
*   **Alert Cooldown**: Alerts are throttled (configurable in `src/config/settings.py`) to prevent spamming.
*   **Confidence**:
    *   Development: Lower thresholds (40-50%) for easier testing.
    *   Production: Higher thresholds (60-70%) for reliability.

## 🤝 Contributing

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
