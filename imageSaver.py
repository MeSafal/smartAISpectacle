import os
import requests
from time import sleep
from datetime import datetime

ESP32_IP = "192.168.4.1"
STATUS_URL = f"http://{ESP32_IP}/status"
CAPTURE_URL = f"http://{ESP32_IP}/capture"
OUTPUT_DIR = r"\data_images"  # Update this path if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_captured_image(image_data, output_dir):
    """
    Saves image_data to output_dir with a filename in the format image_YYYYMMDD_HHMMSS.jpeg.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"image_{timestamp}.jpeg"
    image_path = os.path.join(output_dir, image_name)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    print(f"Image saved: {image_path}")

def download_image(output_dir):
    """
    Downloads the image from CAPTURE_URL and saves it using save_captured_image().
    """
    try:
        response = requests.get(CAPTURE_URL, timeout=10)
        if response.status_code == 200:
            save_captured_image(response.content, output_dir)
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error in download_image: {e}")
        return False

def check_status():
    """
    Checks the status of the ESP32 by reading its STATUS_URL.
    Returns True if the status text is "ON", False otherwise.
    """
    try:
        response = requests.get(STATUS_URL, timeout=5)
        if response.ok:
            return response.text.strip() == "ON"
    except Exception:
        return False
    return False

def main_esp():
    last_state = False
    print("Monitoring ESP32-CAM...")
    while True:
        current_state = check_status()
        if current_state != last_state:
            print(f"State changed to {'ON' if current_state else 'OFF'}")
            if current_state:
                while True:
                    download_image(OUTPUT_DIR)
                    sleep(2)
            last_state = current_state

if __name__ == "__main__":
    main_esp()
