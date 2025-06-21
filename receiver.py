
import os
import requests
from time import sleep

ESP32_IP = "192.168.4.1"
STATUS_URL = f"http://{ESP32_IP}/status"
CAPTURE_URL = f"http://{ESP32_IP}/capture"
OUTPUT_DIR = r"\imagges"  # Update this path

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image(output_dir, image_counter):
    try:
        response = requests.get(CAPTURE_URL, timeout=10)
        if response.status_code == 200:
            sleep(2)
            press_count = response.headers.get('Press-Count', '0')
            image_path = os.path.join(output_dir, f"image_{image_counter}_press_{press_count}.jpg")
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Image saved: {image_path}, Press Count: {press_count}")
            return True
        else:
            print(f"Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_status():
    try:
        response = requests.get(STATUS_URL, timeout=5)
        if response.ok:
            return response.text.strip() == "ON"
    except:
        return False
    return False

def main():
    image_counter = 1
    last_state = False

    print("Monitoring ESP32-CAM...")
    while True:
        current_state = check_status()
        if current_state != last_state:
            print(f"State changed to {'ON' if current_state else 'OFF'}")
            if current_state:
                success = download_image(OUTPUT_DIR, image_counter)
                if success:
                    image_counter += 1
            last_state = current_state

if __name__ == "__main__":
    main()

