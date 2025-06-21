
import cv2
import multiprocessing as mp
from PIL import Image
import numpy as np
from io import BytesIO

from Face import face
from money import money
from sceneBackup import scene

import os
import requests
from time import sleep

import pyttsx3 as tts

ESP32_IP = "192.168.4.1"
STATUS_URL = f"http://{ESP32_IP}/status"
CAPTURE_URL = f"http://{ESP32_IP}/capture"
OUTPUT_DIR = r"\imagges"  # Update this path

os.makedirs(OUTPUT_DIR, exist_ok=True)

def speak(text):
    engine = tts.init()
    engine.say(text)
    engine.runAndWait()

def download_image(output_dir, image_counter):
    try:
        sleep(.8)
        response = requests.get(CAPTURE_URL, timeout=10)        
        if response.status_code == 200:
            press_count = response.headers.get('Press-Count', '0')
            response = requests.get(CAPTURE_URL, timeout=10)
            image_path = os.path.join(output_dir, f"image.jpg")
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Image saved: {image_path}, Press Count: {press_count}")
            
            # Convert to BGR numpy array
            pil_image = Image.open(image_path)
            np_image = np.array(pil_image) 
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR) 
            
            print(f"get press count as {press_count}")
            # Pass numpy array instead
            print(f"Model {int(press_count)-1} Result: {master_function(int(press_count)-1, np_image)}")
            return True
        else:
            print(f"Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error in download_image: {e}")
        return False

    
def check_status():
    try:
        response = requests.get(STATUS_URL, timeout=5)
        if response.ok:
            return response.text.strip() == "ON"
    except:
        return False
    return False

def main_esp():
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

def master_function(model_choice, image):
    """Modified master function to accept image array instead of file path"""
    match model_choice:
        case -1:
            print("invalid choice")
            speak("Invalid choice!! please try again")
            return 
        case 0:
            print("running face recognition model")
            output = face.face_inference(image)
            speak(output)
            return output
        case 1:
            output = money.money_inference(image)
            speak(output)
            return output
        case _:            
            pipeline = scene.InferencePipeline()
            output = pipeline.generate_caption(image, beam_size=2)
            speak(output)
            return output

def process_wrapper(model_choice, image, result_queue, available):
    """Wrapper function for processing with resource management"""
    try:
        result = master_function(model_choice, image)
        result_queue.put((model_choice, result))
    finally:
        available.release()

def main_live():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Use a semaphore to manage concurrent processing
    available = mp.Semaphore(1)
    result_queue = mp.Queue()
    
    print("Webcam ready. Press 0-2 for inference, Q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            cv2.imshow('Real-time Webcam', cv2.flip(frame, 1))

            # Check for results from completed processes
            while not result_queue.empty():
                model_choice, result = result_queue.get()
                print(f"Model {model_choice} Result: {result}")

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('0'), ord('1'), ord('2')):
                if available.acquire(block=False):
                    try:
                        model_choice = key - ord('0')
                        process = mp.Process(
                            target=process_wrapper,
                            args=(model_choice, frame.copy(), result_queue, available)
                        )
                        process.start()
                    except:
                        available.release()
                        raise

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # mp.freeze_support()
    # main_live()
    main_esp()