import cv2
import mediapipe as mp
import numpy as np
import os

from image_utils import create_placeholder_image, overlay_image_alpha
from gesture_detector import recognize_gesture

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    print("Memuat dan mengubah ukuran gambar overlay...")

    new_overlay_size = (100, 100)

    monkey_img = cv2.imread("assets/Monkey_reaction.jpg", cv2.IMREAD_COLOR)
    if monkey_img is not None:
        monkey_img = cv2.resize(
            monkey_img, new_overlay_size, interpolation=cv2.INTER_AREA
        )

    monkey_img_1 = cv2.imread("assets/Monkey_reaction_1.jpg", cv2.IMREAD_COLOR)
    if monkey_img_1 is not None:
        monkey_img_1 = cv2.resize(
            monkey_img_1, new_overlay_size, interpolation=cv2.INTER_AREA
        )

    monkey_img_2 = cv2.imread("assets/Monkey_reaction_2.jpg", cv2.IMREAD_COLOR)
    if monkey_img_2 is not None:
        monkey_img_2 = cv2.resize(
            monkey_img_2, new_overlay_size, interpolation=cv2.INTER_AREA
        )

    monkey_img_3 = cv2.imread("assets/Monkey_reaction_3.jpg", cv2.IMREAD_COLOR)
    if monkey_img_3 is not None:
        monkey_img_3 = cv2.resize(
            monkey_img_3, new_overlay_size, interpolation=cv2.INTER_AREA
        )

    if monkey_img is None:
        print(
            "Peringatan: 'monkey_reaction.jpg' tidak ditemukan. Membuat placeholder 100x100."
        )
        monkey_img = create_placeholder_image(
            "THINKING", color=(200, 0, 200)
        )
    if monkey_img_1 is None:
        print(
            "Peringatan: 'monkey_reaction_1.jpg' tidak ditemukan. Membuat placeholder 100x100."
        )
        monkey_img_1 = create_placeholder_image(
            "AHA!", color=(0, 200, 50)
        )
    if monkey_img_2 is None:
        print(
            "Peringatan: 'monkey_reaction_2.jpg' tidak ditemukan. Membuat placeholder 100x100."
        )
        monkey_img_2 = create_placeholder_image(
            "NO_REACTION", color=(150, 150, 150)
        )
    if monkey_img_3 is None:
        print(
            "Peringatan: 'monkey_reaction_3.jpg' tidak ditemukan. Membuat placeholder 100x100."
        )
        monkey_img_3 = create_placeholder_image(
            "SHOCK", color=(200, 50, 0)
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    print("\nMenjalankan deteksi. Tekan 'q' atau 'Esc' untuk keluar.")

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Mengabaikan frame kamera kosong.")
                continue

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            detected_gestures = []
            num_hands_detected = 0

            if results.multi_hand_landmarks:
                num_hands_detected = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    gesture = recognize_gesture(hand_landmarks) 
                    if gesture != "NONE":
                        detected_gestures.append(gesture)

            current_gesture = "NONE"
            if num_hands_detected == 0:
                current_gesture = "NO_REACTION"
            elif detected_gestures.count("FIST") >= 2:
                current_gesture = "SHOCK"
            elif "AHA!" in detected_gestures:
                current_gesture = "AHA!"
            elif "THINKING" in detected_gestures:
                current_gesture = "THINKING"
            elif "FIST" in detected_gestures:
                current_gesture = "FIST"

            frame_height, frame_width = frame_bgr.shape[:2]
            overlay_width = new_overlay_size[0]

            pos_x = frame_width - overlay_width - 20
            pos_y = 20

    
            if current_gesture == "SHOCK":
                frame_bgr = overlay_image_alpha(frame_bgr, monkey_img_3, pos_x, pos_y)
            elif current_gesture == "NO_REACTION":
                frame_bgr = overlay_image_alpha(frame_bgr, monkey_img_2, pos_x, pos_y)
            elif current_gesture == "AHA!":
                frame_bgr = overlay_image_alpha(frame_bgr, monkey_img_1, pos_x, pos_y)
            elif current_gesture == "THINKING":
                frame_bgr = overlay_image_alpha(frame_bgr, monkey_img, pos_x, pos_y)

            cv2.putText(
                frame_bgr,
                f"Gesture: {current_gesture}",
                (10, frame_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-time Hand Gesture Recognition", frame_bgr)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q") or key == 27:
                print("Menutup program...")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()