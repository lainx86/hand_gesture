import os

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_placeholder_image(text, size=(100, 100), color=(0, 100, 0)):
    image = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    image[:, :, 0] = color[0]
    image[:, :, 1] = color[1]
    image[:, :, 2] = color[2]
    image[:, :, 3] = 150

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (size[0] - text_w) // 2
    text_y = (size[1] + text_h) // 2

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return image


def overlay_image_alpha(img_bg, img_overlay, x, y):
    try:
        h_bg, w_bg = img_bg.shape[:2]
        h_ol, w_ol = img_overlay.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_bg, x + w_ol), min(h_bg, y + h_ol)

        w_roi = x2 - x1
        h_roi = y2 - y1

        if w_roi <= 0 or h_roi <= 0:
            return img_bg

        roi_bg = img_bg[y1:y2, x1:x2]
        roi_ol = img_overlay[0:h_roi, 0:w_roi]

        if roi_ol.shape[2] == 4:
            alpha = roi_ol[:, :, 3] / 255.0
            alpha_mask = np.dstack([alpha] * 3)
            overlay_bgr = roi_ol[:, :, :3]
            composite = (overlay_bgr * alpha_mask) + (roi_bg * (1.0 - alpha_mask))
            img_bg[y1:y2, x1:x2] = composite.astype(np.uint8)
        elif roi_ol.shape[2] == 3:
            img_bg[y1:y2, x1:x2] = roi_ol

        return img_bg

    except Exception as e:
        print(f"Error saat overlay gambar: {e}")
        return img_bg


def recognize_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers_up = [False] * 5

    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    mcp_ids = [5, 9, 13, 17]

    for i in range(4):
        if lm[tip_ids[i]].y < lm[pip_ids[i]].y:
            fingers_up[i + 1] = True

    if lm[4].y < lm[3].y and lm[4].y < lm[2].y:
        fingers_up[0] = True

    thumb_up = fingers_up[0]
    index_up = fingers_up[1]
    middle_up = fingers_up[2]
    ring_up = fingers_up[3]
    pinky_up = fingers_up[4]

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "AHA!"  # <-- DIUBAH

    if not index_up and not middle_up and not ring_up and not pinky_up:
        if (lm[8].y > lm[6].y) and (lm[8].y < lm[5].y):
            return "THINKING"  # <-- DIUBAH

        return "FIST"

    return "NONE"


def main():
    print("Memuat dan mengubah ukuran gambar overlay...")

    new_overlay_size = (100, 100)

    monkey_img = cv2.imread("monkey_reaction.jpg", cv2.IMREAD_COLOR)
    if monkey_img is not None:
        monkey_img = cv2.resize(
            monkey_img, new_overlay_size, interpolation=cv2.INTER_AREA
        )

    monkey_img_1 = cv2.imread("monkey_reaction_1.jpg", cv2.IMREAD_COLOR)
    if monkey_img_1 is not None:
        monkey_img_1 = cv2.resize(
            monkey_img_1, new_overlay_size, interpolation=cv2.INTER_AREA
        )

    monkey_img_2 = cv2.imread("monkey_reaction_2.jpg", cv2.IMREAD_COLOR)
    if monkey_img_2 is not None:
        monkey_img_2 = cv2.resize(
            monkey_img_2, new_overlay_size, interpolation=cv2.INTER_AREA
        )

    monkey_img_3 = cv2.imread("monkey_reaction_3.jpg", cv2.IMREAD_COLOR)
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
        )  # <-- DIUBAH
    if monkey_img_1 is None:
        print(
            "Peringatan: 'monkey_reaction_1.jpg' tidak ditemukan. Membuat placeholder 100x100."
        )
        monkey_img_1 = create_placeholder_image(
            "AHA!", color=(0, 200, 50)
        )  # <-- DIUBAH
    if monkey_img_2 is None:
        print(
            "Peringatan: 'monkey_reaction_2.jpg' tidak ditemukan. Membuat placeholder 100x100."
        )
        monkey_img_2 = create_placeholder_image(
            "NO_REACTION", color=(150, 150, 150)
        )  # <-- DIUBAH
    if monkey_img_3 is None:
        print(
            "Peringatan: 'monkey_reaction_3.jpg' tidak ditemukan. Membuat placeholder 100x100."
        )
        monkey_img_3 = create_placeholder_image(
            "SHOCK", color=(200, 50, 0)
        )  # <-- DIUBAH

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
                current_gesture = "NO_REACTION"  # <-- DIUBAH
            elif detected_gestures.count("FIST") >= 2:
                current_gesture = "SHOCK"  # <-- DIUBAH
            elif "AHA!" in detected_gestures:
                current_gesture = "AHA!"  # <-- DIUBAH
            elif "THINKING" in detected_gestures:
                current_gesture = "THINKING"  # <-- DIUBAH
            elif "FIST" in detected_gestures:
                current_gesture = "FIST"

            frame_height, frame_width = frame_bgr.shape[:2]
            overlay_width = new_overlay_size[0]

            pos_x = frame_width - overlay_width - 20
            pos_y = 20

            if current_gesture == "SHOCK":  # <-- DIUBAH
                frame_bgr = overlay_image_alpha(frame_bgr, monkey_img_3, pos_x, pos_y)
            elif current_gesture == "NO_REACTION":  # <-- DIUBAH
                frame_bgr = overlay_image_alpha(frame_bgr, monkey_img_2, pos_x, pos_y)
            elif current_gesture == "AHA!":  # <-- DIUBAH
                frame_bgr = overlay_image_alpha(frame_bgr, monkey_img_1, pos_x, pos_y)
            elif current_gesture == "THINKING":  # <-- DIUBAH
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
