import csv
import os

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    csv_file = "gesture_data.csv"
    header = ["label"] + [f"v{i}" for i in range(21 * 3)]
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a+", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
            print(f"File '{csv_file}' dibuat dengan header.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    print("\n--- Mulai Pengumpulan Data ---")
    print("Tekan 'a' untuk gestur 'AHA!' (Pointing Up)")
    print("Tekan 't' untuk gestur 'THINKING' (Curved Finger)")
    print("Tekan 'f' untuk gestur 'FIST' (Kepalan')")
    print("Tekan 'q' untuk keluar.")
    print("\nVariasikan posisi dan sudut tangan Anda saat mengambil data!")

    count = {"AHA!": 0, "THINKING": 0, "FIST": 0}

    with (
        mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=1,
        ) as hands
    ):
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            label = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow(
                "Data Collector - Tekan (a, t, f) untuk Simpan, (q) untuk Keluar",
                frame_bgr,
            )
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                print("...Keluar dari data collector.")
                break
            elif key == ord("a"):
                label = "AHA!"
            elif key == ord("t"):
                label = "THINKING"
            elif key == ord("f"):
                label = "FIST"

            if label and results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                row = []
                for lm in hand_lm.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + row)

                count[label] += 1
                print(f"Data '{label}' disimpan! (Total: {count[label]})")

            elif label and not results.multi_hand_landmarks:
                print("Tangan tidak terdeteksi. Coba lagi.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nPengumpulan data selesai. Data disimpan di '{csv_file}'.")
    print(f"Total data terkumpul:")
    for k, v in count.items():
        print(f"- {k}: {v} data")


if __name__ == "__main__":
    main()
