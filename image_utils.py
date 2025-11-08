import cv2
import numpy as np

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