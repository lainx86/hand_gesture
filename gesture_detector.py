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
        return "AHA!"

    if not index_up and not middle_up and not ring_up and not pinky_up:
        if (lm[8].y > lm[6].y) and (lm[8].y < lm[5].y):
            return "THINKING"

        return "FIST"

    return "NONE"