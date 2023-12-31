import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
finger_coordinates = [(8, 6), (12, 10), (16, 14), (20, 10)]
thumb_coordinate = (4, 2)
mp_drawing = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        print("End of video.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multi_landmarks = results.multi_hand_landmarks

    if multi_landmarks:
        hand_points = []
        for hand_lms in multi_landmarks:
            mp_drawing.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_points.append((cx, cy))

        for point in hand_points:
            cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)

        up_count = 0

        for coordinate in finger_coordinates:
            if hand_points[coordinate[0]][1] < hand_points[coordinate[1]][1]:
                up_count += 1

        cv2.putText(img, str(up_count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 12)

    cv2.imshow('Fingercounts', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()