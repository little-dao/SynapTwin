import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start capturing video
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """ Count the number of extended fingers. """
    tips = [8, 12, 16, 20]  # Fingertip landmark indices
    count = 0

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:  # Check if extended
            count += 1

    return count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(hand_landmarks)

            if fingers == 0:
                text = "Fist (Move Forward)"
            elif fingers >= 4:
                text = "Open Palm (Stop)"
            elif fingers == 1:
                text = "One Finger (Turn Left)"
            else:
                text = "Gesture Not Recognized"

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
