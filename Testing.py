import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Check for specific hand gestures here
            # Example: Check if thumb and index finger are close together
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip landmark
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip landmark
            middle_finger_tip = hand_landmarks.landmark[20]
            distance = ((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)**0.5
            middleDistance =((thumb_tip.x - middle_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)**0.5

            if middleDistance < 0.1:  # Adjust this threshold as needed
                cv2.putText(image, 'Thumb and Pinky Finger Close', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if distance < 0.05:  # Adjust this threshold as needed
                cv2.putText(image, 'Thumb and Index Finger Close', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.namedWindow("Resized_Window",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 150, 350) 
    # Display the image
    cv2.imshow('Hand Gesture Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
