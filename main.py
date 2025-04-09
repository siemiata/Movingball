import cv2
import numpy as np

video_file = "movingball.mp4"
video_capture = cv2.VideoCapture(video_file)

if not video_capture.isOpened():
    print("Błąd: Nie można otworzyć pliku wideo.")
    exit()

while True:
    success, frame = video_capture.read()
    if not success:
        break

    frame = cv2.resize(frame, (700, 500))

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
    red_mask = mask_red1 + mask_red2

    morph_kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, morph_kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morph_kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

    cv2.imshow("Movingball", frame)

    if cv2.waitKey(30) & 0xFF == ord('e'):
        break

video_capture.release()
cv2.destroyAllWindows()