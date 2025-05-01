Movingball
- App is tracking red ball on video
- Video here: https://drive.google.com/file/d/17hP-5p_SWXIJOmF47ha9Qq-TtPHE7qMD/view?usp=sharing

1. Wczytanie i przygotowanie wideo:
   - video_capture = cv2.VideoCapture("movingball.mp4")
   - frame = cv2.resize(frame, (700, 500))

2. Konwersja koloru do HSV:
   - hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
3. Maskowanie czerwonego koloru:
   - mask_red1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
   - mask_red2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
   - red_mask = mask_red1 + mask_red2
  
4. Filtracja morfologiczna:
   - red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, morph_kernel)
   - red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morph_kernel)
  
5. Wyszukiwanie konturów i zaznaczanie piłki:
   - contours, _ = cv2.findContours(...)
   - if cv2.contourArea(contour) > 500:
   - (x, y), radius = cv2.minEnclosingCircle(contour)
   - cv2.circle(frame, center, radius, ...)
  
6. Wyświetlanie i zakończenie programu:
   - cv2.imshow("Movingball", frame)
   - if cv2.waitKey(30) & 0xFF == ord('e'):
   - break
