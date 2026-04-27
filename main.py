import cv2
import mediapipe as mp
import pyautogui as pag
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

clicked = False

screen_w, screen_h = pag.size()

cap = cv2.VideoCapture(2)

x, y = pag.position()

smooth = 5
prev_x, prev_y = 0,0

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(f"Dedo {id}: {cx}, {cy}")
                    x1, y1 = handLms.landmark[4].x, handLms.landmark[4].y
                    x2, y2 = handLms.landmark[12].x, handLms.landmark[12].y
                    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    #print(f"Distância: {dist}")
                    if id == 8:
                        target_x = int(lm.x * screen_w)
                        target_y = int(lm.y * screen_h)
                        mouse_x = prev_x + (target_x - prev_x) / smooth
                        mouse_y = prev_y + (target_y - prev_y) / smooth
                        pag.moveTo(mouse_x, mouse_y)
                        prev_x, prev_y = mouse_x, mouse_y
                    if dist < 0.03 and not clicked:
                        pag.click()
                        clicked = True
                        #print("Clicked")
                    if dist > 0.03:
                        clicked = False

        cv2.imshow("Camera", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()