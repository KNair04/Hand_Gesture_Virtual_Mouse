import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui  # Library to simulate keypresses

##########################
wCam, hCam = 720, 480
smoothening = 7
frameR = 100
clickDelay = 0.6  # 500ms delay between clicks
zoomThresholdIn = 30  # Threshold for zoom-in
zoomThresholdOut = 50  # Upper bound threshold for zoom-out
zoomSpeed = 0.001  # Adjust this if needed
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
lastClickTime = 0  # Variable to track the last click time

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip camera image
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        xThumb, yThumb = lmList[4][1:]  # Thumb tip
        xPinkie, yPinkie = lmList[20][1:]  # Pinkie finger tip

        # 2. Check which fingers are up
        fingers = detector.fingersUp()

        # 3. Zoom-in and Zoom-out gesture detection
        if fingers[0] == 1 and fingers[1] == 1:  # Thumb and Index finger up
            # Calculate the distance between the thumb and index finger, without drawing
            length, img, lineInfo = detector.findDistance(4, 8, img, draw=False)
            print(f"Distance between thumb and index finger: {length}")

            # Zoom-in when the distance decreases below zoomThresholdIn
            if length < zoomThresholdIn:
                cv2.putText(img, "Zoom In", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                pyautogui.hotkey('ctrl', '+')  # Simulate Ctrl + for zoom in
                print("Zooming in (Ctrl +)")

            # Zoom-out when the distance is between zoomThresholdIn and zoomThresholdOut
            elif zoomThresholdIn <= length < zoomThresholdOut:
                cv2.putText(img, "Zoom Out", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                pyautogui.hotkey('ctrl', '-')  # Simulate Ctrl - for zoom out
                print("Zooming out (Ctrl -)")

        # 4. Existing functionality: Cursor movement when only the index finger is up
        if fingers[1] == 1 and fingers[2] == 0:  # Only Index Finger: Moving Mode
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 5. Left-Clicking Mode: Both index and middle finger up
        if fingers[1] == 1 and fingers[2] == 1:
            # Calculate the distance between index and middle finger, without drawing
            length, img, lineInfo = detector.findDistance(8, 12, img, draw=False)

            # Register a click if fingers are close, and ensure enough delay between clicks
            if length < 39 and (time.time() - lastClickTime) > clickDelay:
                autopy.mouse.click()  # Perform the left-click action
                lastClickTime = time.time()  # Update last click time

        # 6. Right-Clicking Mode: Only pinkie finger up
        if fingers[4] == 1 and all(f == 0 for f in fingers[:4]):  # Only Pinkie: Right Click Mode
            if (time.time() - lastClickTime) > clickDelay:  # Ensure delay between right clicks
                autopy.mouse.click(autopy.mouse.Button.RIGHT)  # Perform the right-click action
                print("Right click performed")
                lastClickTime = time.time()  # Update last click time

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
