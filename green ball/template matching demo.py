import cv2
import numpy as np
import imutils

greenLower = (40, 86, 6)
greenUpper = (64, 255, 255)
v = cv2.VideoCapture('input.mp4')
cropframe = None
temp = None
while True:
    ret, src = v.read()
# preprocess
    frame = src.copy()
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
# get ball's edge
    cnt = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = imutils.grab_contours(cnt)
# when contour exist
    if len(cnt) > 0:
        c = max(cnt, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
# draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        M = cv2.moments(c)
# cut out target area and get temp
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cropframe = frame[abs(center[1]-30):abs(center[1]+30), abs(center[0]-30):abs(center[0]+30)]
        temp = frame[y:y+h, x:x+w]
# template matching
    frame2 = src.copy()
    match = cv2.matchTemplate(frame2, temp, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (int(temp.shape[0]) + top_left[0], int(temp.shape[1]) + top_left[1])
    cv2.rectangle(frame2, top_left, bottom_right, (255, 0, 0), 2)

# resize the temp match and surround area
    cropframe = cv2.resize(cropframe, dsize=(312, 184), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    frame2 = cv2.resize(frame2, dsize=(312, 184), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
# add title for display
    cv2.putText(frame, 'target detection and optical flow (unrealized)', (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(cropframe, 'surround area', (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame2, 'temp match', (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    leftimg = np.vstack([cropframe, frame2])
    finimg = np.hstack([leftimg, frame])

    # cv2.imshow('template', frame2)
    # cv2.imshow('input', frame)
    # cv2.imshow('target area', cropframe)
    cv2.imshow('output', finimg)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2.VideoWriter('c:/py.resource/output.mp4', fourcc, 60, (936, 184*3), True)
    cv2.waitKey(20)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break


