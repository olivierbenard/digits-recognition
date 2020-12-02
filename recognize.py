import cv2
import numpy as np

def main():

    capture = cv2.VideoCapture(0)
    left, right, top, bottom = 300, 680, 0, 480 # coordinates of the ROI (Region Of Interest)

    while True:

        _, frame = capture.read()
        frame = cv2.flip(frame, 1) # vertical mirror
        clone = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(gray, (7,7), 0) # kernel of 7x7 pixels, standard deviation equal to 0 for both X and Y directions
        roi = smotthed[top:bottom, left:right]

        cv2.imshow('Base environment', clone) # non-treated frame
        cv2.imshow('Treated environment', roi) # pre-treated frame

        keypress = cv2.waitKey(1)

        if keypress == 27: # 'echap'
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
