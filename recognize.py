import cv2
import click
import os
import numpy as np
from sklearn.metrics import pairwise

background = None # at launch, the background is not defined yet

# add dx, dy to the tuple xy
def add(xy, dx, dy):
    x, y = xy # unpack tuple
    return (x+dx, y+dy)

# compute the background by averaging frames
def run_avg(frame, weight):
    global background
    if background is None: # initialization
        background = frame.copy().astype('float')
        return
    cv2.accumulateWeighted(background, background, weight) # https://docs.opencv.org/

# isolates features from the background
def segment(image, threshold):
    global background
    diff = cv2.absdiff(image, background.astype(np.uint8))
    retval, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((13,13),np.uint8)
    threshold = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # closing function
    contours, retval = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    segmented = max(contours, key=cv2.contourArea)
    return (threshold, segmented)

@click.command()
@click.option('-dbg', '--display-background', is_flag=True, help='Display the background which has been computed')
@click.option('-dth', '--display-thresold', is_flag=True, help='Display the threshold which has been computed')
@click.option('-dcroi', '--display-circular-roi', is_flag=True, help='Display the circular roi which has been computed')
def main(display_background, display_thresold, display_circular_roi):

    global background
    capture = cv2.VideoCapture(0)
    left, right, top, bottom = 300, 680, 0, 400 # coordinates of the ROI (Region Of Interest)
    num_frames = 0

    while True:

        keypress = cv2.waitKey(1) # collect user's keyboard inputs

        _, frame = capture.read()
        frame = cv2.flip(frame, 1) # vertical mirror
        clone = frame.copy()
        roi = frame[top:bottom, left:right] # focus on the region of interest
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(gray, (7,7), 0) # kernel of 7x7 pixels, standard deviation equal to 0 for both X and Y directions

        # compute the avergae of the first 30st frames to define the background
        if num_frames < 30:
            if num_frames == 0:
                print('Computation of the background...')
            if num_frames == 29:
                print('The background is defined')
            run_avg(smoothed, weight=0.5)
        else:
            # now that the background has been defined, the hand can be isolate from the background
            hand = segment(smoothed, threshold=20)
            if hand is not None:
                threshold, segmented = hand # unpack the information
                cv2.drawContours(clone, [segmented+(left,top)], -1, color=(0, 0, 255), thickness=1) # add the contours of the hand on the 'clone' image
                if display_thresold: # show the 'threshold' image (binarisation + closing)
                    cv2.imshow('Closing', threshold)

                hull = cv2.convexHull(points=segmented) # find the hand's hull (matrix of coordinates)
                cv2.drawContours(clone, [hull+(left,top)], -1, color=(0, 255, 0), thickness=2) # add hull to the 'clone' image
                # find the coordinates of extrem points of the convex hull
                extrem_left = add(tuple(hull[hull[:,:,0].argmin()][0]), left, top) # add() because of the ROI
                extrem_right = add(tuple(hull[hull[:,:,0].argmax()][0]), left, top)
                extrem_top = add(tuple(hull[hull[:,:,1].argmin()][0]), left, top)
                extrem_bottom = add(tuple(hull[hull[:,:,1].argmax()][0]), left, top)
                # draw them on the clone figure
                cv2.circle(clone, extrem_left, radius=5, color=(255,0,0), thickness=5)
                cv2.circle(clone, extrem_right, radius=5, color=(255,0,0), thickness=5)
                cv2.circle(clone, extrem_top, radius=5, color=(255,0,0), thickness=5)
                cv2.circle(clone, extrem_bottom, radius=5, color=(255,0,0), thickness=5)
                # find the center of the circle
                cx, cy = (extrem_left[0] + extrem_right[0])//2, (extrem_top[1] + extrem_bottom[1])//2
                # draw it on the clone figure
                cv2.circle(clone, (cx,cy), radius=2, color=(255,0,0), thickness=5)
                # find the maximum distance between center of the palm and farest point on the hull
                distances = pairwise.euclidean_distances([(cx, cy)], Y=[extrem_top, extrem_right, extrem_left, extrem_bottom])[0]
                max_distance = np.max(distances)
                radius = int(0.7* max_distance) # radius of a 80% max euclidan distance found radius
                circumference = 2 * np.pi * radius
                cv2.circle(clone, (cx,cy), radius=radius, color=(255,0,0), thickness=2)

                circular_roi = np.zeros(np.shape(threshold), dtype=np.uint8) # initilisation
                cv2.circle(circular_roi, (cx-left, cy-top), radius=radius, color=(255,255,255), thickness=2)
                circular_roi = cv2.bitwise_and(threshold, threshold, mask=circular_roi)

                # count the number of fingers raised
                counts = 0

                contours, retval = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour) # compute the bounding box of the contour
                    cv2.rectangle(circular_roi, (x,y), (x+w, y+h), (255,255,255), 2) # plot the boxes
                    cv2.putText(circular_roi, 'C' + str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 4)

                    # increment the count of fingers only if:
                    # 1. The contour region is not the wrist (bottom area)
                    # 2. The number of points along the contour does not exceed 25% of the circumference of the circular ROI
                    if ((cy + (cy * 0.22)) > (y + h)) and ((circumference * 0.25) > contour.shape[0]):
                        counts += 1

                cv2.putText(clone, str(counts), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if display_circular_roi:
                    cv2.imshow('Circular ROI', circular_roi)

                if keypress == ord('s'):
                    if not os.path.exists('./captures'):
                        os.mkdir('./captures')
                    print('Screenshot has been captured')
                    cv2.imwrite('./captures/frame.png', frame)
                    cv2.imwrite('./captures/background.png', background.astype(np.uint8))
                    cv2.imwrite('./captures/clone.png', clone)
                    cv2.imwrite('./captures/roi.png', roi)
                    cv2.imwrite('./captures/threshold.png', threshold)
                    cv2.imwrite('./captures/threshold.png', circular_roi)

        num_frames += 1

        cv2.imshow('Base environment', clone) # non-treated frame
        #cv2.imshow('Treated environment', smoothed) # pre-treated frame
        if display_background:
            cv2.imshow('Background', background.astype(np.uint8))

        if keypress == 27: # 'echap'
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
