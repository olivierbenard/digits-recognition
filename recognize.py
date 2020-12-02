import cv2
import click
import numpy as np

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
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    segmented = max(contours, key=cv2.contourArea)
    return (closed, segmented)

# count number of digits the hand is holding
def count(threshold, segmented): # args = (binarized image, contours)
    hull = cv2.convexHull(points=segmented)
    return hull

@click.command()
@click.option('-dbg', '--display-background', is_flag=True, help='Display the background which has been computed')
@click.option('-dth', '--display-thresold', is_flag=True, help='Display the threshold which has been computed')
def main(display_background, display_thresold):

    global background
    capture = cv2.VideoCapture(0)
    left, right, top, bottom = 300, 680, 0, 480 # coordinates of the ROI (Region Of Interest)
    num_frames = 0

    while True:

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
                closed, segmented = hand # unpack the information
                cv2.drawContours(clone, [segmented+(left,top)], -1, color=(0, 0, 255), thickness=1) # add the contours of the hand on the 'clone' image
                if display_thresold: # show the 'closed' image (binarisation + closing)
                    cv2.imshow('Closing', closed)
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
                cv2.circle(clone, (cx,cy), radius=2, color=(255,255,0), thickness=5)


        num_frames += 1

        cv2.imshow('Base environment', clone) # non-treated frame
        #cv2.imshow('Treated environment', smoothed) # pre-treated frame
        if display_background:
            cv2.imshow('Background', background.astype(np.uint8))

        keypress = cv2.waitKey(1)

        if keypress == 27: # 'echap'
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
