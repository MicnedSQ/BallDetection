import cv2
import numpy as np

lower_yellow = np.array([20, 0, 190])
upper_yellow = np.array([30, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])


def detect_yellow_balls(frame, hsv):
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        img_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)

        img_yellow_gray = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2GRAY)

        _, binary_yellow = cv2.threshold(img_yellow_gray, 200, 255, cv2.THRESH_BINARY)

        kernel_erode = np.ones((5, 5), np.uint8)
        yellow = cv2.erode(binary_yellow, kernel_erode)
        kernel_dilate = np.ones((7, 7), np.uint8)
        yellow = cv2.dilate(yellow, kernel_dilate)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, 4, cv2.CV_32S)
        for i in range(1, num_labels):
            if stats[i, 4] > 70:
                cv2.putText(frame, "Yellow", (stats[i, 0], stats[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                cv2.rectangle(frame, (stats[i, 0], stats[i, 1]), (stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]), color=(0, 255, 255), thickness=2)
            


def detect_red_balls(frame, hsv):
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        img_red = cv2.bitwise_and(frame, frame, mask=mask_red)

        img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)

        _, binary_red = cv2.threshold(img_red_gray, 0, 255, cv2.THRESH_BINARY)

        kernel_erode = np.ones((5, 5), np.uint8)
        red = cv2.erode(binary_red, kernel_erode)
        kernel_dilate = np.ones((7, 7), np.uint8)
        red = cv2.dilate(red, kernel_dilate)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red, 4, cv2.CV_32S)
        for i in range(1, num_labels):
            cv2.putText(frame, "Red", (stats[i, 0], stats[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.rectangle(frame, (stats[i, 0], stats[i, 1]), (stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]), color=(0, 0, 255), thickness=2)
            


def main():
    cap = cv2.VideoCapture('Data/video_3.mp4')

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            detect_yellow_balls(frame, hsv)
            detect_red_balls(frame, hsv)

            cv2.imshow('Video Frame', frame)
            
            key = cv2.waitKey(10)

            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord(' '):
                continue
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()