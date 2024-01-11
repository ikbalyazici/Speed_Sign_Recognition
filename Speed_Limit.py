import cv2
import numpy as np

# Function to calculate the coordinates of rectangle from the circle coordinates
def get_rect_coords(circle):
    x, y, r = circle
    rn = 2*r / 3
    rn = np.round(rn).astype("int")
    rect = [(x - rn), (y - rn), (x + rn), (y + rn)]
    return rect

# Function to get the coordinates of the red circle
def get_red_circle_coords(img):
    # Blurring and converting to hsv
    img = cv2.GaussianBlur(img, (7, 7), 0)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Masking
    mask_low_red = cv2.inRange(image_hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_high_red = cv2.inRange(image_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask = mask_low_red + mask_high_red

    # Smoothing the edges
    mask = cv2.Canny(mask, 50, 100)
    mask = cv2.GaussianBlur(mask, (13, 13), 0)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=50)

    if circles is not None:
        return circles
    return None

def determine_number(ints):
    if (ints[1,0]==1 and ints[2,0]==1 and ints[3,0]==1 and
        ints[0,1]==1 and ints[1,1]==0 and ints[2,1]==0 and ints[3,1]==0 and 
        ints [4,1]== 1 and ints[1,2]==1 and ints[2,2]==1 and ints[3,2]==1):
        return 0
    elif(ints[1,0]==1 and ints[3,0]==1 and ints[0,1]==1 and
         ints[1,1]==0 and ints[2,1]==1 and ints[3,1]==0 and ints[4,1]==1 and
         ints[1,2]==1 and ints[3,2]==1):
        return 8
    elif(ints[1,0]==0 and ints[2,0]==0 and ints[3,0]==0 and 
         ints[4,0]==0 and ints[0,1]==1 and ints[0,2]==1 and ints[1,2]==1 and 
         ints[2,2]==1 and ints[3,2]==1 and ints[4,2]==1):
        return 1
    elif(ints[1,0]==0 and ints[2,0]==0 and ints[3,0]==0 and
         ints[0,1]==1 and ints[1,1]==0 and ints[3,1]==0 and ints[4,1]==1 and
         ints[1,2]==1 and ints[3,2]==1):
        return 3
    elif(ints[1,0]==1 and ints[3,0]==0 and ints[0,1]==1 and
         ints[1,1]==0 and ints[3,1]==0 and ints[4,1]==1 and
         ints[1,2]==0 and ints[2,2]==1 and ints[3,2]==1):
        return 5
    elif(ints[1,0]==0 and ints[0,1]==1 and ints[1,1]==0 and 
         ints[2,1]==1 and ints[3,1]==1 and ints[0,2]==1 and 
         ints[3,2]==0 and ints[4,2]==0):
        return 7
    elif(ints[2,0]==0 and ints[4,0]==1 and ints[0,1]==1 and 
         ints[1,1]==0 and ints[2,1]==0 and ints[3,1]==1 and ints[4,1]==1 and
         ints[1,2]==1 and ints[2,2]==1 and ints[3,2]==0 and ints[4,2]==1):
        return 2
    elif(ints[0,0]==0 and ints[1,0]==0 and ints[2,0]==1 and 
         ints[3,0]==1 and ints[4,0]==0 and ints[1,1]==1 and ints[2,1]==0 and
         ints[3,1]==1 and ints[3,2]==1):
        return 4
    elif(ints[1,0]==1 and ints[2,0]==1 and ints[3,0]==1 and
         ints[0,1]==1 and ints[3,1]==0 and ints[4,0]==1 and 
         ints[1,2]==0 and ints[2,2]==1 and ints[3,2]==1):
        return 6
    elif(ints[1,0]==1 and ints[2,0]==1 and ints[3,0]==0 and
         ints[0,1]==1 and ints[1,1]==0 and ints[3,1]==0 and ints[4,1]==1 and
         ints[1,2]==1 and ints[2,2]==1 and ints[3,2]==1 and ints[4,2]==0):
        return 9
    else:
        return -1
    

image = cv2.imread('tabela9.jpg', cv2.IMREAD_COLOR)

red_circle_coords = get_red_circle_coords(image)
if red_circle_coords is not None:
    for rcc in red_circle_coords[0]:
        rcc = np.round(rcc).astype("int")
        rect_coords = get_rect_coords(rcc)
        roi = image[rect_coords[1]:rect_coords[3], rect_coords[0]:rect_coords[2]].copy()

        bw_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(bw_roi, 70, 255, cv2.THRESH_BINARY_INV)

        # Konturları bul
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        # Her bir büyük kontur için dikdörtgen çiz
        old = -1
        number = -1
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            m= x
            p,r = np.round(w/3).astype("int"), np.round(h/5).astype("int") 
            ints = np.zeros((5, 3), dtype=int)
            i = 0
            
            while i < 3:
                n,j = y,0
                while j < 5:
                    cv2.rectangle(bw_roi, (m,n), (m+p, n+r),(255,0,0),1)
                    block = thresholded[n:n+r, m:m+p].copy()
                    intensity = cv2.mean(block)
                    if(intensity[0]>128):
                        ints[j,i]=1
                    else:   ints[j,i]=0
                    
                    n += r
                    j += 1

                m += p
                i += 1
                
            cv2.rectangle(bw_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            number = determine_number(ints)
            if number != -1:
                if old != -1:
                    number = old*10 + number
                old = number
            else:
                number = old

        if number != -1:
            frame = cv2.rectangle(image, (rect_coords[0], rect_coords[1]), (rect_coords[2], rect_coords[3]), (0, 255, 0), 4)
            # cv2.imshow('Frame', bw_roi)
            # cv2.waitKey(0)
            text_position = (rect_coords[0], rect_coords[1])    
        # Metni ekleyin
            cv2.putText(image, "Limit {}".format(number), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# OpenCV penceresini göster
cv2.imshow('Frame', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
