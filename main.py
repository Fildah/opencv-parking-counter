import cv2
import pickle
import numpy as np
import urllib.request


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


url = "https://web.mmhk.cz/webcam/webcam.jpg"
img = url_to_image(url)

with open("polygons.pkl", "rb") as f:
    polygons = pickle.load(f)


def empty(a):
    pass


cv2.namedWindow("Vals")
cv2.resizeWindow("Vals", 640, 240)
cv2.createTrackbar("Val1", "Vals", 18, 50, empty)
cv2.createTrackbar("Val2", "Vals", 5, 50, empty)
cv2.createTrackbar("Val3", "Vals", 2, 50, empty)

def check_parking():
    for i, polygon in enumerate(polygons):
        # create a mask for the polygon
        mask = np.zeros(imgThres.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon)], (255, 255, 255))
        cropped_img = cv2.bitwise_and(imgThres, imgThres, mask=mask)
        # count white pixels in the mask
        white_pixels = cv2.countNonZero(cropped_img)
        # Put the number of white pixels in the polygon on the image
        cv2.putText(img, f'Polygon {i + 1}: {white_pixels}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        # Draw the polygon with color green if white pixels are less than 100, red otherwise
        color = (0, 255, 0) if white_pixels < 600 else (0, 0, 255)
        cv2.polylines(img, [np.array(polygon)], True, color, 3)


while True:
    img = url_to_image(url)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    val1 = cv2.getTrackbarPos("Val1", "Vals")
    val2 = cv2.getTrackbarPos("Val2", "Vals")
    val3 = cv2.getTrackbarPos("Val3", "Vals")
    if val1 % 2 == 0: val1 += 1
    if val3 % 2 == 0: val3 += 1
    imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, val1, val2)
    imgThres = cv2.medianBlur(imgThres, val3)
    kernel = np.ones((3, 3), np.uint8)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)

    check_parking()

    cv2.imshow("ImageThres", imgThres)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
exit()
