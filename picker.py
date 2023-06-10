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
points = []
polygons = []


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) >= 4:
            polygons.append(points.copy())
            cv2.polylines(img, [np.array(points)], True, (0, 255, 0), 3)  # color green
            points.clear()


cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

show_polygons = True

while True:
    img_copy = img.copy()
    if show_polygons:
        for polygon in polygons:
            cv2.polylines(img_copy, [np.array(polygon)], True, (0, 255, 0), 3)
    for i, p in enumerate(points):
        cv2.circle(img_copy, p, 5, (0, 0, 255), -1)
        cv2.putText(img_copy, str(i), (p[0] + 10, p[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imshow("image", img_copy)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == 8:  # backspace key
        if points:
            points.pop()
    elif key == ord("h"):  # hide polygons
        show_polygons = not show_polygons

# Save the polygons list to a file
with open("polygons.pkl", "wb") as f:
    pickle.dump(polygons, f)

cv2.destroyAllWindows()
exit()
