import cv2
import imutils

# Load Haar Cascade XML
cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)

# Open camera (0 = default laptop webcam)
cam = cv2.VideoCapture(0)

# Check if camera opened
if not cam.isOpened():
    print("❌ Error: Camera not opened")
    exit()

while True:
    ret, img = cam.read()

    # If frame not received
    if not ret or img is None:
        print("❌ Error: Failed to grab frame")
        break

    # Resize frame
    img = imutils.resize(img, width=1000)

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw rectangles
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Car Detection", img)

    # Traffic logic
    n = len(cars)
    print("------------------------------------")
    print("Vehicles Detected:", n)

    if n >= 8:
        print("Traffic Status: HIGH")
        print("Signal Action: Required")
    else:
        print("Traffic Density: Low")
        print("Decision: No Signal Required")

    # Press ESC to exit
    if cv2.waitKey(50) == 27:
        break

cam.release()
cv2.destroyAllWindows()
