import cv2

# Load the Haar cascade file for luggage detection
cascade = cv2.CascadeClassifier("week-5-haar/haar-luggage-cascade.xml")

# Open the video file
cap = cv2.VideoCapture("week-5-haar/test-video.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    # Read the next frame from the video
    ret, frame = cap.read()

    # If the frame was read correctly
    if ret:
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform object detection
        luggages = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Draw rectangles around detected objects
        for x, y, w, h in luggages:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with detections
        cv2.imshow("Luggage Detection", frame)

        # Press 'q' to exit the video early
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
