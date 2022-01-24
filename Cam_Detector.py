import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#   0 - default camera    meme.mp4'
webcam = cv2.VideoCapture(0)

while True:
    scale = 1.3
    #    True or false      webcam  Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 256, 0), 3)
        # Blur the rectangle
        # Median blur
        #frame[y:y+h, x:x+w] = cv2.medianBlur(frame[y:y+h, x:x+w], 29)
        # Normal blur
        frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (30, 30))

    cv2.imshow("Imisuu's face detection", frame)
    key = cv2.waitKey(1)

    # Stop when Q is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()
