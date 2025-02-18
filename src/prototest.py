import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    # out.write(frame)

    # Display the captured frame
    # cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # adaptive thresholding to use different threshold 
    # values on different regions of the frame.
    Thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow('Thresh', Thresh)
    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()