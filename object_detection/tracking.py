import cv2 as cv

cap = cv.VideoCapture(0)


while True:
    ret, frame = cap.read()

    resized = cv.resize(frame, (600, 400))

    cv.imshow('resized', resized)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()