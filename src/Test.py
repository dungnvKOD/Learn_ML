import cv2

cap = cv2.VideoCapture('111.mp4')
# cap.set(3, 900)  # width
# cap.set(4, 900)  # height

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
while cap.isOpened():
    # chup tung khung hinh
    ret, frame = cap.read()
    # xu ly treen kung anh o day
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 3)
    # print(len(faces))
    # Display the resulting frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # print(roi_color)
        # print(x, y, w, h)
        if count < 100:
            img = frame[y:y + h, x:x + w]
            cv2.imwrite("data/frame%d.jpg" % count, img)  # save frame as JPEG file
            print(count)
            count = count + 1

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27 or 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
