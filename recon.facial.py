import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

video = cv2.VideoCapture('retrato.jpg') 

while True:

    succesful_frame_read, frame = video.read()

    if not succesful_frame_read:
        break 

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.7, minNeighbors=20)

    smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.7, minNeighbors=20)

    eyes = eye_detector.detectMultiScale(frame_grayscale)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 2)
    
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):

        break

video.release()
cv2.destroyAllWindows()

print('Funcionando')