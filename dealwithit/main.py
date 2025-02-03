import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

glasses = cv2.imread("dealwithit.png", -1)
glasses = cv2.resize(glasses, (220, 90))

camera = cv2.VideoCapture(0)

lbp_cascade = "lbpcascades/lbpcascade_frontalface.xml"
haar_cascade = "haarcascades/haarcascade_eye_tree_eyeglasses.xml"

eyes_classifier = cv2.CascadeClassifier(haar_cascade)
face_classifier = cv2.CascadeClassifier(lbp_cascade)


def detector(img, classifier):
    result = img.copy()
    rects = classifier.detectMultiScale(result, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in rects:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255))
    return result


while True:
    ret, frame = camera.read()

    faces = face_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        face_roi = frame[fy:fy + fh, fx:fx + fw]
        eyes_rects = eyes_classifier.detectMultiScale(face_roi)
        eyes = []
        ex_min = np.inf

        for (ex, ey, ew, eh) in eyes_rects:
            ex_full, ey_full = fx + ex, fy + ey
            if ex_full < ex_min:
                ex_min = ex_full
                eyes = [(ex, ey, ew, eh)]

        if len(eyes) > 0:
            ex, ey, ew, eh = eyes[0]
            ex_full, ey_full = fx + ex, fy + ey
            mask = glasses[..., 3]
            glasses_rgb = glasses[..., :3]

            y1, y2 = ey_full, ey_full + glasses_rgb.shape[0]
            x1, x2 = ex_full, ex_full + glasses_rgb.shape[1]

            if x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                for c in range(3):
                    frame[y1-20:y2-20, x1-50:x2-50, c] = ((1 - mask / 255.0) * frame[y1-20:y2-20, x1-50:x2-50, c] + (mask / 255.0) * glasses_rgb[..., c])
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
      
camera.release()
cv2.destroyAllWindows()
