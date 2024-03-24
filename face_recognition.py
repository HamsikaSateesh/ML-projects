import cv2
fac_cap=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vc=cv2.VideoCapture(0)
while True:
    ret , vid = vc.read()
    col = cv2.cvtColor(vid,cv2.COLOR_BAYER_BG2GRAY)
    fac=fac_cap.detectMultiScale(
        col,
        scaleFactor=1.1
        minNeighbors=5,
        minSize=(30, 30)
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in fac:
        cv2.rectangle(vc, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("live-video",vid)
    if cv2.waitkey(10) == ord("a"):
        break
    vc.release()
