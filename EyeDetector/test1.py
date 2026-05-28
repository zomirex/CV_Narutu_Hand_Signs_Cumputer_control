import cv2

# ۱. بارگذاری cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ۲. خواندن تصویر (باید به صورت grayscale باشد)
img = cv2.imread('photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ۳. تشخیص چشم‌ها
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

# ۴. رسم مستطیل اطراف چشم‌ها
for (x,y,w,h) in eyes:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('EYES', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
