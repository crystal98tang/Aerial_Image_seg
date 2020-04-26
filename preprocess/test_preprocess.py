import cv2
path = 'Da'
image = cv2.imread(path,cv2.IMREAD_COLOR)
cv2.imwrite('image.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, 50])
b,g,r = cv2.split(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
image = cv2.merge([b,g,r])
cv2.imwrite('clahe.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, 50])