import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('data/mnist_cnn.keras')
print("모델 로드 완료")


# 1. 이미지 읽기
img = cv2.imread("img/numbers.jpg")

# 2. 컬러 → 흑백
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur 적용
g_img = cv2.medianBlur(g_img,5)



# 3. 이진화 (적응형)
bin_img = cv2.adaptiveThreshold(
    g_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)



# 3. 이진화 (적응형)
# thr, bin_img = cv2.threshold(g_img, 110, 255, cv2.THRESH_BINARY_INV)


# 몰핀
kernel = np.ones((2,2), np.uint8)
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)




# 4. 컨투어 찾기
contours, hierarchy = cv2.findContours(
    bin_img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

h, w = bin_img.shape[:2]

for contour in contours:

    (x, y), radius = cv2.minEnclosingCircle(contour)

    if radius > 3:
        xs = max(0, int(x-radius))
        xe = min(w, int(x+radius))
        ys = max(0, int(y-radius))
        ye = min(h, int(y+radius))

        cv2.rectangle(bin_img, (xs, ys), (xe, ye), (255,255,255), 1)

        roi = bin_img[ys:ye, xs:xe]
        if roi.size == 0:
            continue

        dst = cv2.resize(roi, (50, 50))
        dst = cv2.resize(dst, (16, 16))

        A = np.zeros((20, 20))
        A[2:-2, 2:-2] = dst

        # CNN 입력: (1, 20, 20, 1), 정규화
        A = A.reshape(1, 20, 20, 1).astype(np.float32) / 255.0

        pred = model.predict(A, verbose=0)
        num = np.argmax(pred)

        cv2.putText(
            bin_img,
            str(num),
            (xs, ys),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (200, 0, 0)
        )

cv2.imshow("Image", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()