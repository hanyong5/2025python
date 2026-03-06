import cv2
import numpy as np
from tensorflow import keras

# 저장된 모델 로드
model = keras.models.load_model("data/mnist_cnn.keras")
print("모델 로드 완료 - 카메라 시작 (q: 종료)")

# 카메라 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 흑백 변환
    g_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g_img = cv2.medianBlur(g_img, 5)

    # 이진화 (적응형)
    bin_img = cv2.adaptiveThreshold(
        g_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 컨투어 찾기
    contours, _ = cv2.findContours(
        bin_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = bin_img.shape[:2]
    result = frame.copy()

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)

        if radius > 10:   # 너무 작은 노이즈 제거 (카메라용으로 크게)
            xs = max(0, int(x - radius))
            xe = min(w, int(x + radius))
            ys = max(0, int(y - radius))
            ye = min(h, int(y + radius))

            roi = bin_img[ys:ye, xs:xe]
            if roi.size == 0:
                continue

            dst = cv2.resize(roi, (50, 50))
            dst = cv2.resize(dst, (16, 16))

            A = np.zeros((20, 20))
            A[2:-2, 2:-2] = dst
            A = A.reshape(1, 20, 20, 1).astype(np.float32) / 255.0

            pred = model.predict(A, verbose=0)
            conf = np.max(pred)
            num = np.argmax(pred)

            # 신뢰도 80% 이상만 표시
            if conf > 0.8:
                cv2.rectangle(result, (xs, ys), (xe, ye), (0, 255, 0), 2)
                cv2.putText(
                    result,
                    f"{num}({conf:.0%})",
                    (xs, ys - 5),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Camera - 숫자 인식 (q: 종료)", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
