import cv2
import numpy as np
from tensorflow import keras

(train_X,train_y),(test_X,test_y) = keras.datasets.mnist.load_data()

# 28x28 -> 20x20 사이즈변경 / (60000, 28, 28)
train_X = np.array([cv2.resize(img,(20,20)) for img in train_X])
test_X = np.array([cv2.resize(img,(20,20)) for img in test_X])

# reshape -> (60000, 20, 20, 1)
train_X = train_X.reshape(-1,20,20,1).astype(np.float32) / 255.0
test_X = test_X.reshape(-1,20,20,1).astype(np.float32) / 255.0

# print(train_X.shape)

#cnn 모델
model = keras.Sequential([
    keras.Input(shape=(20,20,1)),
    keras.layers.Conv2D(32, kernel_size=3,padding='same',activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(63, kernel_size=3,padding='same',activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
model.summary()

early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model.fit(train_X,train_y, epochs=20, validation_split=0.2,callbacks=early_stop)

loss,acc = model.evaluate(test_X,test_y)
print(f'정확도 : {acc}')

#모델저장
model.save("data/mnist_cnn.keras")
print("모델저장완료 : data/mnist_cnn.keras")