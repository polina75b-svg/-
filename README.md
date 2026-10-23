# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Загружаем набор данных MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Нормализуем изображения
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Создаём нейросеть
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Компилируем модель
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Обучаем нейросеть
print("Начинаем обучение нейросети...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 6. Проверяем точность
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Точность нейросети: {accuracy * 100:.2f}%")

# 7. Сохраняем обученную модель
model.save("digit_recognition_model.keras")

print("Нейросеть успешно сохранена!")

# 8. Проверяем распознавание цифры
index = np.random.randint(0, len(x_test))

image = x_test[index]

prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)

predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Распознанная цифра: {predicted_digit}")
print(f"Уверенность: {confidence:.2f}%")
print(f"Правильная цифра: {y_test[index]}")

# 9. Показываем изображение
plt.imshow(image, cmap='gray')
plt.title(f"Нейросеть распознала: {predicted_digit}")
plt.axis('off')
plt.show()