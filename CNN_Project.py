import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Veri setinin yolu
train_data_dir = 'C:/Users/Administrator/Desktop/dataset3/train'  # Eğitim seti
test_data_dir = 'C:/Users/Administrator/Desktop/dataset3/test'      # Test seti

# Parametreler
img_width, img_height = 150, 150
batch_size = 16

# Veri artırma ve veri setini yükleme
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    validation_split=0.2  # Eğitim verisini ikiye böl
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # Çoklu sınıf etiketleri
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # Çoklu sınıf etiketleri
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)



# CNN Modeli
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Çıkış katmanı: 'sigmoid' aktivasyon fonksiyonu
model.add(Dense(5, activation='sigmoid'))  # 5 etiket için çıkış

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10  # Epoch sayısını ihtiyaca göre ayarlayın
)

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def load_and_predict(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    return predictions

# Örnek bir görüntü ile tahmin
img_path = 'test_image14.jpg'
predictions = load_and_predict(img_path)

# Tahmin sonuçlarını yazdırma
airplane_prob = predictions[0][0]
bicycle_prob = predictions[0][1]
car_prob = predictions[0][2]
garbage_truck_prob = predictions[0][3]
ship_prob = predictions[0][4]
print(f"Airplane Probability: {airplane_prob:.2f}, Bicycle Probability: {bicycle_prob:.2f}, Car Probability: {car_prob:.2f}, Garbage Truck Probability: {garbage_truck_prob:.2f}, Ship Probability: {ship_prob:.2f}")

# Belirli bir eşik değeri ile etiketleme
threshold = 0.5
airplane_label = airplane_prob > threshold
bicycle_label = bicycle_prob > threshold
car_label = car_prob > threshold
garbage_truck_label = garbage_truck_prob > threshold
ship_label = ship_prob > threshold

print("Airplane detected:" if airplane_label else "Airplane not detected.")
print("Bicycle detected:" if bicycle_label else "Bicycle not detected.")
print("Car detected:" if car_label else "Car not detected.")
print("Garbage Truck detected:" if garbage_truck_label else "Garbage Truck not detected.")
print("Ship detected:" if ship_label else "Ship not detected.")
