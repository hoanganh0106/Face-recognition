import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report
import numpy as np

#Bước 1.2: Tiền xử lý và tổ chức dữ liệu
image_height, image_width = 224, 224
batch_size = 16
data_dir = 'data'



train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(image_height, image_width),
    batch_size=batch_size
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(image_height, image_width),
    batch_size=batch_size
)
class_names = train_dataset.class_names
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#Bước 1.3: Tăng cường dữ liệu (Data Augmentation)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = (image / 127.5) - 1
    return image, label

train_dataset_augmented = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset_rescaled = validation_dataset.map(lambda x, y: ((x / 127.5) - 1, y), num_parallel_calls=tf.data.AUTOTUNE)


#Bước 1.4: Xây dựng kiến trúc mô hình (đóng băng backbone)
base_model = MobileNetV2(input_shape=(image_height, image_width, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(image_height, image_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)


#Bước 1.5: Huấn luyện mô hình – Giai đoạn 1
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]
print("\n[STAGE 1] Training with frozen base model...")
model.fit(train_dataset_augmented, validation_data=validation_dataset_rescaled, epochs=30, callbacks=callbacks)


#Bước 1.6: Fine-tuning mô hình – Giai đoạn 2 (mở một phần backbone)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
    
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks_ft = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]
print("\n[STAGE 2] Fine-tuning last 20 layers...")
model.fit(train_dataset_augmented, validation_data=validation_dataset_rescaled, epochs=30, callbacks=callbacks_ft)


#Bước 1.7: Lưu mô hình đã huấn luyện
model.save('face_recognizer.h5')
print("\nĐã lưu mô hình: face_recognizer.h5")
print(f"Class names: {class_names}")
print(f"   - {class_names[0]} = label 0 (score < 0.5)")
print(f"   - {class_names[1]} = label 1 (score >= 0.5)")
