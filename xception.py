import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import os

# 🔢 Eğitim epoch sayısı (sadece buradan değiştir)
EPOCHS = 20  # örnek: 5, 10, 15, 20 gibi

# 📁 1. Klasör yolları
base_dir = r"C:\Users\Huaweı\OneDrive\Masaüstü\veri_ayrilmis"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# 🧼 2. Görselleri normalleştirme ve veri artırma
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)

# 📦 3. Verileri yükle
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 🧠 4. Xception modeli
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# 🔧 5. Üst katmanlar
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 🔓 6. Fine-tuning için sadece son 20 katmanı eğitilebilir yap
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# ⚙️ 7. Derleme
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🚀 8. Eğitim (EarlyStopping yok)
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# 📊 9. Eğitim grafiği
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Başarı Grafiği')
plt.legend()
plt.show()

# 🔍 10. Precision, Recall, F1-Score Hesaplama
Y_pred = model.predict(val_data)
y_pred = np.argmax(Y_pred, axis=1)

# Gerçek etiketler
y_true = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Sınıflandırma raporu
report = classification_report(y_true, y_pred, target_names=class_labels)
print("Sınıflandırma Raporu:\n")
print(report)

# 💾 11. Modeli kaydet
model.save("xception_model_finetuned.h5")
print("Model başarıyla kaydedildi.")
