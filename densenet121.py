import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# 📁 1. Klasör yolları
base_dir = r"C:\Users\Huaweı\OneDrive\Masaüstü\veri_ayrilmis"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# 📦 2. Görüntü verisi ayarları
batch_size = 32
image_size = (224, 224)  # DenseNet121 için ideal giriş boyutu

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())

# 🧠 3. DenseNet121 tabanlı model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # İlk aşamada özellik çıkarıcı olarak kullan

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ⚙️ 4. Derleme
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 🏋️ 5. Eğitim
history = model.fit(
    train_generator,
    epochs=10,  # Daha hızlı bitmesi için epoch sayısı azaltıldı
    validation_data=val_generator
)

# 💾 6. Modeli Kaydet
model.save("densenet121_custom.h5")

# 📊 7. Eğitim grafiği
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Başarı Grafiği (DenseNet121)')
plt.legend()
plt.grid(True)
plt.show()

# 📈 8. Değerlendirme
val_generator.reset()
pred_probs = model.predict(val_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_generator.classes

print("📋 Sınıf Etiketleri:", class_labels)

print("\n🧮 Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

cm = confusion_matrix(y_true, y_pred)
print("🔍 Confusion Matrix:\n", cm)
