import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Veri klasörleri (manuel ayırdığın train/test klasörleri)
train_dir = "data/train"
test_dir = "data/test"
img_size = (224, 224)
batch_size = 32
num_classes = 4

# Veri artırma ve normalizasyon
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim verisi
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Test (doğrulama) verisi
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# VGG16 modelini yükle
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Yeni sınıflandırıcı ekle
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Modeli derle
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Erken durdurma tanımla
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğit (burada test_data validation olarak kullanılıyor)
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5,
    # callbacks=[early_stop],
    verbose=1
)

# Modelin değerlendirilmesi
print("\nEvaluating model on test data...")

# Tahminleri al
test_data.reset()
preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes

# Sınıf adlarını al
class_labels = list(test_data.class_indices.keys())

# Sınıflandırma raporu
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
conf_mat = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n")
print(conf_mat)

# Doğruluk grafiği
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.title('Model Doğruluğu (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)
plt.show()
