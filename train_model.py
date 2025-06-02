import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# ==== 1. Paths ====
train_dir = 'dataset/archive/train'
test_dir = 'dataset/archive/test'
model_save_dir = 'model_vgg19'
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, 'emotion_model_vgg19.h5')

# ==== 2. Image Parameters ====
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 100
NUM_CLASSES = 7

# ==== 3. Data Generators ====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ==== 4. Transfer Learning Model ====
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==== 5. Callbacks ====
checkpoint = ModelCheckpoint(
    model_path, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)

# ==== 6. Train ====
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

# ==== 7. Evaluate ====
print("\nEvaluating on test set:")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ==== 8. Plot History ====
def plot_history(history_obj, save_dir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_obj.history['accuracy'], label='Train Accuracy')
    plt.plot(history_obj.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "training_accuracy_plot.png"))

    plt.subplot(1, 2, 2)
    plt.plot(history_obj.history['loss'], label='Train Loss')
    plt.plot(history_obj.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "training_loss_plot.png"))
    plt.close()

plot_history(history, model_save_dir)
print(f"✅ Training plots saved to {model_save_dir}")
