import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping

# ✅ Use raw string (r"...") to prevent Windows path errors
dataset_path = r'C:\Users\saksa\Desktop\NUM\ITSATSUN\Year4\AI\Dataset\HandwrittenKhmerDigit\train'

# ✅ Load dataset (make sure folder contains subfolders named 0,1,...,9)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    color_mode="grayscale",
    image_size=(28, 28),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    color_mode="grayscale",
    image_size=(28, 28),
    batch_size=32
)

# ✅ Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ✅ Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)  # Output layer: 10 classes
])

# ✅ Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ✅ Define EarlyStopping callback to stop training when validation accuracy plateaus
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the best epoch
    verbose=1  # Print a message when early stopping occurs
)

# ✅ Train the model with EarlyStopping
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # Set a high number of epochs, but early stopping will stop it when necessary
    callbacks=[early_stopping]  # Include early stopping callback
)

# ✅ Save the trained model
model.save("digit_model.keras")
print("✅ Model training complete and saved to 'digit_model.keras'")

# ✅ Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

# ✅ Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# ✅ Evaluate the model
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc*100:.2f}%")