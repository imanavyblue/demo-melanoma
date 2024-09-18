import os
import wandb
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize W&B
wandb.init(project="melanoma", entity="suphawansr20-chiang-mai-university")

# Define constants
image_size = (224, 224)
batch_size = 32
epochs = 10
learning_rate = 0.0001

# Configure W&B
wandb.config = {
    "model": "InceptionV3",
    "input_shape": image_size + (3,),
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate
}

# Define data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'train_data'
validation_dir = 'validation_data'
test_dir = 'test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=image_size + (3,))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# WandB callbacks
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

metrics_logger = WandbMetricsLogger()
# WandB ModelCheckpoint for logging in WandB (with .keras format)
wandb_checkpoint = WandbModelCheckpoint(
    filepath='InceptionV3_epoch_{epoch:02d}.keras',  # เพิ่ม {epoch:02d}
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

# Custom callback to save model as .h5 after training
class SaveH5Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save model as .h5 at the end of each epoch
        if logs is not None and logs.get('val_loss') == min(logs.get('val_loss'), epoch):
            self.model.save('InceptionV3.h5')

save_h5_callback = SaveH5Callback()

# Train the model with both WandBModelCheckpoint and SaveH5Callback
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, metrics_logger, wandb_checkpoint, save_h5_callback]  # ใช้ callback บันทึก .keras และ .h5
)
# Evaluate the model
evaluation_results = model.evaluate(validation_generator)
val_loss, val_accuracy = evaluation_results[0], evaluation_results[1]

# Log evaluation results to W&B
wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy})

# Predict on validation data
predictions = model.predict(validation_generator)
predicted_classes = predictions.argmax(axis=1)
true_labels = validation_generator.classes

# Confusion matrix and classification report
cm = confusion_matrix(true_labels, predicted_classes)
accuracy = accuracy_score(true_labels, predicted_classes)
precision = precision_score(true_labels, predicted_classes, average='weighted')
recall = recall_score(true_labels, predicted_classes, average='weighted')
f1 = f1_score(true_labels, predicted_classes, average='weighted')

# Log metrics to W&B
wandb.log({
    'test_accuracy': accuracy,
    'test_precision': precision,
    'test_recall': recall,
    'test_f1_score': f1
})

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Log confusion matrix plot to W&B
wandb.log({"confusion_matrix": wandb.Image(plt)})
plt.close()
