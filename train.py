import os
import argparse
import wandb
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a melanoma classification model.")
    parser.add_argument("--project_name", type=str, default="melanoma", help="W&B project name.")
    parser.add_argument("--entity", type=str, default="suphawansr20-chiang-mai-university", help="W&B entity.")
    parser.add_argument("--model_name", type=str, default="InceptionV3", help="Name of the model architecture.")
    parser.add_argument("--train_dir", type=str, default="train_data", help="Directory for training data.")
    parser.add_argument("--val_dir", type=str, default="validation_data", help="Directory for validation data.")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (width and height).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--model_save_path", type=str, default="InceptionV3_best.h5", help="Path to save the best model.")
    return parser.parse_args()

def create_data_generators(args):
    """Creates and returns train and validation data generators."""
    image_shape = (args.image_size, args.image_size)

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

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=image_shape,
        batch_size=args.batch_size,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        args.val_dir,
        target_size=image_shape,
        batch_size=args.batch_size,
        class_mode='categorical'
    )
    return train_generator, validation_generator

def build_model(args, num_classes):
    """Builds and compiles the InceptionV3 model."""
    input_shape = (args.image_size, args.image_size, 3)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def train_model(model, train_generator, validation_generator, args):
    """Trains the model with specified callbacks."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True  # This is key for getting the best model
    )

    # The WandbModelCheckpoint will save the best model to W&B
    # We will also save it locally
    wandb_checkpoint = WandbModelCheckpoint(
        filepath=args.model_save_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, WandbMetricsLogger(), wandb_checkpoint]
    )
    return history

def evaluate_model(model, validation_generator):
    """Evaluates the model and logs metrics to W&B."""
    results = model.evaluate(validation_generator)
    val_loss, val_accuracy = results[0], results[1]
    wandb.log({'final_val_loss': val_loss, 'final_val_accuracy': val_accuracy})

    predictions = model.predict(validation_generator)
    predicted_classes = predictions.argmax(axis=1)
    true_labels = validation_generator.classes

    cm = confusion_matrix(true_labels, predicted_classes)
    accuracy = accuracy_score(true_labels, predicted_classes)
    precision = precision_score(true_labels, predicted_classes, average='weighted')
    recall = recall_score(true_labels, predicted_classes, average='weighted')
    f1 = f1_score(true_labels, predicted_classes, average='weighted')

    wandb.log({
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1_score': f1
    })

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=validation_generator.class_indices.keys(),
                yticklabels=validation_generator.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

def main():
    """Main function to run the training pipeline."""
    args = parse_args()

    wandb.init(
        project=args.project_name,
        entity=args.entity,
        config=vars(args)  # Log all command-line arguments
    )

    train_generator, validation_generator = create_data_generators(args)

    model = build_model(args, num_classes=train_generator.num_classes)

    train_model(model, train_generator, validation_generator, args)

    # Because restore_best_weights=True, the model object now holds the best weights.
    # We can now save this model. The WandbModelCheckpoint already saved it,
    # but an explicit save here makes it clearer.
    print(f"Saving the best model to {args.model_save_path}")
    model.save(args.model_save_path)
    wandb.save(os.path.basename(args.model_save_path)) # Save to W&B artifacts

    print("\nEvaluating the best model on the validation set...")
    evaluate_model(model, validation_generator)

    wandb.finish()
    print("\nTraining and evaluation complete.")

if __name__ == "__main__":
    main()
