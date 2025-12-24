import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


def load_data(folder_path, labels_df):
    data = []
    labels = []

    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(subdir, file)

            # Extract patient_id from the file name
            patient_id = int(file.split('_')[0])

            # Get the corresponding label from the Excel sheet
            label = labels_df.loc[labels_df['patient_id'] == patient_id, 'case_csPCa'].values

            if len(label) == 1:  # Ensure there's a corresponding label
                # Read .mha file using SimpleITK
                image = sitk.ReadImage(file_path)
                image_array = sitk.GetArrayFromImage(image)

                data.append(image_array)
                labels.append(label[0])

    return np.array(data), np.array(labels)


def preprocess_data(data, labels):
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Convert labels to categorical format
    labels_one_hot = to_categorical(labels_encoded)

    # Normalize and reshape data
    data = data.astype('float32') / 255.0

    return data, labels_one_hot


def build_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


if __name__ == "__main__":
    # Set your paths
    train_folder = r'E:\ml_train_images'
    test_folder = r'E:\ml_test_images'
    excel_path = r'path\to\marksheet.xlsx'  # Update with the actual path

    # Load labels from Excel sheet
    labels_df = pd.read_excel(excel_path)

    # Load and preprocess training data
    train_data, train_labels = load_data(train_folder, labels_df)
    train_data, train_labels = preprocess_data(train_data, train_labels)

    # Load and preprocess testing data
    test_data, test_labels = load_data(test_folder, labels_df)
    test_data, test_labels = preprocess_data(test_data, test_labels)

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,
                                                                      random_state=42)

    # Define input shape and number of classes
    input_shape = train_data[0].shape
    num_classes = len(np.unique(train_labels))

    # Build the CNN model
    model = build_cnn(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

    # Define checkpoint to save the best model during training
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

    # Train the model
    history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels),
                        callbacks=[checkpoint])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
