import os
from keras import models, layers, optimizers, callbacks
from glob import glob
from typing import Dict, Tuple
import numpy as np
from keras import utils


from config import PROJECT_DIR, datasets_dir
from data_vectorization import ImageVectorizer

def collect_labeled_images_from_dir(dir_path):
    # Directory_sentiment map
    dir_sentiment_map = {
        'happy': 0,
        'sad': 1,
        'fear': 2,
        'surprise': 3,
        'neutral': 4,
        'angry': 5,
        'disgust': 6,
    }
    labeled_images = {}
    for sentiment, label in dir_sentiment_map.items():
        sentiment_img_paths = os.path.join(dir_path, sentiment)
        for img_path in glob(os.path.join(sentiment_img_paths, '*jpg')):
            labeled_images[img_path] = label
    return labeled_images


def extract_features_and_labels(
    labeled_images: Dict[str, int], vectorizer: ImageVectorizer
)-> Tuple[np.ndarray, np.ndarray]:
    image_paths = list(labeled_images.keys())
    labels = list(labeled_images.values())
    # Extract features
    image_features = vectorizer.extract_features_from_images(image_paths)
    one_hot_labels = utils.to_categorical(labels, num_classes=len(set(labels)))
    return image_features, one_hot_labels    

class FaceSentimentCNNModel:
    def __init__(self, image_shape, num_sentiments):
        self.image_shape = image_shape
        self.num_sentiments = num_sentiments
        self.model = self._build_model()
        self._compile_model()
        
    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.image_shape),
            
            layers.Conv2D(
                32, kernel_size=(3, 3), padding='same', 
                kernel_initializer='glorot_uniform', bias_initializer='zeros'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(
                64, kernel_size=(3, 3), padding='same', 
                kernel_initializer='glorot_uniform', bias_initializer='zeros'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(
                128, kernel_size=(3, 3), padding='same', 
                kernel_initializer='glorot_uniform', bias_initializer='zeros'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(
                256, kernel_size=(3, 3), padding='same', 
                kernel_initializer='glorot_uniform', bias_initializer='zeros'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2D(
                256, kernel_size=(3, 3), padding='same', 
                kernel_initializer='glorot_uniform', bias_initializer='zeros'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Dropout(0.5),
            
            layers.Flatten(),
            
            layers.Dense(
                512, kernel_initializer='glorot_uniform', 
                bias_initializer='zeros'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(
                self.num_sentiments, kernel_initializer='glorot_uniform', 
                bias_initializer='zeros'
            ),
            layers.Activation('softmax')
        ])
        return model

    def _compile_model(self):
            fer_optimizer = optimizers.Adam(learning_rate=0.001)
            self.model.compile(
                optimizer=fer_optimizer, loss='categorical_crossentropy', 
                metrics=['accuracy']
            )
    
    def get_model(self):
        return self.model

if __name__ == '__main__':
    # Directories of FER2013 test and train datasets
    fer_train_dir = os.path.join(datasets_dir, 'FER2013', 'train')
    fer_test_dir = os.path.join(datasets_dir, 'FER2013', 'test')
    # Dimensions per image
    dims = (256, 256)
    img_vectorizer = ImageVectorizer(dims=dims)
    # make dictionaries for images_paths with sentiments
    images_to_test = collect_labeled_images_from_dir(fer_test_dir)
    images_for_training = collect_labeled_images_from_dir(fer_train_dir)
    # Extract features and one-hot labels for the 7 classes
    X_train, y_train = extract_features_and_labels(
        images_for_training, img_vectorizer
    )
    X_test, y_test = extract_features_and_labels(
        images_to_test, img_vectorizer
    )
    image_shape = (dims[0], dims[1], 3)
    batch_size = 32
    epochs = 30
    num_sentiments = 7
    face_sentiment_model = FaceSentimentCNNModel(image_shape, num_sentiments)
    model = face_sentiment_model.get_model()
    checkpoint_dir = os.path.join(PROJECT_DIR, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    cp = callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'face_modelCNN.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    # Train the model
    model.fit(
        X_train, y_train, batch_size=batch_size, epochs=epochs, 
        validation_data=(X_test, y_test), shuffle=True, callbacks=cp
    )
