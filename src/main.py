import os
import numpy as np
from keras import models

from config import PROJECT_DIR, datasets_dir
from data_vectorization import GloveVectorizer, ImageVectorizer

MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
PICKLED_GLOVE_DIR = datasets_dir

# model paths
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_text.keras')
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'facemodelCNN.keras')

def is_image_file(file_path):
    """Check if a file is an image"""
    return any(
        file_path.endswith(ext) 
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    )

def load_text_model(text_model_path):
    """Loads the pretrained model on tweets"""
    return models.load_model(text_model_path)

def load_image_model(image_model_path):
    return models.load_model(image_model_path)

def predict_sentiment(input_data):
    """
    Predict the sentiment of the input data.
    Parameters:
    input_data (str): Path to an image file or a text string.
    Returns:
    str: Predicted sentiment.
    """
    if os.path.exists(input_data) and is_image_file(input_data):
        image_vectorizer = ImageVectorizer(dims=(256, 256))
        image_array = image_vectorizer.convert_image_to_array(input_data)
        # Add batch dim
        image_array.expand_dims(image_array, axis=0)
        img_model = load_image_model(IMAGE_MODEL_PATH)
        prediction = img_model.predict(image_array)
        sentiment_map = {
            'happy': 0,
            'sad': 1,
            'fear': 2,
            'surprise': 3,
            'neutral': 4,
            'angry': 5,
            'disgust': 6,
        }
        sentiment_idx = np.argmax(prediction, axis=1)[0]
        sentiment = sentiment_map[sentiment_idx]
    else:
        # process as a string
        glove_vectorizer = GloveVectorizer(PICKLED_GLOVE_DIR)
        text_vector = glove_vectorizer.vectorize_text(input_data)
        # Add batch dimension
        text_vector = np.expand_dims(text_vector, axis=0)  
        model = load_text_model(TEXT_MODEL_PATH)
        prediction = model.predict(text_vector)
        
        # Assuming the text model outputs binary data (1 class)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment

if __name__ == '__main__':
    # Use a pretrained model to predict input sentiment
    input_data = input("Please enter the path to an image file or text: ")
    sentiment = predict_sentiment(input_data)
    print(f"Predicted Sentiment: {sentiment}")
        