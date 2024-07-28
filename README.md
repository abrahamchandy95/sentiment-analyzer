# Sentiment Analyzer

This repository contains the code for a sentiment prediction project that utilizes both text and image data to predict the sentiment. 
The project is structured to handle separate training for text and image models, leveraging natural language processing and image vectorization techniques.

## Setup

All datasets and pretrained models are excluded from this repository to manage the repository size and comply with data sharing policies. 
However, you can download and initialize these datasets using scripts included in the repository.

### Downloading Necessary Data

Navigate to the `src` directory of the cloned project, which contains a subdirectory named `scripts`. 
Run the following scripts in order to download the necessary datasets and tools:

1. **GloVe Dataset**:
   ```bash
   python scripts/download_tools/glove.py
   This script downloads the GloVe dataset, used for text vectorization.

2. **NLTK Tools**:
   ```bash
   python scripts/download_tools/nltk_tools.py
To download necessary NLTK tools for natural language processing, used for tokenization, etc.

3. **Face Images**:
   ```bash
   python scripts/download_training_data/initialize_face_images.py
Downloads images for training and testing the model on sentiment prediction related to facial expressions.

4. **Twitter Dataset**:
   ```bash
   python scripts/download_training_data/initialize_twitter_datasets.py
Downloads Twitter data to train the model on textual data for sentiment prediction.

## Training the Models

The training scripts are located within the `src/training` directory. 
Execute each script from the project's root directory to ensure proper file access. 
Each script is dedicated to a specific part of the training process:

1. **Image Data Training**:
   Navigate to the root of the project and run:
   ```bash
   python src/training/training_images.py
2. **Text Data Training**:
   Navigate to the root of the project and run:
   ```bash
   python src/training/training_text.py

## Running the Main Application

After training the models, execute the main application to perform sentiment prediction. 
From the root directory of the project, run:

```bash
python src/main.py

### Application Usage

Upon execution, the application prompts the user to input either a text string or an image file path:

- **Text Input**: Type a sentence or paragraph, and the application will analyze and predict the sentiment of the text.
- **Image Input**: Provide the path to an image file, and the application will evaluate the sentiment expressed in the image.

The application utilizes the trained models to perform predictions and outputs the sentiment prediction directly to the console.

## Project Structure

Ensure that all scripts are executed from the root directory of the cloned project.
This practice maintains proper access to all necessary resources.
The training process automatically creates a `model` directory at the project's root, where all trained models are stored.

## Contributing

We welcome contributions to this project. To contribute:

- Fork the repository.
- Create a new branch for your feature (`git checkout -b feature/YourFeatureName`).
- Commit your changes (`git commit -am 'Add a new feature'`).
- Push to the branch (`git push origin feature/YourFeatureName`).
- Open a pull request.

## License

This project is licensed under the MIT License. For more details, refer to the [LICENSE.md](LICENSE) file in this repository.

