import os
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re

class GloveVectorizer:
    """
    class to vectorize any text using the GloVe dataset
    
    Attributes:
        pickled_glove_dir (str): Directory where the pickled GloVe vectors are stored.
        glove_vectors (Dict[str, np.ndarray]): Dictionary containing word vectors.
        vector_dims (int): Number of dimensions per GloVe vector. 
    """
    def __init__(self, pickled_glove_dir: str):
        self.pickled_glove_dir = pickled_glove_dir
        self.glove_vectors, self.vector_dims = self.load_glove_vectors()
        self.tokenizer = RegexpTokenizer(pattern=r'\w+')
        self.lemmatizer = WordNetLemmatizer()
        
    def load_glove_vectors(
        self
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Loads the GloVe vectors the pickled file.

        Returns:
            Tuple[Dict[str, np.ndarray], int]: A tuple containing the 
            GloVe vectors dictionary and the number of dimensions.
        """
        pickled_files = [
            f for f in os.listdir(self.pickled_glove_dir) 
            if f.endswith('.pkl')
        ]
        # number of dimensions per word
        n_dims = None
        chosen_file = None
        for file in pickled_files:
            matched_file = re.search(r'glove_vectors_(\d+)d.pkl', file)
            vector_dims = int(matched_file.group(1))
            if n_dims is None or vector_dims < n_dims:
                n_dims = vector_dims # Take the minimum number of dims
                chosen_file = file
        with open(os.path.join(self.pickled_glove_dir, chosen_file), 'rb') as f:
            glove_vectors = pickle.load(f)
        return glove_vectors, n_dims
    
    def convert_text_to_tokens(
        self, text:str
    )-> List[str]:
        """
        Converts text to a list of useful tokens.

        Parameters:
            text (str): The input text to be tokenized.

        Returns:
            List[str]: A list of useful tokens can be vectorized.
        """
        tokens = self.tokenizer.tokenize(text=text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        lowercase_tokens = [t.lower() for t in lemmatized_tokens]
        useful_tokens = [
            t for t in lowercase_tokens if t in self.glove_vectors
        ]
        return useful_tokens

    def vectorize_text(
        self, text: str, max_length: int=120
    )-> Optional[np.ndarray]:
        """
        Vectorizes the input text.

        Parameters:
            text (str): The input text to be vectorized.
            max_length (int): The maximum length of the vectorized text. Defaults to 120.

        Returns:
            Optional[np.ndarray]: A numpy array of shape (max_length, vector_dims) containing
            the vectorized and padded text.
        """
        tokenized_message = self.convert_text_to_tokens(text)
        vectors = []
        for token in tokenized_message:
            if token not in self.glove_vectors:
                continue
            token_vector = self.glove_vectors[token]
            vectors.append(token_vector)
        
        if not vectors:
            vectors = np.zeros((max_length, self.vector_dims))
        else:
            vectors = np.array(vectors, dtype=float)

        # padding
        padded_vectors = np.zeros((max_length, self.vector_dims))
        actual_length = min(len(vectors), max_length)
        for i in range(actual_length):
            padded_vectors[i] = vectors[i]
        return padded_vectors