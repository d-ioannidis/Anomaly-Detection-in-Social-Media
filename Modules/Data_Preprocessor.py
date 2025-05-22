from Data_Collector import DataCollector
import spacy
from nltk.stem import *
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics import silhouette_score

class DataPreprocessor:
    def __init__(self, data_collector: DataCollector):
        """
        Initialize the DataPreprocessor object.

        Parameters
        ----------
        data_collector : DataCollector
            An instance of DataCollector used to fetch and structure the data.

        Attributes
        ----------
        data_collector : DataCollector
            Stores the passed DataCollector instance.
        data : DataFrame
            Stores the structured data fetched from the DataCollector.
        """

        self.data_collector = data_collector
        self.data = data_collector.get_structured_data()

    def convert_to_numeric(self, x):
        """
        Converts a string representation of a number with 'K' or 'M' suffix into a numeric value.
        
        Parameters:
        x (str): The string representation of the number to be converted.
        
        Returns:
        float: The numeric value of the input string.
        """
        if 'K' in x:
            return float(x.replace('K', '')) * 1000
        elif 'M' in x:
            return float(x.replace('M', '')) * 1000000
        else:
            return float(x)
    
    def preprocess_data(self):
        """
        Preprocesses the data by converting the Likes column to numeric, converting specified columns to lowercase, 
        removing special characters and hashtags, performing tokenization and lemmatization, removing stop words, 
        and using a stemmer to reduce words to their root form.

        Returns
        -------
        DataPreprocessor
            The preprocessed data object.
        """
        def lower_if_string(x):
            """
            Converts a string or a list of strings to lowercase.

            Parameters
            ----------
            x : str or list
                The string or list of strings to be converted to lowercase.

            Returns
            -------
            str or list
                The converted string or list of strings.
            """
            if isinstance(x, str):
                return x.lower()
            elif isinstance(x, list):
                return [lower_if_string(i) for i in x]
            else:
                return x

        # Convert Likes column to numeric
        self.data['Likes'] = self.data['Likes'].apply(self.convert_to_numeric)
        
        # Convert specified columns to lowercase
        columns_to_lowercase = ['Name', 'UserName', 'Tweets', 'Tags', 'Tweet Link', 'Disaster']
        self.data[columns_to_lowercase] = self.data[columns_to_lowercase].map(lower_if_string)

        # Keep a copy of Tweets column
        self.data['Original Tweets'] = self.data['Tweets']

        # Strip URL and HTML tags, and remove special characters and hashtags from categorical columns
        columns_to_clean = ['Name', 'UserName', 'Tweets', 'Tags', 'Disaster']
        self.data[columns_to_clean] = self.data[columns_to_clean].apply(
            lambda x: x.str.replace(r'http\S+', '', regex=True)
            .str.replace(r'<.*?>', '', regex=True)
            .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        )

        # Perform tokenization using spaCy
        nlp = spacy.load('en_core_web_sm')
        self.data['Tokens'] = self.data['Tweets'].apply(lambda x: [token.text for token in nlp(x)])

        # Lemmatize tokens
        self.data['Tokens'] = self.data['Tokens'].apply(lambda x: [lower_if_string(token.lemma_) for token in nlp(' '.join(x))])

        # Remove stop words
        stop_words = nlp.Defaults.stop_words
        self.data['Tokens'] = self.data['Tokens'].apply(lambda x: [token for token in x if token not in stop_words])

        # Use stemmer to reduce words to their root form
        stemmer = PorterStemmer()
        self.data['Tokens'] = self.data['Tokens'].apply(lambda x: [stemmer.stem(token) for token in x])

        return self
    
    def one_hot_encode(self):
        """
        Performs one-hot encoding on the 'Disaster' column and adds it to the original data.

        The one-hot encoded columns have the prefix 'Disaster' and are concatenated to the original data on the 1st axis.

        Returns
        -------
        DataPreprocessor
            The preprocessed data object.
        """
        self.data['Original_Disaster'] = self.data['Disaster']

        disaster_one_hot = pd.get_dummies(self.data['Disaster'], prefix='Disaster')

        self.data = pd.concat([self.data, disaster_one_hot], axis=1)

        return self
    
    def bert_tokenize(self):
        """
        Tokenizes and generates BERT embeddings for the 'Tweets' column.

        This method uses the BERT tokenizer and model to encode tweets into token embeddings.
        It extracts the [CLS] token representation as the embedding for each tweet and stores
        the result in the 'Embeddings' column of the data.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array where each row corresponds to the BERT [CLS] embedding of a tweet.
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()  # Set to eval mode

        # Generate BERT token embeddings
        embeddings = []
        for text in self.data['Tweets']:
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            with torch.no_grad():  # Disable gradient tracking
                outputs = model(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            # Get [CLS] token representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)

        self.data['Embeddings'] = embeddings

        return np.vstack(embeddings) # Return embeddings

    def calculate_document_term_matrix(self):
        """
        Calculates the document-term matrix using the preprocessed data.
        
        Returns:
        DocumentTermMatrix: The document-term matrix.
        """
        
        self.data['Token_String'] = self.data['Tokens'].apply(lambda toks: " ".join(toks))

        vectorizer = CountVectorizer()
        dtm = vectorizer.fit_transform(self.data['Token_String'])

        return dtm
    
    def calculate_term_frequency_inverse_document_frequency_matrix(self):
        """
        Calculates the term-frequency-inverse document-frequency matrix using the preprocessed data.
        
        Returns:
        TermFrequencyInverseDocumentFrequencyMatrix: The term-frequency-inverse document-frequency matrix.
        """
        
        self.data['Token_String'] = self.data['Tokens'].apply(lambda toks: " ".join(toks))

        vectorizer = CountVectorizer()
        tfidf = vectorizer.fit_transform(self.data['Token_String'])

        return tfidf