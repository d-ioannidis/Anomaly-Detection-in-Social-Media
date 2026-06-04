from Data_Collector import DataCollector
from NLP_Engine import NLPEngine
import spacy
from nltk.stem import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

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
        self.nlp_engine = NLPEngine()

    def convert_to_numeric(self, x):
        """
        Converts a string representation of a number with 'K' or 'M' suffix
        into a numeric value. Also safely handles already-numeric inputs.
        """
        if pd.isna(x):
            return 0.0

        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)

        x = str(x).strip()

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

        self.data['Sentiment_Scores'] = self.data['Original Tweets'].apply(
            lambda x: self.nlp_engine.get_sentiment_scores(x)
        )

        self.data['Emotion_Labels'] = self.data['Original Tweets'].apply(
            lambda x: self.nlp_engine.get_emotion_labels(x) 
        )

        self.data['Sentiment_Label'] = self.data['Sentiment_Scores'].apply(
            lambda x: x[0]['label'] if x else None
        )

        self.data['Sentiment_Score'] = self.data['Sentiment_Scores'].apply(
            lambda x: x[0]['score'] if x else None
        )

        # Remove exact duplicate tweets after normalization
        self.data = self.data.drop_duplicates(subset=['Tweets']).reset_index(drop=True)

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

        # Move model to device if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

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

            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

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
        Calculates the TF-IDF matrix using the preprocessed token strings.
        Adapts min_df/max_df for small live batches.
        """
        self.data['Token_String'] = self.data['Tokens'].apply(lambda toks: " ".join(toks))

        n_docs = len(self.data)

        # Safer settings for small API batches
        if n_docs < 5:
            min_df = 1
            max_df = 1.0
        else:
            min_df = 2
            max_df = 0.95

        vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(self.data['Token_String'])

        tfidf_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_matrix,
            index=self.data.index,
            columns=vectorizer.get_feature_names_out()
        )

        return tfidf_df
    
    def print_results(self):
        """
        Prints a comprehensive summary of the preprocessing results, including key metrics
        and dimensions of the transformed data. This output is designed to be used directly
        in the 'Results' section of the thesis.
        """
        # 1. Basic Data Shape
        original_shape = self.data_collector.data.shape
        processed_shape = self.data.shape
        print(f"1. DATA VOLUME:")
        print(f"   - Original dataset dimensions: {original_shape[0]} rows, {original_shape[1]} columns")
        print(f"   - Processed dataset dimensions: {processed_shape[0]} rows, {processed_shape[1]} columns (after adding new features)\n")

        # 2. Text Cleaning Efficacy (Example: Calculate avg. tweet length before and after)
        # Assuming 'Tweets' is the cleaned text and 'Original Tweets' is the raw text
        avg_chars_before = self.data['Original Tweets'].str.len().mean()
        avg_chars_after = self.data['Tweets'].str.len().mean()
        chars_removed_pct = ((avg_chars_before - avg_chars_after) / avg_chars_before) * 100

        avg_tokens = self.data['Tokens'].apply(len).mean()

        print(f"2. TEXT CLEANING EFFICACY:")
        print(f"   - Avg. characters per tweet (raw): {avg_chars_before:.2f}")
        print(f"   - Avg. characters per tweet (cleaned): {avg_chars_after:.2f}")
        print(f"   - Estimated noise removed: {chars_removed_pct:.2f}% of characters")
        print(f"   - Avg. tokens per tweet (after stopword removal): {avg_tokens:.2f}\n")

        # 3. Feature Extraction Outputs
        # Calculate DTM/TF-IDF dimensions
        dtm = self.calculate_document_term_matrix()
        tfidf_df = self.calculate_term_frequency_inverse_document_frequency_matrix()
        vocab_size = tfidf_df.shape[1] # Number of columns in TF-IDF matrix is the vocabulary size

        # Get BERT Embeddings shape
        bert_embeddings = self.bert_tokenize()
        
        print(f"3. FEATURE EXTRACTION OUTPUTS:")
        print(f"   - Document-Term Matrix (DTM) dimensions: {dtm.shape}")
        print(f"   - TF-IDF Matrix dimensions: {tfidf_df.shape}")
        print(f"   - Vocabulary size (n_features_tfidf): {vocab_size}")
        print(f"   - BERT Embeddings matrix dimensions: {bert_embeddings.shape}\n")

        # 4. Sentiment/Emotion Distribution (Top-level counts)
        print(f"4. SENTIMENT/EMOTION DISTRIBUTION (Preview):")
        print(f"   - Sentiment Label counts:")
        print(self.data['Sentiment_Label'].value_counts().to_string())
        print(f"\n   - Emotion Label counts:")
        print(self.data['Emotion_Labels'].value_counts().to_string())