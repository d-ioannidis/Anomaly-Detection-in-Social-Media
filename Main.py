import sys
sys.path.insert(0, 'Modules')
from Data_Collector import DataCollector
from Data_Preprocessor import DataPreprocessor

def main():
    # 1. Initialize and load data
    data_collector = DataCollector("DisasterTweets.csv")
    data_collector.load_data()

    # 2. Preprocess data
    data_preprocessor = DataPreprocessor(data_collector)
    data_preprocessor.preprocess_data()

    # 3. Inspect the first 5 records of the dataframe
    print(data_preprocessor.data.head())

    # 4. Calculate the document-term matrix
    dtm = data_preprocessor.calculate_document_term_matrix()

    # 5. Calculate the term-frequency-inverse document-frequency matrix
    tfidf = data_preprocessor.calculate_term_frequency_inverse_document_frequency_matrix()
    
    # 6. Print the first 5 rows of the document-term matrix
    print(dtm[:5])

    # 7. Print the first 5 rows of the term-frequency-inverse document-frequency matrix
    print(tfidf[:5])
    
if __name__ == "__main__":
    main()