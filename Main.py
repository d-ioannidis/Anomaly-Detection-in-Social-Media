import sys
sys.path.insert(0, 'Modules')
from Data_Collector import DataCollector
from Data_Preprocessor import DataPreprocessor
from Anomaly_Detection import AnomalyDetector

def main():
    # 1. Initialize and load data
    data_collector = DataCollector("DisasterTweets.csv")
    data_collector.load_data()

    # 2. Preprocess data
    data_preprocessor = DataPreprocessor(data_collector)
    data_preprocessor.preprocess_data()

    # 3. Inspect the first 5 records of the dataframe
    print(data_preprocessor.data.head())

    # Display the column names of the dataframe
    #print(data_preprocessor.data.columns)
    #print(data_preprocessor.data['Disaster'])

    # 4. Calculate the document-term matrix
    dtm = data_preprocessor.calculate_document_term_matrix()

    # 5. Calculate the term-frequency-inverse document-frequency matrix
    tfidf = data_preprocessor.calculate_term_frequency_inverse_document_frequency_matrix()

    # 6. Generate BERT embeddings
    bert_embeds = data_preprocessor.bert_tokenize()
    
    # 7. Use anomaly detection
    anomaly_detector = AnomalyDetector()
    anomaly_results = {
        'kmeans_tfidf': anomaly_detector.kmeans_anomaly_detection(tfidf),
        'kmeans_dtm': anomaly_detector.kmeans_anomaly_detection(dtm),
        'kmeans_bert': anomaly_detector.kmeans_anomaly_detection(bert_embeds),
        'dbscan_tfidf': anomaly_detector.dbscan_anomaly_detection(tfidf),
        'dbscan_dtm': anomaly_detector.dbscan_anomaly_detection(dtm),
        'dbscan_bert': anomaly_detector.dbscan_anomaly_detection(bert_embeds),
        'hybrid_dtsvmnb_tfidf': anomaly_detector.hybrid_dtsvmnb_anomaly_detection(tfidf, data_preprocessor.data['Disaster']),
        'hybrid_dtsvmnb_dtm': anomaly_detector.hybrid_dtsvmnb_anomaly_detection(dtm, data_preprocessor.data['Disaster']),
        #'hybrid_dtsvmnb_bert': anomaly_detector.hybrid_dtsvmnb_anomaly_detection(bert_embeds, data_preprocessor.data['Disaster']),
        'autoencoder_tfidf': anomaly_detector.autoencoder_anomaly_detection(tfidf),
        'autoencoder_dtm': anomaly_detector.autoencoder_anomaly_detection(dtm),
        'autoencoder_bert': anomaly_detector.autoencoder_anomaly_detection(bert_embeds)
    }

    # for name, result in anomaly_results.items():
    #     print(f"Anomaly results for {name}:")
    #     print(result)
    
if __name__ == "__main__":
    main()