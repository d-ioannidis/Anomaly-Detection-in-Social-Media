# Fact_Checker.py
import lmstudio as lms
import logging
import re
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class FactChecker:
    def __init__(self):
        """
        Initializes the FactChecker object.

        This method initializes the FactChecker object by fetching the reasoning model and
        initializing an empty cache to store the results of previous queries.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logging.info("Initializing FactChecker...")
        self.model = lms.llm("google/gemma-3-1b")
        self.cache = {}
        logging.info("Model initialized.")

    def get_evidence_from_web_search(self, query):
        """
        Performs a web search for the given query and returns the evidence.

        This method takes a query string, performs a web search using the Bing API, and
        returns the combined text of the search results as a single string.

        Parameters
        ----------
        query : str
            The query string to search for.

        Returns
        -------
        str
            The combined text of the search results as a single string.
        """

        logging.debug(f"Performing web search for query: {query}")

        url = f"https://www.bing.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = [result.text for result in soup.find_all('li', class_='b_algo')]

        return ' '.join(results)

    def verify_fact(self, claim):
        """
        Verifies the given claim by using a language model to evaluate evidence gathered from a web search.

        This method takes a claim string, performs a web search for evidence related to the claim, and
        uses a language model to evaluate the evidence and return a verdict on the claim's accuracy.

        The method caches the results of previous queries to avoid redundant computations.

        Parameters
        ----------
        claim : str
            The claim string to verify.

        Returns
        -------
        str
            The verdict on the claim's accuracy, one of True, False, Partially True, Partially False, Unclear.
        """
        if (claim,) in self.cache:
            return self.cache[(claim,)]
            
        message = {
            "role": "user",
            "content": f"Claim: {claim}\nEvidence: {self.get_evidence_from_web_search(claim)}\nIs this claim supported?"
        }

        result = self.model.respond({"messages": [message]})
        text = result.content if hasattr(result, 'content') else str(result)
        self.cache[(claim,)] = text

        return text
    
    def flag_with_llm(self, model, prediction):
        """
        Classify the factual accuracy of a given prediction using a language model.

        Parameters
        ----------
        model : Language Model
            The language model to use for classification.
        prediction : str
            The prediction to classify.

        Returns
        -------
        str
            The classification label, one of True, False, Partially True, Partially False, Unclear.
        """
        snippet = prediction
        
        prompt = f"""The following text is the result of a fact-checking model evaluating a claim.  
    Please classify the factual accuracy as one of the following exactly:  
    True, False, Partially True, Partially False, Unclear.

    Excerpt:
    "{snippet}"

    Return only one label."""
        
        response = model.respond({"messages": [{"role": "user", "content": prompt}]})
        return response.content.strip()
    
    def fact_check_tweets(self, tweets):
        """
        Fact-checks the given tweets by performing a web search, retrieving evidence, and
        querying the reasoning model to determine if the evidence supports the claim.

        Parameters
        ----------
        tweets : pd.DataFrame
            DataFrame with a column named 'Original Tweets' containing the tweets to fact-check.

        Returns
        -------
        pd.DataFrame
            The input DataFrame with a new column 'Fact_Check_Prediction' added, containing the
            prediction of the reasoning model, either 'supported', 'not supported', or 'inconclusive'.
        """
        logging.info("Starting fact-checking of tweets…")
        tweets = tweets.copy()
        batch_size = 50

        # 1) Define output path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))      # Modules folder
        out_path = os.path.join(base_dir, '..', 'fact_check_results.csv')
        out_path = os.path.normpath(out_path)

        # 2) Create file with header if it doesn't exist
        if not os.path.isfile(out_path):
            logging.info(f"fact_check_results.csv not found at {out_path}, creating it with header.")
            pd.DataFrame(
                columns=['Original Tweets', 'Fact_Check_Prediction']
            ).to_csv(out_path, index=False)
        else:
            logging.info(f"Using existing fact_check_results.csv at {out_path}")

        # 3) Load just the 'Original Tweets' column to know what's done
        done = pd.read_csv(out_path, usecols=['Original Tweets'])['Original Tweets'].tolist()

        # 4) Process in batches & append
        for start in range(0, len(tweets), batch_size):
            batch = tweets.iloc[start : start + batch_size].copy()
            batch = batch[~batch['Original Tweets'].isin(done)]
            if batch.empty:
                logging.info(f"Batch {start//batch_size+1}: nothing new to process, skipping.")
                continue

            claims = batch['Original Tweets'].tolist()

            with ThreadPoolExecutor(max_workers=32) as executor:
                predictions = list(executor.map(self.verify_fact, claims))
                flags       = list(executor.map(lambda p: self.flag_with_llm(self.model, p), predictions))

            batch['Fact_Check_Prediction'] = flags

            # Append only the two needed columns
            batch[['Original Tweets','Fact_Check_Prediction']].to_csv(
                out_path, mode='a', header=False, index=False
            )
            logging.info(f"Appended {len(batch)} rows to fact_check_results.csv")

            # Update done list so we don’t re-process in this run
            done.extend(batch['Original Tweets'].tolist())

        logging.info("Fact-checking completed.")

        return tweets