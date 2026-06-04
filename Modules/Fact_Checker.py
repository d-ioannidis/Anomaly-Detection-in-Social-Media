# Fact_Checker.py
import lmstudio as lms
import logging
import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class FactChecker:
    def __init__(self, output_dir=None, model_name="google/gemma-3-1b"):
        """
        Initializes the FactChecker object.

        Parameters
        ----------
        output_dir : str or None
            Optional directory where fact_check_results.csv should be stored.
        model_name : str
            LM Studio model identifier.
        """
        logging.info("Initializing FactChecker...")
        self.model = lms.llm(model_name)
        self.cache = {}
        self.output_dir = output_dir
        self.model_name = model_name
        logging.info(f"Model initialized: {model_name}")

    def _get_fact_check_output_path(self):
        """
        Resolve the output path for fact_check_results.csv.
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "fact_check_results.csv")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(base_dir, '..', 'fact_check_results.csv'))

    def clean_claim(self, claim: str, max_len: int = 500) -> str:
        """
        Clean and shorten noisy social-media text before search/LLM use.
        """
        claim = str(claim)
        claim = re.sub(r"http\S+|www\.\S+", " ", claim)
        claim = re.sub(r"\s+", " ", claim).strip()
        return claim[:max_len]

    def get_evidence_from_web_search(self, query: str, max_results: int = 5, max_chars_per_result: int = 400) -> str:
        """
        Performs a web search for the given query and returns a concise evidence string.
        Uses requests params= for safe query encoding.
        """
        try:
            query = self.clean_claim(query)
            logging.debug(f"Performing web search for query: {query}")

            url = "https://www.bing.com/search"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(
                url,
                params={"q": query},
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            blocks = soup.find_all('li', class_='b_algo')[:max_results]

            snippets = []
            for block in blocks:
                text = " ".join(block.stripped_strings)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    snippets.append(text[:max_chars_per_result])

            evidence = "\n\n".join(snippets)
            return evidence if evidence else "No clear external evidence found."

        except Exception as e:
            logging.exception(f"Web search failed for query: {query} | error: {e}")
            return "No clear external evidence found."

    def normalize_label(self, text: str) -> str:
        """
        Normalize model output to one of the allowed labels.
        """
        t = str(text).strip()

        # Use first non-empty line only
        first_line = next((line.strip() for line in t.splitlines() if line.strip()), "")
        low = first_line.lower()

        if low.startswith("partially false"):
            return "Partially False"
        if low.startswith("partially true"):
            return "Partially True"
        if low.startswith("false"):
            return "False"
        if low.startswith("true"):
            return "True"
        if low.startswith("unclear"):
            return "Unclear"

        # fallback contains-based normalization
        if "partially false" in low:
            return "Partially False"
        if "partially true" in low:
            return "Partially True"
        if "false" in low:
            return "False"
        if "true" in low:
            return "True"
        if "unclear" in low:
            return "Unclear"

        return "Unclear"

    def verify_fact_label(self, claim: str) -> str:
        """
        Verify a claim and return a single final label directly:
        True, False, Partially True, Partially False, Unclear
        """
        clean_claim = self.clean_claim(claim)

        if clean_claim in self.cache:
            return self.cache[clean_claim]

        try:
            evidence = self.get_evidence_from_web_search(clean_claim)

            prompt = f"""
You are verifying a crisis-related social media claim.

Claim:
{clean_claim}

Evidence:
{evidence}

Classify the claim as exactly one of the following labels:
True
False
Partially True
Partially False
Unclear

Rules:
- Return only one label.
- Use Unclear if the evidence is insufficient, noisy, contradictory, or unrelated.
- Do not explain your reasoning.
""".strip()

            # LM Studio docs support a single-string .respond(...) call for one-message prompts
            result = self.model.respond(prompt)
            raw_text = result.content if hasattr(result, 'content') else str(result)

            label = self.normalize_label(raw_text)
            self.cache[clean_claim] = label
            return label

        except Exception as e:
            logging.exception(f"verify_fact_label failed for claim: {clean_claim[:120]} | error: {e}")
            self.cache[clean_claim] = "Unclear"
            return "Unclear"

    def load_fact_check_results(self):
        """
        Loads the fact-check results from CSV.
        """
        out_path = self._get_fact_check_output_path()

        if not os.path.exists(out_path):
            return pd.DataFrame(columns=['Tweet ID', 'Original Tweets', 'Fact_Check_Prediction'])

        return pd.read_csv(out_path)

    def fact_check_tweets(self, tweets, reset_output=False):
        """
        Fact-checks the given tweets and writes labels to fact_check_results.csv.

        Parameters
        ----------
        tweets : pd.DataFrame
            DataFrame containing at least 'Tweet ID' and 'Original Tweets'
        reset_output : bool
            If True, deletes any existing fact_check_results.csv before running.
        """
        logging.info("Starting fact-checking of tweets…")
        tweets = tweets.copy()
        batch_size = 50

        out_path = self._get_fact_check_output_path()

        if reset_output and os.path.isfile(out_path):
            logging.info(f"Reset requested. Removing existing fact_check_results.csv at {out_path}")
            os.remove(out_path)

        if not os.path.isfile(out_path):
            logging.info(f"fact_check_results.csv not found at {out_path}, creating it with header.")
            pd.DataFrame(
                columns=['Tweet ID', 'Original Tweets', 'Fact_Check_Prediction']
            ).to_csv(out_path, index=False)
        else:
            logging.info(f"Using existing fact_check_results.csv at {out_path}")

        if tweets.empty:
            logging.warning("No tweets provided to fact_check_tweets; returning empty result file.")
            return pd.read_csv(out_path)

        done = pd.read_csv(
            out_path,
            usecols=['Tweet ID', 'Original Tweets']
        )[['Tweet ID', 'Original Tweets']].to_dict('records')

        print("fact_check_tweets received rows:", len(tweets))
        print("fact_check_tweets columns:", list(tweets.columns))
        print(tweets[['Tweet ID', 'Original Tweets']].head())
        print("Existing done rows:", len(done))

        for start in range(0, len(tweets), batch_size):
            batch = tweets.iloc[start:start + batch_size].copy()
            print("Batch size before filtering:", len(batch))

            batch = batch[~batch['Tweet ID'].isin([row['Tweet ID'] for row in done])]
            print("Batch size after filtering:", len(batch))

            if batch.empty:
                logging.info(f"Batch {start // batch_size + 1}: nothing new to process, skipping.")
                continue

            claims = batch['Original Tweets'].tolist()
            print("Claims to verify:", len(claims))

            try:
                # Keep low while stabilizing LM Studio behavior
                with ThreadPoolExecutor(max_workers=1) as executor:
                    labels = list(executor.map(self.verify_fact_label, claims))
                    print("Labels returned:", len(labels))

                batch['Fact_Check_Prediction'] = labels

                batch[['Tweet ID', 'Original Tweets', 'Fact_Check_Prediction']].to_csv(
                    out_path, mode='a', header=False, index=False
                )
                logging.info(f"Appended {len(batch)} rows to fact_check_results.csv")

                done.extend(batch[['Tweet ID', 'Original Tweets']].to_dict('records'))

            except Exception as e:
                logging.exception(f"Batch {start // batch_size + 1} failed: {e}")

                batch['Fact_Check_Prediction'] = "Unclear"
                batch[['Tweet ID', 'Original Tweets', 'Fact_Check_Prediction']].to_csv(
                    out_path, mode='a', header=False, index=False
                )
                logging.warning(
                    f"Batch {start // batch_size + 1} wrote fallback 'Unclear' labels for {len(batch)} rows."
                )

                done.extend(batch[['Tweet ID', 'Original Tweets']].to_dict('records'))

        logging.info("Fact-checking completed.")
        return pd.read_csv(out_path)