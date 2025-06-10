import pandas as pd

class DataCollector:
    def __init__(self, data_source):
        """
        Initialize the DataCollector object.
        
        Parameters
        ----------
        data_source : str
            The source of the data. This could be a file path, a database
            connection string, or a web API.
        """
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        """
        Load data from the specified data source into a pandas DataFrame.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.data = pd.read_csv(self.data_source)

    def get_structured_data(self):
        return self.data
