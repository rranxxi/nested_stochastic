import numpy as np
import pandas as pd

class fundMapper:
    def __init__(self, data_path):
        self.df = self.readFundMapping(data_path)

    def readFundMapping(self, data_path):
        return pd.read_excel(data_path, sheet_name = 'Fund Mapping')
        
    def getMappingWeightsForFund(self, origina_fund_name):
        """Map an original fund name to the mapped fund or index and will return
        the weights of the mapped fund
        Args:
            origina_fund_name (str): Original fund name from the invoice files
        """
        return self.df['origina_fund_name'].to_numpy()
    
    def getMappingMatrix(self):
        return self.df.iloc[0:10, 1:11].to_numpy()
    
    def getMappedIndexNames(self):
        self.df.iloc[0:10, 0].to_numpy()
    

        