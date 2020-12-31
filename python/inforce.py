import numpy as np
import pandas as pd
from  fundmapper import  fundMapper

class inforce:
    def __init__(self, inforce_path, data_path):
        self.df = pd.read_csv(inforce_path)
        self.fund_mapper = fundMapper(data_path)
        # Mapped index df     
        self.df_index = self.calculateMappedIndex()   
        self.num_of_funds = 10 

    def numOfFunds(self):
        return self.num_of_funds
    
    def numOfInforces(self):
        return self.df.shape[0]
    
    def inforcesGBAmount(self):
        return self.df['gbAmt'].to_numpy()
    
    def fundValuesNumpy(self):
        fund1_index = list(self.df.columns).index('FundValue1')
        return self.df.iloc[:,fund1_index : fund1_index + 10].to_numpy()

    def fundFeeNumpy(self):
        fund1_index = list(self.df.columns).index('FundFee1')
        return self.df.iloc[:,fund1_index : fund1_index + 10].to_numpy()
    
    def calculateMappedIndex(self):
        index_names = self.fund_mapper.getMappedIndexNames()
        mapping_mat = self.fund_mapper.getMappingMatrix().T  
        fund1_index = list(self.df.columns).index('FundValue1')
        fund_mat = self.df.iloc[:,fund1_index : fund1_index + 10].to_numpy()
        # Map the fund into the index 
        index_mat = fund_mat.dot(mapping_mat)
        # Create DF for the mapped index
        return pd.DataFrame(columns = index_names, data = index_mat)
    
    def randomSampleInforce(self, nsample):
        """Randomly sample the inforce

        Args:
            nsample (int): number of target inforces
        """
        return self.df.sample(n = nsample, random_state=1)


        