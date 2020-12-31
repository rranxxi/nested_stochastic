import numpy as np
import pandas as pd

class RealWorldScenario:
    def __init__(self, data_path, num_years = 30):
        self.df = pd.read_excel(data_path, sheet_name = 'RW Scenarios')
        # import pdb; pdb.set_trace()
        self.num_datapoints = self.df[self.df['// Scenario'] == 1].shape[0]
        self.num_scenarios = self.df['// Scenario'].to_numpy()[-1]
        self.num_years = num_years
    
    def numberOfScenarios(self):
        return self.num_scenarios
    
           
    def getScenario(self, scenarios_id):
        if scenarios_id >= 1 and scenarios_id <= self.num_scenarios:
            df_tmp = self.df[self.df['// Scenario'] == scenarios_id]
            return df_tmp.iloc[:,2:].to_numpy()
        return None
    

class RiskNerualScenario:
    def __init__(self, data_path, num_years = 30):
        self.df = pd.read_excel(data_path, sheet_name = 'RN Scenarios')
        self.num_datapoints = self.df[self.df['// Scenario'] == 1].shape[0]
        self.num_scenarios = self.df['// Scenario'].to_numpy()[-1]
        self.num_years = num_years
        
    def numberOfScenarios(self):
        return self.num_scenarios
    
    def getScenario(self, scenarios_id):
        if scenarios_id >= 1 and scenarios_id <= self.num_scenarios:
            df_tmp = self.df[scenarios_id]
            return df_tmp.to_numpy()
        return None
    
    