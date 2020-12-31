import numpy as np
import  math
import pdb
from inforce import  inforce
from scenarios import RealWorldScenario
from scenarios import RiskNerualScenario
from datetime import datetime
from copy import deepcopy


import tqdm

from params import *

class VACalculator:
    def __init__(self, inforce_path = None, excel_path = None, 
            fixed_mortality_rate = 0.02, fixed_laps_rate = 0.015, 
            risk_free_interest_rate = 0.03, valuation_frequency = 1.0 / 12,
            projection_term_years = 30):
        self.fixed_mortality_rate = fixed_mortality_rate
        self.fixed_laps_rate = fixed_laps_rate
        self.risk_free_interest_rate = risk_free_interest_rate
        self.valuation_frequency = valuation_frequency
        self.projection_term_years = projection_term_years
        
        self.inforce = inforce(inforce_path, excel_path)
        self.rw_scenarios = RealWorldScenario(excel_path)
        self.rn_scenarios = RiskNerualScenario(excel_path)
        
        self.projection_period = np.linspace(0, 12 * self.projection_term_years, 12 * self.projection_term_years, dtype=np.int)
        self.survivorship = self.computeSurvivorship()
        self.accum_disfactor = self.computeAccumDisFactor()
        self.disc_factor = self.computeDiscFactor()
        
    
    def projectionPeriod(self):
        return self.projection_period
    
    def fixedMortalityRate(self):
        return self.fixed_mortality_rate
    
    def valuationFrequency(self):
        """The evaluation steps for each year

        Returns:
            float: 1.0 / num_of_evaluation_data_points
        """
        return self.valuation_frequency

    def riskFreeInterestRate(self):
        """Risk-free interest rate

        Returns:
            float: A constane value of risk free interest rate. Loaded from assumptions
        """
        return self.risk_free_interest_rate

    def projectionTermYears(self):
        """Number of projection years
        Returns:
            int: a constant value of projection years: 30 
        """
        return self.projection_term_years
    
    def computeSurvivorship(self, init_prob = 1.0):
        """Compute Projected Survivor Rate from the provided moraity rate

        Args:
            init_prob (float, optional): Initial probability of survivorship. Defaults to 1.0.

        Returns:
            numpy array: Projected survivorship
        """
        survivorship = np.empty(self.projection_term_years * 12 + 1)
        prev = init_prob
        survivorship[0] = prev
        for i in range(1, survivorship.size):
            survivorship[i] = prev * math.exp( 0 - self.fixedMortalityRate() * self.valuationFrequency())
            prev = survivorship[i]
        return survivorship
    
    def computeAccumDisFactor(self):
        return np.exp(0.0 - self.riskFreeInterestRate() * self.valuationFrequency() * self.projectionPeriod())

    def computeDiscFactor(self):
        return np.ones(12 * self.projection_term_years + 1) * np.exp(self.riskFreeInterestRate() * self.valuationFrequency())

    def projectFundbyRealWorldScenario(self, scenario, num_inforces = 1000):
        """Project inforce fund values by the real world scenario and the index fund mapping
            N: number of evaluation points in each scenario
            M: number of inforce
            K: number of funds
            L: number of mapped index (fund will be remapped to index)  
            X(i,j, t): i-th inforce, j-th fund value at evluation time point t, where the
                t is within the range of [0, N]

            index_t = index_fund_map^T * inforce_fund_t-1
            
            X(i,j, t) = exp(rw_senario[t] * ) * () 
        Args:
            rw_senario (numpy array): N x L dimension of matrix
            inforce_fund (numpy array): M X K dimension of matrix
            index_fund_map (numpy array): K X L dimension of matrix
            fund_fee_percentage (numpy array): K X 1 dimension of vector
        """
        fund_map_np = self.inforce.fund_mapper.getMappingMatrix().T
        fund_fee_np = self.inforce.fundFeeNumpy()
        inforce_fund = vac.inforce.fundValuesNumpy()

        
        N = scenario.shape[0]
        fund_remain = 1.0 - fund_fee_np[0] / 12.0
        
        start=datetime.now()

        inforces_projected = []
        
        for i in tqdm.tqdm(range(num_inforces)):
            inforce = inforce_fund[i]
            x0 = inforce
            inforce_fund_projected = [deepcopy(x0)]
            
            for i in np.arange(1, N, 1):
                if i == 1:
                    x1 = x0
                else:
                    x1 = x0 * fund_remain 
                w = scenario[i-1].reshape(1, -1)
                for j in np.arange(0, 10, 1):
                    lamda = fund_map_np[j].reshape(-1, 1)
                    beta = w.dot(lamda).ravel()[0] / 12.0
                    try:
                        x1[j] = math.exp(beta) * x1[j]
                    except:
                        import pdb; pdb.set_trace()
                inforce_fund_projected.append(x1)
                x0 = x1    
            inforces_projected.append(np.array(inforce_fund_projected))
            
        inforces_projected = np.array(inforces_projected)
        return inforces_projected
    
    def GMDBCalculation(self, inforcesFundSum, inforcesGBAmount):
        """[summary]

        Args:
            fundSum ([type]): [description]
            gbAmount ([type]): [description]
        """
        dist_factor = self.disc_factor
        GMDB = []
        
        for idx in range(len(inforcesFundSum)):
            gbAmount = inforcesGBAmount[idx]
            fundSum = inforcesFundSum[idx]
            gmdb = [gbAmount]
            gmdb_prev = gbAmount
            for i in range(1, len(fundSum)):
                gmdb_curr = max(fundSum[i], gmdb_prev * dist_factor[i])
                gmdb.append(gmdb_curr)
                gmdb_prev = gmdb_curr
            gmdb = np.array(gmdb)    
            GMDB.append(gmdb)
        
        GMDB = np.array(GMDB)
        survivorship = np.roll(self.survivorship, 1)
        survivorship[0] = 0.0
        GMDB_payout = (GMDB - inforcesFundSum) * (1 - math.exp(0.0 - self.fixed_mortality_rate * self.valuationFrequency()))
        GMDB_payout = GMDB_payout * survivorship
        return GMDB, GMDB_payout
    
    
    def GMMBCalculation(self, inforcesFundSum, GMDB):
        GMMB = np.maximum(inforcesFundSum, 0.75 * GMDB)
        GMMB_payout = np.zeros_like(GMMB)
        GMMB_payout[:,-1] =  GMMB[:,-1] * self.survivorship[-1]
        return GMMB, GMMB_payout
        
        
    def GMWBCalculation(self, inforcesFundSum, GMDB):
        """[summary]

        Args:
            fundSum ([type]): [description]
            gbAmount ([type]): [description]
        """
        GMWB = 0.75 * np.maximum(GMDB * 0.75, inforcesFundSum) 
        
        survivorship = self.survivorship
        fixed_laps_rate = self.fixed_laps_rate * self.valuationFrequency()
        factor = 1.0 - math.exp(0.0 - fixed_laps_rate)
        
        GMWB_payout = []
        for idx in range(len(inforcesFundSum)):
            gmdb = GMDB[idx]
            gmwb = GMWB[idx]
            fundSum = inforcesFundSum[idx]
            gmwb_payout = factor * survivorship * (gmwb - fundSum)
            GMWB_payout.append(gmwb_payout)
        return GMWB, np.array(GMWB_payout)

    
    def PVOfPayoutCalculation(self, totalPayout):
        risk_free_interest_rate = self.risk_free_interest_rate * self.valuationFrequency()
        risk_factor = math.exp(0.0 - risk_free_interest_rate)
        pv_payout = np.zeros_like(totalPayout)
        pv_payout[:,-1] = totalPayout[:,-1]
        num_steps = totalPayout.shape[-1]
        for idx in range(len(totalPayout)):
            for i in range(num_steps - 2, -1, -1):
                pv_payout[idx][i] = pv_payout[idx][i+1] * risk_factor + totalPayout[idx][i]
        return pv_payout
                
        
    def reserveCalculationOne(self, scenarioRW, numOfInforces = 100):
        # Projected fund values of all the selected inforces
        infoces_projected_fund_values = self.projectFundbyRealWorldScenario(scenarioRW, numOfInforces)
        # Compute all the infoces projected total fund values
        infoces_projected_fund_sum = np.sum(infoces_projected_fund_values, axis = 2)
        infoces_gb_amount = self.inforce.inforcesGBAmount()[0:numOfInforces]
        GMDB, GMDB_payout = self.GMDBCalculation(infoces_projected_fund_sum, infoces_gb_amount)
        GMMB, GMMB_payout = self.GMMBCalculation(infoces_projected_fund_sum, GMDB)
        GMWB, GMWB_Payout = self.GMWBCalculation(infoces_projected_fund_sum, GMDB)
        Total_payout = GMDB_payout + GMMB_payout + GMWB_Payout
        pv = self.PVOfPayoutCalculation(Total_payout)
        return GMDB, GMDB_payout,  GMMB, GMMB_payout, GMWB, GMWB_Payout, Total_payout, pv
    
    
    def reserveCalculation(self, numOfInforces):
        
        
if __name__ == "__main__":
    vac = VACalculator(INFORCE_PATH, EXCEL_CALCULATION_PATH)
    scenarioRW = vac.rw_scenarios.getScenario(1)
    vac.reserveCalculation(scenarioRW, 10)
        
    import pdb; pdb.set_trace()

     