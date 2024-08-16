import numpy as np
import pandas as pd
import math
import json
import itertools
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import uncertainties
from uncertainties import unumpy, ufloat
from fair.forward import fair_scm
from fair.inverse import inverse_fair_scm
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#from matplotlib.figure import Figure

def make_base_demand_EJ_KM_II(df_input, growth_rate, efficiency_increase, rerouting = False):  # default arguments value

    ## arguments: (1) input data frame, growth rate, efficiency increase
    ## and returns 2 vectors: baseDemandKM and baseDemandEJ

    FUEL_USE_PER_KM_FACTOR = 2.415545192
    TRACK_TO_SLANT_DISTANCE_FACTOR = 1.165792401
    if rerouting is True:
        REROUTING_INCREASE_FUEL = 1.0088 # based on Dray et al. 2022
    baseDemandKM = np.zeros(41)
    baseDemandEJ = np.zeros(41)

    baseDemandEJ[0:5] = df_input.loc[:, 'DEMAND_EJ_BASE'].values[0:5]
    baseDemandKM[0:5] = df_input.loc[:, 'DEMAND_M_KM_BASE'].values[0:5]

    for year in range(5, 41):
        #baseDemandKM[year] = baseDemandKM[year -1] * (1 + growth_rate)#TRACK_TO_SLANT_DISTANCE_FACTOR
        #baseDemandEJ[year] = baseDemandKM[year]/(10000*TRACK_TO_SLANT_DISTANCE_FACTOR)*FUEL_USE_PER_KM_FACTOR
        baseDemandEJ[year] = baseDemandEJ[year - 1] * (1 + growth_rate)
        baseDemandKM[year] = baseDemandEJ[year] / FUEL_USE_PER_KM_FACTOR * 10000 * TRACK_TO_SLANT_DISTANCE_FACTOR
        FUEL_USE_PER_KM_FACTOR = FUEL_USE_PER_KM_FACTOR * (1 - efficiency_increase)

    if rerouting is True:
        baseDemandEJ *= REROUTING_INCREASE_FUEL
    return baseDemandEJ, baseDemandKM


def make_DACCU_FF_EJ_Tg_II(df_input, baseDemandEJ):
    

    CIVIL_FRACTION                        = 0.95  
    progressionCurve = df_input.loc[:, 'PROGRESSION_CURVE'].values
    SPECIFIC_ENERGY_jetFuel =  0.023174971
    
    DACCU_EJ_vector = np.zeros((3,2,41))
    FF_EJ_vector = np.zeros((3,2,41))
    DACCU_Tg_vector = np.zeros((3,2,41))
    FF_Tg_vector = np.zeros((3,2,41))

    
    for i in range(0, 3): 
        for j in range(0, 2):
        
            DACCU_EJ_vector[i,j] = CIVIL_FRACTION * baseDemandEJ * progressionCurve * (1-j)
            FF_EJ_vector[i,j]  = CIVIL_FRACTION * baseDemandEJ * (1 - (progressionCurve * (1-j)))

            # in mass [Tg/year]
            DACCU_Tg_vector[i,j] = DACCU_EJ_vector[i,j] * 1000 * SPECIFIC_ENERGY_jetFuel 
            FF_Tg_vector[i,j]  = FF_EJ_vector[i,j] * 1000 * SPECIFIC_ENERGY_jetFuel
        
    return DACCU_EJ_vector, FF_EJ_vector, DACCU_Tg_vector, FF_Tg_vector


def make_historic_demand_II(df_input, df_emissions_input, baseDemandEJ, baseDemandKM):

    # Create vectors for DACCU AND FF demand in KM and Tg starting from 1990 up to 2060.
    # Returns 4 vectors: FF and DACCU demand in millions KM, FF and DACCU demand in Tg

    baseDemandEJ_1990 = np.zeros(71)
    baseDemandKM_1990 = np.zeros(71)
    progressionCurve_1990 = np.zeros(71)

    baseDemandEJ_1990[0:30] = df_emissions_input.loc[:, 'HISTORIC_DEMAND_EJ'].values[0:30]
    baseDemandKM_1990[0:30] = df_emissions_input.loc[:, 'HISTORIC_DEMAND_KM'].values[0:30]


    baseDemandEJ_1990[30:71] = baseDemandEJ
    baseDemandKM_1990[30:71] = baseDemandKM

    progressionCurve_1990[0:30] = 0

    progressionCurve_1990[30:71] = df_input.loc[:, 'PROGRESSION_CURVE'].values
    progressionCurve = df_input.loc[:, 'PROGRESSION_CURVE'].values

    DACCU_EJ_vector_1990 = np.zeros((3,2,71))
    FF_EJ_vector_1990 = np.zeros((3,2,71))
    DACCU_Tg_vector_1990 = np.zeros((3,2,71))
    FF_Tg_vector_1990 = np.zeros((3,2,71))

    DACCU_KM_vector_1990 = np.zeros((3,2,71))
    FF_KM_vector_1990 = np.zeros((3,2,71))
    
    CIVIL_FRACTION         = 0.95
    SPECIFIC_ENERGY_jetFuel =  0.023174971 

# Calculating share of demand DACCU and FF in EJ and KM

    for i in range(0, 3): 
        for j in range(0, 2):

            DACCU_EJ_vector_1990[i,j] = CIVIL_FRACTION * baseDemandEJ_1990 * progressionCurve_1990 * (1-j)
            FF_EJ_vector_1990[i,j]  = CIVIL_FRACTION * baseDemandEJ_1990 * (1 - (progressionCurve_1990 * (1-j)))


            DACCU_KM_vector_1990[i,j] = CIVIL_FRACTION * baseDemandKM_1990 * progressionCurve_1990 * (1-j)
            FF_KM_vector_1990[i,j] = CIVIL_FRACTION * baseDemandKM_1990 * (1 - (progressionCurve_1990 * (1-j)))

            # in mass [Tg/year]
            DACCU_Tg_vector_1990[i,j] = DACCU_EJ_vector_1990[i,j] * 1000 * SPECIFIC_ENERGY_jetFuel 
            FF_Tg_vector_1990[i,j]  = FF_EJ_vector_1990[i,j] * 1000 * SPECIFIC_ENERGY_jetFuel

        
        
    return FF_KM_vector_1990, DACCU_KM_vector_1990, FF_Tg_vector_1990, DACCU_Tg_vector_1990

def import_lee():
    """
    Functions to import historical dataset from Lee et al. (2021)
    :return: ERF values from 1990-2018 for all aviation species,
             aviation values (e.g. fuel usage, km traveled for 1990-2018,
             emissions of all aviation species for 1990-2018,
             Sensitivity to emissions factors (constant)
    """
    leedf = pd.read_csv('timeseries_lee2021.csv', sep=';') # import data from csv file
    aviation_2018 = leedf.iloc[2:31,1:12]
    aviation_2018.columns = leedf.iloc[0,1:12]
    aviation_2018.index = leedf.iloc[2:31,0]
    aviation_2018.index = pd.to_datetime(aviation_2018.index)
    emissions_2018 = leedf.iloc[2:31, 13:19]
    emissions_2018.columns = leedf.iloc[0, 13:19]
    emissions_2018.index = leedf.iloc[2:31, 0]
    emissions_2018.index = pd.to_datetime(aviation_2018.index)
    ERF_factors_2018 =  leedf.iloc[12, 20:29]
    ERF_factors_2018.index = leedf.iloc[0, 20:29]
    ERF_2018 = leedf.iloc[12:31,30:40] # isolate only ERF time series
    ERF_2018.columns = leedf.iloc[0, 30:40]  # set column names
    ERF_2018.index = leedf.iloc[12:31,0]
    ERF_2018.index = pd.to_datetime(ERF_2018.index)
    ERF_2018 = ERF_2018.apply(lambda x: x.str.replace(',', '.'))
    aviation_2018 = aviation_2018.apply(lambda x: x.str.replace(',', '.'))
    emissions_2018 = emissions_2018.apply(lambda x: x.str.replace(',', '.'))
    ERF_factors_2018 = ERF_factors_2018.apply(lambda x: x.replace(',', '.'))
    return ERF_2018, aviation_2018, emissions_2018, ERF_factors_2018

## Calculating ERFS and GWP*
def calculate_ERF_uncertainty(df_emissions_input, e_factors):
    index = df_emissions_input.index
    columns = e_factors.index
    # sensitivity to emissions for other species + uncertainties (as in Lee et al. 2021)
    erf_data = np.array([ufloat(34.44, 9.90), ufloat(-18.60,6.90), ufloat(-9.35,3.40), ufloat(-2.80,1.00), ufloat(5.46,8.10),
                     ufloat(100.67, 165.50), ufloat(-19.91,16.00), ufloat(0.0052, 0.0026), ufloat(9.36*10**(-10),6.57*10**(-10))])
    erf_factors = pd.DataFrame(index = columns, columns = ['ERF factors'],
                                 data = erf_data)
    return erf_factors

def make_CO2aviation_hist():
    """
    Makes CO2 emissions from aviation from 1940-2018 from concentrations reported in Lee et al. 2021
    :return: historical aviation CO2 emissions and forcing due to CO2 emissions
    """
    CO2_C_1940_2018 = np.array([0.0042, 0.0078, 0.0113, 0.0149, 0.0187, 0.0227, 0.0269, 0.0314, 0.0362, 0.0413, 0.0468,
                          0.0527, 0.0590, 0.0658, 0.0731, 0.0810, 0.0894, 0.0986, 0.1085, 0.1192, 0.1308, 0.1437,
                          0.1579, 0.1724, 0.1870, 0.2024, 0.2193, 0.2409, 0.2657, 0.2907, 0.3143, 0.3386, 0.3647,
                          0.3916, 0.4162, 0.4404, 0.4643, 0.4908, 0.5185, 0.5475, 0.5762, 0.6038, 0.6319, 0.6598,
                          0.6898, 0.7213, 0.7558, 0.7924, 0.8315, 0.8725, 0.9130, 0.9507, 0.9872, 1.0216, 1.0596,
                          1.0997, 1.1434, 1.1892, 1.2361, 1.2853, 1.3382, 1.3869, 1.4357, 1.4843, 1.5381, 1.5956,
                          1.6530, 1.7123, 1.7704, 1.8227, 1.8803, 1.9401, 2.0004, 2.0633, 2.1291, 2.2002, 2.2737,
                          2.3496, 2.4281])
    CO2_C_1940_2018 += 278
    E1, F1, T1 = inverse_fair_scm(C=CO2_C_1940_2018, rt=0)
    return E1, F1
def calc_ERF_CO2(E, start_year=1990, end_year=2061, class_emissions = True):
    """
    Calculate the ERF of CO2
    :param E: dataframe with future emissions
    :param start_year: start date of future emissions
    :return: forcing of CO2 emissions from start date
    """
    length_date = end_year-1940
    E_CO2_hist = make_CO2aviation_hist()[0]
    E_input = np.zeros((3,2,length_date))
    C_CO2 = np.zeros((3,2, length_date))
    F_CO2 = np.zeros((3,2,length_date))
    T_CO2 = np.zeros((3,2,length_date))

    if class_emissions is True:
        for i in range(0,3):
            for j in range (0,2):
                E_input[i,j] = np.concatenate((E_CO2_hist[:start_year-1940], E.CO2[i,j]/(3.677)), axis = 0)
                C_CO2[i,j], F_CO2[i,j], T_CO2[i,j]= fair_scm(
                    emissions = E_input[i,j],
                    useMultigas= False)
        for x in range(0,length_date-1):
            if (F_CO2[:,:,x+1] < F_CO2[:,:,x]).any():
                F_CO2[:,:,x+1] = F_CO2[:,:,x]

    else:
        for i in range(0,3):
            for j in range (0,2):
                E_input[i,j] = np.concatenate((E_CO2_hist[:start_year-1940], E[i,j]/(3.677)), axis = 0)
                C_CO2[i,j], F_CO2[i,j], T_CO2[i,j]= fair_scm(
                    emissions = E_input[i,j],
                    useMultigas= False)
    return F_CO2[:,:,start_year-1940:]*10**3 #in mW/m2

def calculate_scaled_SAF(old, new, old_err, new_err, perc):
    """
    Function to calculate reduction in emissions through SAF for a blending by 100% when data are available only for less than 100% blendings
    :param old: old emissions per kg (with JetA1 fuels)
    :param new: new emissions per kg (with Zero-CO$_2$ fuels)
    :param old_err: old emissions std  (with Jet A1 fuels)
    :param new_err: old emissions std  (with Zero-CO$_2$ fuels)
    :param perc: percent of blending of SAF
    :return: percent reduction in emissions with Zero-CO$_2$ fuels
    """
    y = 1 - ((old-new)/old)/perc
    y_err = 1- ((old+old_err)-(new+new_err))/(old+old_err)/perc
    err = y_err - y
    if y < 0:
        y = 0
    return ufloat(y, np.absolute(err))

def calculate_EI_DACCU():
    EI_CO2_DACCU_FACTOR = ufloat(0.,
                       0.)  # assuming 100% CO2-free Zero-CO$_2$ fuels, even in life cycle (e.g. thanks to 100% renewable energy)
    EI_BC_DACCU_FACTOR = calculate_scaled_SAF(18.8, 11.4, 0.025, 0.025,
                                    0.41)  # calculated from change in aromatics reported in Voigt et al. (2021), scaled to a 100% FT SAF # FINAL: 3.9+-6.7%
    EI_SO2_DACCU_FACTOR = calculate_scaled_SAF(0.117, 0.057, 0.003, 0.002,
                                     0.41)  # calculated from sulfate content of Voigt et al. (2021) scaled to a 100% FT SAF # FINAL: -25+-11%
    EI_contrails_DACCU_FACTOR = ufloat(0.55, 0.15)  # scaled from Voigt et al. 2021, Kärcher 2018, and Burkhardt et al. 2018
    # calculate_scaled_SAF(4.2*10**15, 2.0*10**15, 0.6*10**15, 0.2*10**15, 1)
    EI_NOx_DACCU_FACTOR = ufloat(0.9, 0.08)  # reported by Jagtap, 2019, Braun-Unkhoff et al., 2017, Blakey et al., 2011
    EI_H2O_DACCU_FACTOR = calculate_scaled_SAF(13.67, 14.36, 0.14, 0.02,
                                     0.41)  # calculated from hydrogen content of Voigt et al. (2021) scaled to a 100% FT SAF # FINAL: 18+-68%

    return EI_CO2_DACCU_FACTOR, EI_BC_DACCU_FACTOR, EI_SO2_DACCU_FACTOR, EI_contrails_DACCU_FACTOR, EI_NOx_DACCU_FACTOR, EI_H2O_DACCU_FACTOR

def calculate_ERF(df_emissions_input, e_factors):
    """
    Function to calculate ERF from sensitivity to emissions reported in Lee et al. 2021
    :param df_emissions_input: dataframe with emissions
    :param e_factors: sensitivity to emissions reported in Lee et al. 2021
    :return: ERF of each species in each year
    """
    #f_CO2 = ufloat(0.035, 0.00057) # sensitivity to emissions for CO2
    index = df_emissions_input.index
    columns = e_factors.index
    # sensitivity to emissions for other species + uncertainties (as in Lee et al. 2021)
    erf_data = np.array([ufloat(34.44, 9.90), ufloat(-18.60,6.90), ufloat(-9.35,3.40), ufloat(-2.80,1.00), ufloat(5.46,8.10),
                     ufloat(100.67, 165.50), ufloat(-19.91,16.00), ufloat(0.0052, 0.0026), ufloat(9.36*10**(-10),6.57*10**(-10))])
    erf_factors = pd.DataFrame(index = columns, columns = ['ERF factors'],
                                 data = erf_data)
    ERF_df = pd.DataFrame(index=index, columns=columns)
    ERF_df = ERF_df.fillna(0.)
    #ERF_df['CO2'] = df['CO2'].values*f_CO2
    ERF_df['CO2'] = calc_ERF_CO2(df_emissions_input, start_year=df_emissions_input.index[0].year)
    ERF_df['O3 short'] = df_emissions_input['NOx'].values * erf_factors.loc['O3 short', :].values
    ERF_df['CH4'] = df_emissions_input['NOx'].values * erf_factors.loc['CH4', :].values
    ERF_df['O3 long'] = df_emissions_input['NOx'].values * erf_factors.loc['O3 long', :].values
    ERF_df['SWV'] = df_emissions_input['NOx'].values * erf_factors.loc['SWV', :].values
    ERF_df['netNOx'] = df_emissions_input['NOx'].values * erf_factors.loc['netNOx', :].values# here actually 6,4
    ERF_df['BC'] = df_emissions_input['BC'].values * erf_factors.loc['BC', :].values
    ERF_df['SO4'] = df_emissions_input['SO2'].values * erf_factors.loc['SO4', :].values
    ERF_df['H2O'] = df_emissions_input['H2O'].values * erf_factors.loc['H2O', :].values
    ERF_df['Contrails and C-C'] = df_emissions_input['Contrail'].values * erf_factors.loc['Contrails and C-C', :].values
    ERF_df['non-CO2'] = ERF_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C']].sum(axis=1)
    ERF_df['Tot'] = ERF_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C', 'CO2']].sum(axis=1)
    return ERF_df


def make_emissions_and_ERF(df_input, df_emissions_input, FF_KM_vector_1990,
                   DACCU_KM_vector_1990,FF_Tg_vector_1990, DACCU_Tg_vector_1990, scenario = "Baseline",
                      CC_efficacy = 1, erf_uncertain = None, uncertainty_daccu = False, rerouting = False):
    if rerouting is True:
        CC_efficacy = 0.5

    progressionCurve = df_input.loc[:, 'PROGRESSION_CURVE'].values
    
    #Initialising arrays for emissions calculations
    EI_CO2_FF = 3.16
    EI_NOX_FF = df_emissions_input.loc[:, 'EI_NOx'].values
    EI_BC_FF =  df_emissions_input.loc[:, 'EI_BC'].values
    EI_SO2_FF =  df_emissions_input.loc[:, 'EI_SO2'].values
    EI_H2O_FF =  df_emissions_input.loc[:, 'EI_H2O'].values
    TRANSIENT_FACTOR =  df_emissions_input.loc[:, 'TRANSIENT_FACTOR'].values

    #DACCU Emissions Indices relative to FF Emissions

    if uncertainty_daccu is True:
        EI_CO2_DACCU_FACTOR, EI_BC_DACCU_FACTOR, EI_SO2_DACCU_FACTOR, \
            EI_CC_DACCU_FACTOR, EI_NOX_DACCU_FACTOR, EI_H2O_DACCU_FACTOR = calculate_EI_DACCU()
    else:
        EI_CO2_DACCU_FACTOR = 0
        EI_NOX_DACCU_FACTOR = 0.9
        EI_BC_DACCU_FACTOR = 0.04
        EI_SO2_DACCU_FACTOR = 0
        EI_H2O_DACCU_FACTOR = 1.12
        EI_CC_DACCU_FACTOR = 0.55

    EI_CO2_DACCU = EI_CO2_FF * EI_CO2_DACCU_FACTOR
    EI_NOX_DACCU = EI_NOX_FF * EI_NOX_DACCU_FACTOR
    EI_BC_DACCU = EI_BC_FF * EI_BC_DACCU_FACTOR
    EI_SO2_DACCU = EI_SO2_FF * EI_SO2_DACCU_FACTOR
    EI_H2O_DACCU = EI_H2O_FF * EI_H2O_DACCU_FACTOR

    # Calculating Scalars

    if uncertainty_daccu is True:
        SCALAR_CO2_DACCU_MIN = np.zeros((3, 2, 71))
        SCALAR_NOX_DACCU_MIN = np.zeros((3, 2, 71))
        SCALAR_BC_DACCU_MIN = np.zeros((3, 2, 71))
        SCALAR_SO2_DACCU_MIN = np.zeros((3, 2, 71))
        SCALAR_H2O_DACCU_MIN = np.zeros((3, 2, 71))
        SCALAR_CONTRAILS_DACCU_MIN = np.zeros((3, 2, 71))
        SCALAR_CO2_DACCU_MAX = np.zeros((3, 2, 71))
        SCALAR_NOX_DACCU_MAX = np.zeros((3, 2, 71))
        SCALAR_BC_DACCU_MAX = np.zeros((3, 2, 71))
        SCALAR_SO2_DACCU_MAX = np.zeros((3, 2, 71))
        SCALAR_H2O_DACCU_MAX = np.zeros((3, 2, 71))
        SCALAR_CONTRAILS_DACCU_MAX = np.zeros((3, 2, 71))

    class SCALAR:
        def __init__(self, SCALAR_CO2, SCALAR_NOX, SCALAR_BC, SCALAR_SO2, SCALAR_H2O, SCALAR_CONTRAILS,
                     SCALAR_CO2_MIN = None, SCALAR_NOX_MIN=None, SCALAR_BC_MIN=None, SCALAR_SO2_MIN=None,
                     SCALAR_H2O_MIN=None, SCALAR_CONTRAILS_MIN=None, SCALAR_CO2_MAX = None, SCALAR_NOX_MAX=None,
                     SCALAR_BC_MAX=None, SCALAR_SO2_MAX=None, SCALAR_H2O_MAX=None,
                     SCALAR_CONTRAILS_MAX=None):
            self.CO2 = SCALAR_CO2
            self.NOX = SCALAR_NOX
            self.BC = SCALAR_BC
            self.SO2 = SCALAR_SO2
            self.H2O = SCALAR_H2O
            self.CONTRAILS = SCALAR_CONTRAILS

            if SCALAR_CO2_MIN is not None:
                self.CO2_MIN = SCALAR_CO2_MIN
            if SCALAR_CO2_MAX is not None:
                self.CO2_MAX = SCALAR_CO2_MAX

            if SCALAR_NOX_MIN is not None:
                self.NOX_MIN = SCALAR_NOX_MIN
            if SCALAR_NOX_MAX is not None:
                self.NOX_MAX = SCALAR_NOX_MAX

            if SCALAR_BC_MIN is not None:
                self.BC_MIN = SCALAR_BC_MIN
            if SCALAR_BC_MAX is not None:
                self.BC_MAX = SCALAR_BC_MAX

            if SCALAR_SO2_MIN is not None:
                self.SO2_MIN = SCALAR_SO2_MIN
            if SCALAR_SO2_MAX is not None:
                self.SO2_MAX = SCALAR_SO2_MAX

            if SCALAR_H2O_MIN is not None:
                self.H2O_MIN = SCALAR_H2O_MIN
            if SCALAR_H2O_MAX is not None:
                self.H2O_MAX = SCALAR_H2O_MAX

            if SCALAR_CONTRAILS_MIN is not None:
                self.CONTRAILS_MIN = SCALAR_CONTRAILS_MIN
            if SCALAR_CONTRAILS_MAX is not None:
                self.CONTRAILS_MAX = SCALAR_CONTRAILS_MAX

    SCALAR_CO2_FF = np.zeros((3,2,71))
    SCALAR_NOX_FF = np.zeros((3,2,71))
    SCALAR_BC_FF = np.zeros((3,2,71))
    SCALAR_SO2_FF = np.zeros((3,2,71))
    SCALAR_H2O_FF = np.zeros((3,2,71))
    SCALAR_CONTRAILS_FF = np.zeros((3,2,71))

    SCALAR_CO2_DACCU = np.zeros((3,2,71))
    SCALAR_NOX_DACCU = np.zeros((3,2,71))
    SCALAR_BC_DACCU = np.zeros((3,2,71))
    SCALAR_SO2_DACCU = np.zeros((3,2,71))
    SCALAR_H2O_DACCU = np.zeros((3,2,71))
    SCALAR_CONTRAILS_DACCU = np.zeros((3,2,71))

    for i in range(0, 3): 
        for j in range(0, 2):
            SCALAR_CO2_FF[i,j] = FF_Tg_vector_1990[i,j] * EI_CO2_FF / 1000
            SCALAR_NOX_FF[i,j] = FF_Tg_vector_1990[i,j] * EI_NOX_FF * (14/46) / 1000
            SCALAR_BC_FF[i,j] = FF_Tg_vector_1990[i,j] * EI_BC_FF / 1000
            SCALAR_SO2_FF[i,j] = FF_Tg_vector_1990[i,j] * EI_SO2_FF / 1000
            SCALAR_H2O_FF[i,j] = FF_Tg_vector_1990[i,j] * EI_H2O_FF / 1000 
            SCALAR_CONTRAILS_FF[i,j] = FF_KM_vector_1990[i,j] * (10**6) * CC_efficacy

            if uncertainty_daccu is True:
                SCALAR_CO2_DACCU[i, j] = DACCU_Tg_vector_1990[i, j] * unumpy.nominal_values(EI_CO2_DACCU) / 1000
                SCALAR_NOX_DACCU[i, j] = DACCU_Tg_vector_1990[i, j] * unumpy.nominal_values(EI_NOX_DACCU) * (14 / 46) / 1000
                SCALAR_BC_DACCU[i, j] = DACCU_Tg_vector_1990[i, j] * unumpy.nominal_values(EI_BC_DACCU) / 1000
                SCALAR_SO2_DACCU[i, j] = DACCU_Tg_vector_1990[i, j] * unumpy.nominal_values(EI_SO2_DACCU) / 1000
                SCALAR_H2O_DACCU[i, j] = DACCU_Tg_vector_1990[i, j] * unumpy.nominal_values(EI_H2O_DACCU) / 1000
                SCALAR_CO2_DACCU_MIN[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_CO2_DACCU) -
                                                                           unumpy.std_devs(EI_CO2_DACCU)) / 1000
                SCALAR_NOX_DACCU_MIN[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_NOX_DACCU) -
                                                                           unumpy.std_devs(EI_NOX_DACCU))* (14 / 46) / 1000
                SCALAR_BC_DACCU_MIN[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_BC_DACCU) \
                                            - unumpy.std_devs(EI_BC_DACCU))                / 1000
                SCALAR_SO2_DACCU_MIN[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_SO2_DACCU)  \
                                                                           - unumpy.std_devs(EI_SO2_DACCU)) / 1000
                SCALAR_H2O_DACCU_MIN[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_H2O_DACCU)  -  \
                                                                           unumpy.std_devs(EI_H2O_DACCU))/ 1000
                SCALAR_CO2_DACCU_MAX[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_CO2_DACCU) +
                                                                           unumpy.std_devs(EI_CO2_DACCU)) / 1000
                SCALAR_NOX_DACCU_MAX[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_NOX_DACCU) +
                                                                           unumpy.std_devs(EI_NOX_DACCU))* (14 / 46) / 1000
                SCALAR_BC_DACCU_MAX[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_BC_DACCU) \
                                            + unumpy.std_devs(EI_BC_DACCU))                / 1000
                SCALAR_SO2_DACCU_MAX[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_SO2_DACCU)  \
                                                                           + unumpy.std_devs(EI_SO2_DACCU)) / 1000
                SCALAR_H2O_DACCU_MAX[i, j] = DACCU_Tg_vector_1990[i, j] * (unumpy.nominal_values(EI_H2O_DACCU)  +  \
                                                                           unumpy.std_devs(EI_H2O_DACCU))/ 1000
                SCALAR_CONTRAILS_DACCU[i, j] = DACCU_KM_vector_1990[i, j] * (10 ** 6) * CC_efficacy
                SCALAR_CONTRAILS_DACCU_MIN[i, j] = DACCU_KM_vector_1990[i, j] * (10 ** 6) * CC_efficacy
                SCALAR_CONTRAILS_DACCU_MAX[i, j] = DACCU_KM_vector_1990[i, j] * (10 ** 6) * CC_efficacy
                SCALAR_DACCU = SCALAR(SCALAR_CO2_DACCU, SCALAR_NOX_DACCU, SCALAR_BC_DACCU, SCALAR_SO2_DACCU, SCALAR_H2O_DACCU,
                             SCALAR_CONTRAILS_DACCU, SCALAR_CO2_DACCU_MIN, SCALAR_NOX_DACCU_MIN, SCALAR_BC_DACCU_MIN, SCALAR_SO2_DACCU_MIN,
                             SCALAR_H2O_DACCU_MIN, SCALAR_CONTRAILS_DACCU_MIN,
                                      SCALAR_CO2_DACCU_MAX, SCALAR_NOX_DACCU_MAX, SCALAR_BC_DACCU_MAX,
                         SCALAR_SO2_DACCU_MAX, SCALAR_H2O_DACCU_MAX, SCALAR_CONTRAILS_DACCU_MAX)
            else:
                SCALAR_CO2_DACCU[i, j] = DACCU_Tg_vector_1990[i, j] * unumpy.nominal_values(EI_CO2_DACCU) / 1000
                SCALAR_NOX_DACCU[i,j] = DACCU_Tg_vector_1990[i,j] * EI_NOX_DACCU * (14/46) / 1000
                SCALAR_BC_DACCU[i,j] = DACCU_Tg_vector_1990[i,j] * EI_BC_DACCU / 1000
                SCALAR_SO2_DACCU[i,j] = DACCU_Tg_vector_1990[i,j] * EI_SO2_DACCU / 1000
                SCALAR_H2O_DACCU[i,j] = DACCU_Tg_vector_1990[i,j] * EI_H2O_DACCU / 1000
                SCALAR_CONTRAILS_DACCU[i,j] = DACCU_KM_vector_1990[i,j] * (10**6) * CC_efficacy

    # ERF Factors

    if erf_uncertain is not None:
        ERF_factors_disaggregated = pd.DataFrame(data = unumpy.nominal_values(erf_uncertain), index = erf_uncertain.index,
                                                 columns=['Best estimate'])
        ERF_factors_disaggregated['Min'] = unumpy.nominal_values(erf_uncertain) - unumpy.std_devs(erf_uncertain)
        ERF_factors_disaggregated['Max'] = unumpy.nominal_values(erf_uncertain) + unumpy.std_devs(erf_uncertain)
        ERF_FACTOR_O3_SHORT = ERF_factors_disaggregated.loc['O3 short', 'Best estimate']
        ERF_FACTOR_O3_SHORT_MIN = ERF_factors_disaggregated.loc['O3 short', 'Min']
        ERF_FACTOR_O3_SHORT_MAX = ERF_factors_disaggregated.loc['O3 short', 'Max']
        ERF_FACTOR_CH4 = ERF_factors_disaggregated.loc['CH4', 'Best estimate']
        ERF_FACTOR_CH4_MIN = ERF_factors_disaggregated.loc['CH4', 'Min']
        ERF_FACTOR_CH4_MAX = ERF_factors_disaggregated.loc['CH4', 'Max']
        ERF_FACTOR_O3_LONG = ERF_factors_disaggregated.loc['O3 long', 'Best estimate']
        ERF_FACTOR_O3_LONG_MIN = ERF_factors_disaggregated.loc['O3 long', 'Min']
        ERF_FACTOR_O3_LONG_MAX = ERF_factors_disaggregated.loc['O3 long', 'Max']
        ERF_FACTOR_SWV = ERF_factors_disaggregated.loc['SWV', 'Best estimate']
        ERF_FACTOR_SWV_MIN = ERF_factors_disaggregated.loc['SWV', 'Min']
        ERF_FACTOR_SWV_MAX = ERF_factors_disaggregated.loc['SWV', 'Max']
        ERF_FACTOR_BC = ERF_factors_disaggregated.loc['BC', 'Best estimate']
        ERF_FACTOR_BC_MIN = ERF_factors_disaggregated.loc['BC', 'Min']
        ERF_FACTOR_BC_MAX = ERF_factors_disaggregated.loc['BC', 'Max']
        ERF_FACTOR_SO4 = ERF_factors_disaggregated.loc['SO4', 'Best estimate']
        ERF_FACTOR_SO4_MIN = ERF_factors_disaggregated.loc['SO4', 'Min']
        ERF_FACTOR_SO4_MAX = ERF_factors_disaggregated.loc['SO4', 'Max']
        ERF_FACTOR_H2O = ERF_factors_disaggregated.loc['H2O', 'Best estimate']
        ERF_FACTOR_H2O_MIN = ERF_factors_disaggregated.loc['H2O', 'Min']
        ERF_FACTOR_H2O_MAX = ERF_factors_disaggregated.loc['H2O', 'Max']
        ERF_FACTOR_CONTRAILS_FF = ERF_factors_disaggregated.loc['Contrails and C-C', 'Best estimate']
        ERF_FACTOR_CONTRAILS_FF_MIN = ERF_factors_disaggregated.loc['Contrails and C-C', 'Min']
        ERF_FACTOR_CONTRAILS_FF_MAX = ERF_factors_disaggregated.loc['Contrails and C-C', 'Max']
        ERF_FACTOR_CONTRAILS_DACCU = ERF_FACTOR_CONTRAILS_FF * unumpy.nominal_values(EI_CC_DACCU_FACTOR)
        ERF_FACTOR_CONTRAILS_DACCU_MIN = ERF_FACTOR_CONTRAILS_FF_MIN * (unumpy.nominal_values(EI_CC_DACCU_FACTOR) -
                                                                    unumpy.std_devs(EI_CC_DACCU_FACTOR))
        ERF_FACTOR_CONTRAILS_DACCU_MAX = ERF_FACTOR_CONTRAILS_FF_MAX * (unumpy.nominal_values(EI_CC_DACCU_FACTOR) +
                                                                    unumpy.std_devs(EI_CC_DACCU_FACTOR))
    else:
        ERF_FACTOR_O3_SHORT = 34.44295775
        ERF_FACTOR_CH4 = -18.69308608
        ERF_FACTOR_O3_LONG = -9.34654304192212
        ERF_FACTOR_SWV = -2.803962913
        ERF_FACTOR_BC = 100.6711409
        ERF_FACTOR_SO4 = -19.90950226
        ERF_FACTOR_H2O = 0.005178827
        ERF_FACTOR_CONTRAILS_FF = 9.359503783 * (10**-10)
        ERF_FACTOR_CONTRAILS_DACCU = ERF_FACTOR_CONTRAILS_FF * EI_CC_DACCU_FACTOR

    #ERF Calculations________________________________________
    ERF_O3_SHORT_FF = np.zeros((3,2,71))
    ERF_CH4_FF = np.zeros((3,2,71))
    ERF_O3_LONG_FF = np.zeros((3,2,71))
    ERF_SWV_FF = np.zeros((3,2,71))
    ERF_BC_FF = np.zeros((3,2,71))
    ERF_SO4_FF = np.zeros((3,2,71))
    ERF_H2O_FF = np.zeros((3,2,71))
    ERF_CC_FF = np.zeros((3,2,71))


    ERF_O3_SHORT_DACCU = np.zeros((3,2,71))
    ERF_CH4_DACCU = np.zeros((3,2,71))
    ERF_O3_LONG_DACCU = np.zeros((3,2,71))
    ERF_SWV_DACCU = np.zeros((3,2,71))
    ERF_BC_DACCU = np.zeros((3,2,71))
    ERF_SO4_DACCU = np.zeros((3,2,71))
    ERF_H2O_DACCU = np.zeros((3,2,71))
    ERF_CC_DACCU = np.zeros((3,2,71))

    if erf_uncertain is not None:
        ERF_O3_SHORT_FF_MIN = np.zeros((3, 2, 71))
        ERF_CH4_FF_MIN = np.zeros((3, 2, 71))
        ERF_O3_LONG_FF_MIN = np.zeros((3, 2, 71))
        ERF_SWV_FF_MIN = np.zeros((3, 2, 71))
        ERF_BC_FF_MIN = np.zeros((3, 2, 71))
        ERF_SO4_FF_MIN = np.zeros((3, 2, 71))
        ERF_H2O_FF_MIN = np.zeros((3, 2, 71))
        ERF_CC_FF_MIN = np.zeros((3, 2, 71))

        ERF_O3_SHORT_FF_MAX = np.zeros((3, 2, 71))
        ERF_CH4_FF_MAX = np.zeros((3, 2, 71))
        ERF_O3_LONG_FF_MAX = np.zeros((3, 2, 71))
        ERF_SWV_FF_MAX = np.zeros((3, 2, 71))
        ERF_BC_FF_MAX = np.zeros((3, 2, 71))
        ERF_SO4_FF_MAX = np.zeros((3, 2, 71))
        ERF_H2O_FF_MAX = np.zeros((3, 2, 71))
        ERF_CC_FF_MAX = np.zeros((3, 2, 71))

        if uncertainty_daccu is True:
            ERF_O3_SHORT_DACCU_MIN = np.zeros((3, 2, 71))
            ERF_CH4_DACCU_MIN = np.zeros((3, 2, 71))
            ERF_O3_LONG_DACCU_MIN = np.zeros((3, 2, 71))
            ERF_SWV_DACCU_MIN = np.zeros((3, 2, 71))
            ERF_BC_DACCU_MIN = np.zeros((3, 2, 71))
            ERF_SO4_DACCU_MIN = np.zeros((3, 2, 71))
            ERF_H2O_DACCU_MIN = np.zeros((3, 2, 71))
            ERF_CC_DACCU_MIN = np.zeros((3, 2, 71))

            ERF_O3_SHORT_DACCU_MAX = np.zeros((3, 2, 71))
            ERF_CH4_DACCU_MAX = np.zeros((3, 2, 71))
            ERF_O3_LONG_DACCU_MAX = np.zeros((3, 2, 71))
            ERF_SWV_DACCU_MAX = np.zeros((3, 2, 71))
            ERF_BC_DACCU_MAX = np.zeros((3, 2, 71))
            ERF_SO4_DACCU_MAX = np.zeros((3, 2, 71))
            ERF_H2O_DACCU_MAX = np.zeros((3, 2, 71))
            ERF_CC_DACCU_MAX = np.zeros((3, 2, 71))

    for i in range(0, 3): 
        for j in range(0, 2):
            ERF_O3_SHORT_FF[i,j] = SCALAR_NOX_FF[i,j] * ERF_FACTOR_O3_SHORT * TRANSIENT_FACTOR
            ERF_CH4_FF[i,j] = SCALAR_NOX_FF[i,j] * ERF_FACTOR_CH4 * TRANSIENT_FACTOR
            ERF_O3_LONG_FF[i,j] = SCALAR_NOX_FF[i,j] * ERF_FACTOR_O3_LONG * TRANSIENT_FACTOR
            ERF_SWV_FF[i,j] = SCALAR_NOX_FF[i,j] * ERF_FACTOR_SWV * TRANSIENT_FACTOR
            ERF_BC_FF[i,j] = SCALAR_BC_FF[i,j] * ERF_FACTOR_BC
            ERF_SO4_FF[i,j] = SCALAR_SO2_FF[i,j] * ERF_FACTOR_SO4 
            ERF_H2O_FF[i,j] = SCALAR_H2O_FF[i,j] * ERF_FACTOR_H2O
            ERF_CC_FF[i,j] = SCALAR_CONTRAILS_FF[i,j] * ERF_FACTOR_CONTRAILS_FF

            if erf_uncertain is not None:
                ERF_O3_SHORT_FF_MIN[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_O3_SHORT_MIN * TRANSIENT_FACTOR
                ERF_CH4_FF_MIN[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_CH4_MIN * TRANSIENT_FACTOR
                ERF_O3_LONG_FF_MIN[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_O3_LONG_MIN * TRANSIENT_FACTOR
                ERF_SWV_FF_MIN[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_SWV_MIN * TRANSIENT_FACTOR
                ERF_BC_FF_MIN[i, j] = SCALAR_BC_FF[i, j] * ERF_FACTOR_BC_MIN
                ERF_SO4_FF_MIN[i, j] = SCALAR_SO2_FF[i, j] * ERF_FACTOR_SO4_MIN
                ERF_H2O_FF_MIN[i, j] = SCALAR_H2O_FF[i, j] * ERF_FACTOR_H2O_MIN
                ERF_CC_FF_MIN[i, j] = SCALAR_CONTRAILS_FF[i, j] * ERF_FACTOR_CONTRAILS_FF_MIN

                ERF_O3_SHORT_FF_MAX[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_O3_SHORT_MAX * TRANSIENT_FACTOR
                ERF_CH4_FF_MAX[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_CH4_MAX * TRANSIENT_FACTOR
                ERF_O3_LONG_FF_MAX[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_O3_LONG_MAX * TRANSIENT_FACTOR
                ERF_SWV_FF_MAX[i, j] = SCALAR_NOX_FF[i, j] * ERF_FACTOR_SWV_MAX * TRANSIENT_FACTOR
                ERF_BC_FF_MAX[i, j] = SCALAR_BC_FF[i, j] * ERF_FACTOR_BC_MAX
                ERF_SO4_FF_MAX[i, j] = SCALAR_SO2_FF[i, j] * ERF_FACTOR_SO4_MAX
                ERF_H2O_FF_MAX[i, j] = SCALAR_H2O_FF[i, j] * ERF_FACTOR_H2O_MAX
                ERF_CC_FF_MAX[i, j] = SCALAR_CONTRAILS_FF[i, j] * ERF_FACTOR_CONTRAILS_FF_MAX

            ERF_O3_SHORT_DACCU[i,j] = SCALAR_NOX_DACCU[i,j] * ERF_FACTOR_O3_SHORT * TRANSIENT_FACTOR
            ERF_CH4_DACCU[i,j] = SCALAR_NOX_DACCU[i,j] * ERF_FACTOR_CH4 * TRANSIENT_FACTOR
            ERF_O3_LONG_DACCU[i,j] = SCALAR_NOX_DACCU[i,j] * ERF_FACTOR_O3_LONG * TRANSIENT_FACTOR
            ERF_SWV_DACCU[i,j] = SCALAR_NOX_DACCU[i,j] * ERF_FACTOR_SWV * TRANSIENT_FACTOR
            ERF_BC_DACCU[i,j] = SCALAR_BC_DACCU[i,j] * ERF_FACTOR_BC
            ERF_SO4_DACCU[i,j] = SCALAR_SO2_DACCU[i,j] * ERF_FACTOR_SO4
            ERF_H2O_DACCU[i,j] = SCALAR_H2O_DACCU[i,j] * ERF_FACTOR_H2O
            ERF_CC_DACCU[i,j] = SCALAR_CONTRAILS_DACCU[i,j] * ERF_FACTOR_CONTRAILS_DACCU

            if uncertainty_daccu is True:
                ERF_O3_SHORT_DACCU[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_O3_SHORT * TRANSIENT_FACTOR
                ERF_CH4_DACCU[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_CH4 * TRANSIENT_FACTOR
                ERF_O3_LONG_DACCU[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_O3_LONG * TRANSIENT_FACTOR
                ERF_SWV_DACCU[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_SWV * TRANSIENT_FACTOR
                ERF_BC_DACCU[i, j] = SCALAR_DACCU.BC[i, j] * ERF_FACTOR_BC
                ERF_SO4_DACCU[i, j] = SCALAR_DACCU.SO2[i, j] * ERF_FACTOR_SO4
                ERF_H2O_DACCU[i, j] = SCALAR_DACCU.H2O[i, j] * ERF_FACTOR_H2O
                ERF_CC_DACCU[i, j] = SCALAR_DACCU.CONTRAILS[i, j] * ERF_FACTOR_CONTRAILS_DACCU

                ERF_O3_SHORT_DACCU_MIN[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_O3_SHORT_MIN * TRANSIENT_FACTOR
                ERF_CH4_DACCU_MIN[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_CH4_MIN * TRANSIENT_FACTOR
                ERF_O3_LONG_DACCU_MIN[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_O3_LONG_MIN * TRANSIENT_FACTOR
                ERF_SWV_DACCU_MIN[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_SWV_MIN * TRANSIENT_FACTOR
                ERF_BC_DACCU_MIN[i, j] = SCALAR_DACCU.BC[i, j] * ERF_FACTOR_BC_MIN
                ERF_SO4_DACCU_MIN[i, j] = SCALAR_DACCU.SO2[i, j] * ERF_FACTOR_SO4_MIN
                ERF_H2O_DACCU_MIN[i, j] = SCALAR_DACCU.H2O[i, j] * ERF_FACTOR_H2O_MIN
                ERF_CC_DACCU_MIN[i, j] = SCALAR_DACCU.CONTRAILS[i, j] * ERF_FACTOR_CONTRAILS_DACCU_MIN

                ERF_O3_SHORT_DACCU_MAX[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_O3_SHORT_MAX * TRANSIENT_FACTOR
                ERF_CH4_DACCU_MAX[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_CH4_MAX * TRANSIENT_FACTOR
                ERF_O3_LONG_DACCU_MAX[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_O3_LONG_MAX * TRANSIENT_FACTOR
                ERF_SWV_DACCU_MAX[i, j] = SCALAR_DACCU.NOX[i, j] * ERF_FACTOR_SWV_MAX * TRANSIENT_FACTOR
                ERF_BC_DACCU_MAX[i, j] = SCALAR_DACCU.BC[i, j] * ERF_FACTOR_BC_MAX
                ERF_SO4_DACCU_MAX[i, j] = SCALAR_DACCU.SO2[i, j] * ERF_FACTOR_SO4_MAX
                ERF_H2O_DACCU_MAX[i, j] = SCALAR_DACCU.H2O[i, j] * ERF_FACTOR_H2O_MAX
                ERF_CC_DACCU_MAX[i, j] = SCALAR_DACCU.CONTRAILS[i, j] * ERF_FACTOR_CONTRAILS_DACCU_MAX

    # Total ERFs
    ERF_O3_SHORT_TOTAL = ERF_O3_SHORT_FF + ERF_O3_SHORT_DACCU
    ERF_CH4_TOTAL = ERF_CH4_FF + ERF_CH4_DACCU
    ERF_O3_LONG_TOTAL = ERF_O3_LONG_FF + ERF_O3_LONG_DACCU 
    ERF_SWV_TOTAL = ERF_SWV_FF + ERF_SWV_DACCU

    ERF_TOTAL_NOX = ERF_O3_SHORT_TOTAL + ERF_CH4_TOTAL + ERF_O3_LONG_TOTAL + ERF_SWV_TOTAL
    ERF_BC_TOTAL = ERF_BC_FF + ERF_BC_DACCU
    ERF_SO4_TOTAL = ERF_SO4_FF + ERF_SO4_DACCU
    ERF_H2O_TOTAL = ERF_H2O_FF + ERF_H2O_DACCU
    ERF_CC_TOTAL = ERF_CC_FF + ERF_CC_DACCU

    if uncertainty_daccu is False:
        SCALAR_DACCU = SCALAR(SCALAR_CO2_DACCU, SCALAR_NOX_DACCU, SCALAR_BC_DACCU, SCALAR_SO2_DACCU, SCALAR_H2O_DACCU,
                              SCALAR_CONTRAILS_DACCU)
    SCALAR_FF = SCALAR(SCALAR_CO2_FF, SCALAR_NOX_FF, SCALAR_BC_FF, SCALAR_SO2_FF, SCALAR_H2O_FF,
                          SCALAR_CONTRAILS_FF)
    SCALAR_TOTAL = SCALAR(SCALAR_CO2_FF+SCALAR_CO2_DACCU, SCALAR_NOX_FF+SCALAR_NOX_DACCU, SCALAR_BC_FF+SCALAR_BC_DACCU, SCALAR_SO2_FF+SCALAR_SO2_DACCU,
                          SCALAR_H2O_FF+SCALAR_H2O_DACCU, SCALAR_CONTRAILS_FF + SCALAR_CONTRAILS_DACCU)

    ERF_CO2_FF = calc_ERF_CO2(SCALAR_FF)
    ERF_CO2_DACCU = calc_ERF_CO2(SCALAR_DACCU)
    ERF_CO2_TOTAL = calc_ERF_CO2(SCALAR_TOTAL)
    class ERF():
        def __init__(self, CO2, O3_SHORT, CH4, O3_LONG, SWV, BC, SO4, H2O, CC):
            self.CO2 = CO2
            self.O3_SHORT = O3_SHORT
            self.CH4 = CH4
            self.O3_LONG = O3_LONG
            self.SWV = SWV
            self.netNOX = O3_SHORT + CH4 + O3_LONG + SWV
            self.BC = BC
            self.SO4 = SO4
            self.H2O = H2O
            self.CC = CC

    ERF_FF = ERF(ERF_CO2_FF, ERF_O3_SHORT_FF, ERF_CH4_FF, ERF_O3_LONG_FF, ERF_SWV_FF, ERF_BC_FF, ERF_SO4_FF, ERF_H2O_FF, ERF_CC_FF)
    ERF_DACCU = ERF(ERF_CO2_DACCU, ERF_O3_SHORT_DACCU, ERF_CH4_DACCU, ERF_O3_LONG_DACCU, ERF_SWV_DACCU, ERF_BC_DACCU, ERF_SO4_DACCU, ERF_H2O_DACCU, ERF_CC_DACCU)
    ERF_TOTAL = ERF(ERF_CO2_TOTAL, ERF_O3_SHORT_TOTAL, ERF_CH4_TOTAL, ERF_O3_LONG_TOTAL, ERF_SWV_TOTAL, ERF_BC_TOTAL, ERF_SO4_TOTAL, ERF_H2O_TOTAL, ERF_CC_TOTAL)

    if erf_uncertain is not None:
        ERF_O3_SHORT_TOTAL_MIN = ERF_O3_SHORT_FF_MIN + ERF_O3_SHORT_DACCU_MIN
        ERF_CH4_TOTAL_MIN = ERF_CH4_FF_MIN + ERF_CH4_DACCU_MIN
        ERF_O3_LONG_TOTAL_MIN = ERF_O3_LONG_FF_MIN + ERF_O3_LONG_DACCU_MIN
        ERF_SWV_TOTAL_MIN = ERF_SWV_FF_MIN + ERF_SWV_DACCU_MIN

        ERF_TOTAL_NOX_MIN = ERF_O3_SHORT_TOTAL_MIN + ERF_CH4_TOTAL_MIN + ERF_O3_LONG_TOTAL_MIN + ERF_SWV_TOTAL_MIN
        ERF_BC_TOTAL_MIN = ERF_BC_FF_MIN + ERF_BC_DACCU_MIN
        ERF_SO4_TOTAL_MIN = ERF_SO4_FF_MIN + ERF_SO4_DACCU_MIN
        ERF_H2O_TOTAL_MIN = ERF_H2O_FF_MIN + ERF_H2O_DACCU_MIN
        ERF_CC_TOTAL_MIN = CC_efficacy * (ERF_CC_FF_MIN + ERF_CC_DACCU_MIN)

        ERF_O3_SHORT_TOTAL_MAX = ERF_O3_SHORT_FF_MAX + ERF_O3_SHORT_DACCU_MAX
        ERF_CH4_TOTAL_MAX = ERF_CH4_FF_MAX + ERF_CH4_DACCU_MAX
        ERF_O3_LONG_TOTAL_MAX = ERF_O3_LONG_FF_MAX + ERF_O3_LONG_DACCU_MAX
        ERF_SWV_TOTAL_MAX = ERF_SWV_FF_MAX + ERF_SWV_DACCU_MAX

        ERF_TOTAL_NOX_MAX = ERF_O3_SHORT_TOTAL_MAX + ERF_CH4_TOTAL_MAX + ERF_O3_LONG_TOTAL_MAX + ERF_SWV_TOTAL_MAX
        ERF_BC_TOTAL_MAX = ERF_BC_FF_MAX + ERF_BC_DACCU_MAX
        ERF_SO4_TOTAL_MAX = ERF_SO4_FF_MAX + ERF_SO4_DACCU_MAX
        ERF_H2O_TOTAL_MAX = ERF_H2O_FF_MAX + ERF_H2O_DACCU_MAX
        ERF_CC_TOTAL_MAX = CC_efficacy * (ERF_CC_FF_MAX + ERF_CC_DACCU_MAX)

        ERF_FF_MIN = ERF(ERF_CO2_FF, ERF_O3_SHORT_FF_MIN, ERF_CH4_FF_MIN, ERF_O3_LONG_FF_MIN, ERF_SWV_FF_MIN, ERF_BC_FF_MIN, ERF_SO4_FF_MIN, ERF_H2O_FF_MIN,
                     ERF_CC_FF_MIN)
        ERF_DACCU_MIN = ERF(ERF_CO2_DACCU, ERF_O3_SHORT_DACCU_MIN, ERF_CH4_DACCU_MIN, ERF_O3_LONG_DACCU_MIN, ERF_SWV_DACCU_MIN, ERF_BC_DACCU_MIN,
                        ERF_SO4_DACCU_MIN, ERF_H2O_DACCU_MIN, ERF_CC_DACCU_MIN)
        ERF_TOTAL_MIN = ERF(ERF_CO2_TOTAL, ERF_O3_SHORT_TOTAL_MIN, ERF_CH4_TOTAL_MIN, ERF_O3_LONG_TOTAL_MIN, ERF_SWV_TOTAL_MIN, ERF_BC_TOTAL_MIN,
                        ERF_SO4_TOTAL_MIN, ERF_H2O_TOTAL_MIN, ERF_CC_TOTAL_MIN)

        ERF_FF_MAX = ERF(ERF_CO2_FF, ERF_O3_SHORT_FF_MAX, ERF_CH4_FF_MAX, ERF_O3_LONG_FF_MAX, ERF_SWV_FF_MAX, ERF_BC_FF_MAX, ERF_SO4_FF_MAX, ERF_H2O_FF_MAX,
                     ERF_CC_FF_MAX)
        ERF_DACCU_MAX = ERF(ERF_CO2_DACCU, ERF_O3_SHORT_DACCU_MAX, ERF_CH4_DACCU_MAX, ERF_O3_LONG_DACCU_MAX, ERF_SWV_DACCU_MAX, ERF_BC_DACCU_MAX,
                        ERF_SO4_DACCU_MAX, ERF_H2O_DACCU_MAX, ERF_CC_DACCU_MAX)
        ERF_TOTAL_MAX = ERF(ERF_CO2_TOTAL, ERF_O3_SHORT_TOTAL_MAX, ERF_CH4_TOTAL_MAX, ERF_O3_LONG_TOTAL_MAX, ERF_SWV_TOTAL_MAX, ERF_BC_TOTAL_MAX,
                        ERF_SO4_TOTAL_MAX, ERF_H2O_TOTAL_MAX, ERF_CC_TOTAL_MAX)

    if erf_uncertain is None:
        return SCALAR_FF, SCALAR_DACCU, SCALAR_TOTAL, ERF_FF, ERF_DACCU, ERF_TOTAL
    else:
        return SCALAR_FF, SCALAR_DACCU, SCALAR_TOTAL, ERF_FF, ERF_FF_MIN, ERF_FF_MAX, ERF_DACCU, ERF_DACCU_MIN, ERF_DACCU_MAX, \
            ERF_TOTAL, ERF_TOTAL_MIN, ERF_TOTAL_MAX

def make_emissions_CO2equivalent_star(df_input, ERF_TOTAL, SCALAR_TOTAL, ERF_TOTAL_MIN = None, ERF_TOTAL_MAX = None,
                                      separate_contrails = False, output_all_emissiong_GWPstar = False, option_contrails = 'offset_all'):

    # four ways to tackle contrails:
    # OPTION 1 - 'offset_all'
    # flying_nonCO2_abated =: flying_nonCO2_abated --> ALL non-CO2 effects, including contrails (CDR counterbalances everything)
    # OPTION2 - 'offset_other_nonCO2'
    # flying_nonCO2_abated =: flying_othernonCO2_abated --> Only other non-CO2 effects, without contrails
    # OPTION3 - 'offset_abated_CC'
    # fliyng_nonCO2_abated =: flying_othernonCO2_abated + difference in CC between DACCS and DACCU ('flying_CC_abated_from_DACCU') --> Here you only do CDR in DACCS to get to the same climate effects as under DACCU
    # OPTION4 - 'rerouting' --> rerouting and avoiding 50% of CC, offsetting the rest (based on Dray et al., 2022)
    # flying_nonCO2_abated =: flying_othernonCO2_abated + flying_CC_abated
    # flying_CC_abated =: 0.5*flying_CC_emissions
    # flying_CO2_abated =: 1.0088*fuel_Tg*EI_CO2_per_fuel

    #_____GWP CALCULATIONS____________________________________
    delta_t = 20
    AGWP = 0.088
    H = 100

    progressionCurve = df_input.loc[:, 'PROGRESSION_CURVE'].values

    GWP_STAR_non_CO2_emissions_NOX = np.zeros((3,2,41))
    GWP_STAR_non_CO2_emissions_BC = np.zeros((3,2,41))
    GWP_STAR_non_CO2_emissions_SO4 = np.zeros((3,2,41))
    GWP_STAR_non_CO2_emissions_H2O = np.zeros((3,2,41))
    GWP_STAR_non_CO2_emissions_CC = np.zeros((3,2,41))

    if ERF_TOTAL_MIN is not None:
        GWP_STAR_non_CO2_emissions_NOX_MIN = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_BC_MIN = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_SO4_MIN = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_H2O_MIN = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_CC_MIN = np.zeros((3, 2, 41))

        GWP_STAR_non_CO2_emissions_NOX_MAX = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_BC_MAX = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_SO4_MAX = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_H2O_MAX = np.zeros((3, 2, 41))
        GWP_STAR_non_CO2_emissions_CC_MAX = np.zeros((3, 2, 41))


    for i in range(0, 3):
        for j in range(0, 2):
            for species in range(0,5):
                for year in range(0,41):

                    GWP_STAR_non_CO2_emissions_NOX[i,j,year] = (ERF_TOTAL.netNOX[i,j,year + 30] - ERF_TOTAL.netNOX[i,j,year - delta_t + 30])/ delta_t * (H / AGWP)
                    GWP_STAR_non_CO2_emissions_BC[i,j,year] = (ERF_TOTAL.BC[i,j,year + 30] - ERF_TOTAL.BC[i,j,year - delta_t + 30])/ delta_t * (H / AGWP)
                    GWP_STAR_non_CO2_emissions_SO4[i,j,year] = (ERF_TOTAL.SO4[i,j,year + 30] - ERF_TOTAL.SO4[i,j,year - delta_t + 30])/ delta_t * (H / AGWP)
                    GWP_STAR_non_CO2_emissions_H2O[i,j,year] = (ERF_TOTAL.H2O[i,j,year + 30] - ERF_TOTAL.H2O[i,j,year - delta_t + 30])/ delta_t * (H / AGWP)
                    GWP_STAR_non_CO2_emissions_CC[i,j,year] = (ERF_TOTAL.CC[i,j,year + 30] - ERF_TOTAL.CC[i,j,year - delta_t + 30])/ delta_t * (H / AGWP)

                    if ERF_TOTAL_MIN is not None:
                        GWP_STAR_non_CO2_emissions_NOX_MIN[i, j, year] = (ERF_TOTAL_MIN.netNOX[i, j, year + 30] - ERF_TOTAL_MIN.netNOX[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_BC_MIN[i, j, year] = (ERF_TOTAL_MIN.BC[i, j, year + 30] - ERF_TOTAL_MIN.BC[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_SO4_MIN[i, j, year] = (ERF_TOTAL_MIN.SO4[i, j, year + 30] - ERF_TOTAL_MIN.SO4[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_H2O_MIN[i, j, year] = (ERF_TOTAL_MIN.H2O[i, j, year + 30] - ERF_TOTAL_MIN.H2O[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_CC_MIN[i, j, year] = (ERF_TOTAL_MIN.CC[i, j, year + 30] - ERF_TOTAL_MIN.CC[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)

                        GWP_STAR_non_CO2_emissions_NOX_MAX[i, j, year] = (ERF_TOTAL_MAX.netNOX[i, j, year + 30] - ERF_TOTAL_MAX.netNOX[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_BC_MAX[i, j, year] = (ERF_TOTAL_MAX.BC[i, j, year + 30] - ERF_TOTAL_MAX.BC[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_SO4_MAX[i, j, year] = (ERF_TOTAL_MAX.SO4[i, j, year + 30] - ERF_TOTAL_MAX.SO4[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_H2O_MAX[i, j, year] = (ERF_TOTAL_MAX.H2O[i, j, year + 30] - ERF_TOTAL_MAX.H2O[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)
                        GWP_STAR_non_CO2_emissions_CC_MAX[i, j, year] = (ERF_TOTAL_MAX.CC[i, j, year + 30] - ERF_TOTAL_MAX.CC[
                            i, j, year - delta_t + 30]) / delta_t * (H / AGWP)

    GWP_STAR_non_CO2_emissions_total_Gt = (GWP_STAR_non_CO2_emissions_NOX + GWP_STAR_non_CO2_emissions_BC + GWP_STAR_non_CO2_emissions_SO4 + GWP_STAR_non_CO2_emissions_H2O + GWP_STAR_non_CO2_emissions_CC) / 1000
    if separate_contrails is True:
        GWP_STAR_other_nonCO2_emissions_total_Gt = (GWP_STAR_non_CO2_emissions_NOX + GWP_STAR_non_CO2_emissions_BC + GWP_STAR_non_CO2_emissions_SO4 + GWP_STAR_non_CO2_emissions_H2O) / 1000
        if ERF_TOTAL_MIN is not None:
            GWP_STAR_other_nonCO2_emissions_total_Gt_MIN = (GWP_STAR_non_CO2_emissions_NOX_MIN + GWP_STAR_non_CO2_emissions_BC_MIN +
                                                                  GWP_STAR_non_CO2_emissions_SO4_MIN + GWP_STAR_non_CO2_emissions_H2O_MIN ) / 1000
            GWP_STAR_other_nonCO2_emissions_total_Gt_MAX = (GWP_STAR_non_CO2_emissions_NOX_MAX + GWP_STAR_non_CO2_emissions_BC_MAX +
                                                                  GWP_STAR_non_CO2_emissions_SO4_MAX + GWP_STAR_non_CO2_emissions_H2O_MAX) / 1000

    if ERF_TOTAL_MIN is not None:
        GWP_STAR_non_CO2_emissions_total_Gt_MIN = (GWP_STAR_non_CO2_emissions_NOX_MIN + GWP_STAR_non_CO2_emissions_BC_MIN +
                                               GWP_STAR_non_CO2_emissions_SO4_MIN + GWP_STAR_non_CO2_emissions_H2O_MIN +
                                               GWP_STAR_non_CO2_emissions_CC_MIN) / 1000
        GWP_STAR_non_CO2_emissions_total_Gt_MAX = (GWP_STAR_non_CO2_emissions_NOX_MAX + GWP_STAR_non_CO2_emissions_BC_MAX +
                                               GWP_STAR_non_CO2_emissions_SO4_MAX + GWP_STAR_non_CO2_emissions_H2O_MAX +
                                               GWP_STAR_non_CO2_emissions_CC_MAX) / 1000

    flying_CO2_emissions = np.zeros((3,2,41))
    if separate_contrails is True:
        flying_othernonCO2_abated = np.zeros((3,2,41))
        flying_CC_abated = np.zeros((3,2,41))
    else:
        flying_nonCO2_abated = np.zeros((3,2,41))

    for i in range(0, 3):
        for j in range(0, 2):
            flying_CO2_emissions[i,j] = SCALAR_TOTAL.CO2[i,j,30:]

    flying_CO2_abated = flying_CO2_emissions * progressionCurve
    flying_CO2_abated[0,0] = 0 #DACCUs baseline neutrality
    flying_CO2_abated[1,0] = 0 # DACCUs climate neutrality
    flying_CO2_abated[2, 0] = 0 # DACCU carbon neutrality

    flying_nonCO2_emissions = GWP_STAR_non_CO2_emissions_total_Gt
    if separate_contrails is True:
        flying_othernonCO2_emissions = GWP_STAR_other_nonCO2_emissions_total_Gt
        flying_CC_emissions = GWP_STAR_non_CO2_emissions_CC / 1000
        if ERF_TOTAL_MIN is not None:
            flying_CC_emissions_min = GWP_STAR_non_CO2_emissions_CC_MIN / 1000
            flying_CC_emissions_max = GWP_STAR_non_CO2_emissions_CC_MAX / 1000
            flying_othernonCO2_emissions_min = GWP_STAR_other_nonCO2_emissions_total_Gt_MIN
            flying_othernonCO2_emissions_max = GWP_STAR_other_nonCO2_emissions_total_Gt_MAX

    if separate_contrails is False:
        # flying_nonCO2_abated[0,0] (DACCUs - baseline neutrality) = 0
        flying_nonCO2_abated[0,1] = flying_nonCO2_emissions[0,1] - flying_nonCO2_emissions[0,0] # DACCS emissions to abate following DACCU baseline
        flying_nonCO2_abated[1,0] = flying_nonCO2_emissions[1,0] * progressionCurve
        flying_nonCO2_abated[1,1] = flying_nonCO2_emissions[1,1] * progressionCurve
        # flying_nonCO2_abated[2,0] (DACCUs - carbon neutrality) = 0
        # flying_nonCO2_abated[2,1] (DACCS - carbon neutrality) = 0

        output = [flying_CO2_emissions, flying_CO2_abated, flying_nonCO2_emissions, flying_nonCO2_abated]
        if ERF_TOTAL_MIN is not None:
            flying_nonCO2_abated_MIN = np.zeros((3, 2, 41))
            flying_nonCO2_abated_MAX = np.zeros((3, 2, 41))

            flying_nonCO2_emissions_MIN = GWP_STAR_non_CO2_emissions_total_Gt_MIN
            flying_nonCO2_emissions_MAX = GWP_STAR_non_CO2_emissions_total_Gt_MAX
            flying_nonCO2_abated_MIN[0, 1] = flying_nonCO2_emissions_MIN[0, 1] - flying_nonCO2_emissions_MIN[
                0, 0]  # DACCS emissions to abate following DACCU baseline
            flying_nonCO2_abated_MIN[1, 0] = flying_nonCO2_emissions_MIN[1, 0] * progressionCurve
            flying_nonCO2_abated_MIN[1, 1] = flying_nonCO2_emissions_MIN[1, 1] * progressionCurve
            flying_nonCO2_abated_MAX[0, 1] = flying_nonCO2_emissions_MAX[0, 1] - flying_nonCO2_emissions_MAX[
                0, 0]  # DACCS emissions to abate following DACCU baseline
            flying_nonCO2_abated_MAX[1, 0] = flying_nonCO2_emissions_MAX[1, 0] * progressionCurve
            flying_nonCO2_abated_MAX[1, 1] = flying_nonCO2_emissions_MAX[1, 1] * progressionCurve
            output = [flying_CO2_emissions, flying_CO2_abated, flying_nonCO2_emissions, flying_nonCO2_emissions_MIN, \
                flying_nonCO2_emissions_MAX, flying_nonCO2_abated, flying_nonCO2_abated_MIN, flying_nonCO2_abated_MAX]

    else:
        # flying_nonCO2_abated[0,0] (DACCUs - baseline neutrality) = 0
        flying_othernonCO2_abated[0, 1] = (flying_othernonCO2_emissions[0, 1] - flying_othernonCO2_emissions[0, 0]) # DACCS emissions to abate following DACCU baseline
        flying_othernonCO2_abated[1, 0] = (flying_othernonCO2_emissions[1, 0]) * progressionCurve
        flying_othernonCO2_abated[1, 1] = (flying_othernonCO2_emissions[1, 1]) * progressionCurve
        # flying_nonCO2_abated[2,0] (DACCUs - carbon neutrality) = 0
        # flying_nonCO2_abated[2,1] (DACCS - carbon neutrality) = 0
        flying_CC_abated[0, 1] = (flying_CC_emissions[0, 1] - flying_CC_emissions[0, 0]) # DACCS emissions to abate following DACCU baseline
        flying_CC_abated[1, 0] = (flying_CC_emissions[1, 0]) * progressionCurve
        flying_CC_abated[1, 1] = (flying_CC_emissions[1, 1]) * progressionCurve

        output = [flying_CO2_emissions, flying_CO2_abated, flying_othernonCO2_emissions, flying_othernonCO2_abated, flying_CC_emissions, flying_CC_abated]
        if ERF_TOTAL_MIN is not None:
            flying_othernonCO2_abated_MIN = np.zeros((3, 2, 41))
            flying_othernonCO2_abated_MAX = np.zeros((3, 2, 41))

            flying_othernonCO2_emissions_MIN = GWP_STAR_other_nonCO2_emissions_total_Gt_MIN
            flying_othernonCO2_emissions_MAX = GWP_STAR_other_nonCO2_emissions_total_Gt_MAX
            flying_othernonCO2_abated_MIN[0, 1] = flying_othernonCO2_emissions_MIN[0, 1] - flying_othernonCO2_emissions_MIN[
                0, 0]  # DACCS emissions to abate following DACCU baseline
            flying_othernonCO2_abated_MIN[1, 0] = flying_othernonCO2_emissions_MIN[1, 0] * progressionCurve
            flying_othernonCO2_abated_MIN[1, 1] = flying_othernonCO2_emissions_MIN[1, 1] * progressionCurve
            flying_othernonCO2_abated_MAX[0, 1] = flying_othernonCO2_emissions_MAX[0, 1] - flying_othernonCO2_emissions_MAX[
                0, 0]  # DACCS emissions to abate following DACCU baseline
            flying_othernonCO2_abated_MAX[1, 0] = flying_othernonCO2_emissions_MAX[1, 0] * progressionCurve
            flying_othernonCO2_abated_MAX[1, 1] = flying_othernonCO2_emissions_MAX[1, 1] * progressionCurve

            flying_CC_abated_MIN = np.zeros((3, 2, 41))
            flying_CC_abated_MAX = np.zeros((3, 2, 41))

            flying_CC_emissions_MIN = GWP_STAR_non_CO2_emissions_CC_MIN / 1000
            flying_CC_emissions_MAX = GWP_STAR_non_CO2_emissions_CC_MAX / 1000
            flying_CC_abated_MIN[0, 1] = flying_CC_emissions_MIN[0, 1] - flying_CC_emissions_MIN[
                0, 0]  # DACCS emissions to abate following DACCU baseline
            flying_CC_abated_MIN[1, 0] = flying_CC_emissions_MIN[1, 0] * progressionCurve
            flying_CC_abated_MIN[1, 1] = flying_CC_emissions_MIN[1, 1] * progressionCurve
            flying_CC_abated_MAX[0, 1] = flying_CC_emissions_MAX[0, 1] - flying_CC_emissions_MAX[
                0, 0]  # DACCS emissions to abate following DACCU baseline
            flying_CC_abated_MAX[1, 0] = flying_CC_emissions_MAX[1, 0] * progressionCurve
            flying_CC_abated_MAX[1, 1] = flying_CC_emissions_MAX[1, 1] * progressionCurve

            output = [flying_CO2_emissions, flying_CO2_abated, flying_othernonCO2_emissions, flying_othernonCO2_emissions_MIN, \
                flying_othernonCO2_emissions_MAX, flying_othernonCO2_abated, flying_othernonCO2_abated_MIN, flying_othernonCO2_abated_MAX, \
                    flying_CC_emissions, flying_CC_emissions_min, flying_CC_emissions_max, flying_CC_abated, flying_CC_abated_MIN, flying_CC_abated_MAX]

    if output_all_emissiong_GWPstar is True:
        output.extend([GWP_STAR_non_CO2_emissions_NOX, GWP_STAR_non_CO2_emissions_BC, GWP_STAR_non_CO2_emissions_SO4,
                      GWP_STAR_non_CO2_emissions_H2O, GWP_STAR_non_CO2_emissions_CC])
    return tuple(output)


# make ERF and emissions of base scenario


#================================================================================================================================================


def make_WTT_emissions_II(df_input, FF_EJ_vector):
    
    np.zeros((1,41))
    
    WTT_UPSTREAM                          = 10.7               # Well to Tank upstream emissions [gCO2/MJ]
    WTT_DISTRIBUTION                      = 0.5                # Well to Tank distribussion emissions [gCO2/MJ]

    WTTupstreamArray = np.full((1,41),WTT_UPSTREAM)
    WTTdistributionArray = np.full((1,41),WTT_DISTRIBUTION)

    WTTrefiningArray =  df_input.loc[:, 'REFINING_WTT'].values 

    WTT_FactorVector = WTTupstreamArray + WTTdistributionArray + WTTrefiningArray 



    yearlyWTT = np.zeros((3,2,41))

    for i in range(0, 3): 
        for j in range(0, 2):

            yearlyWTT[i,j] = WTT_FactorVector * FF_EJ_vector[i,j] / 1000

    return yearlyWTT 

#================================================================================================================================================

def make_DACCU_need_DAC_H2_CO_FT_electricity_II(df_input, DACCU_Tg_vector, JETFUEL_ALLOCATION_SHARE,
                                                configuration = 'PEM+El.CO2+FT', efficiency_increase = False):
    
    DAC_electricity_need = df_input.loc[:, 'ELECTRICITY_DAC_KWH_KGCO2'].values #MWhth/tCO2
    DAC_heat_need = 1.75  # MWhth/tCO2 - 2035 estimate of Gabrielli et al., 2020

    TOT_FT_OUTPUT_PER_JETFUEL = JETFUEL_ALLOCATION_SHARE**-1
    DIESEL_IN_TOT_FT_OUTPUT = (1-JETFUEL_ALLOCATION_SHARE)*TOT_FT_OUTPUT_PER_JETFUEL
    #DAC electricity_need is an array defined in Inputs (changes over time)

    DAC_DACCU_Gt = np.zeros((3, 2, 41))
    DAC_DACCU_MWh = np.zeros((3, 2, 41))
    DAC_DACCU_MWh_heat = np.zeros((3, 2, 41))
    H2_DACCU_Mt = np.zeros((3, 2, 41))
    H2_DACCU_MWh = np.zeros((3, 2, 41))
    CO_DACCU_Mt = np.zeros((3, 2, 41))
    CO_DACCU_MWh = np.zeros((3, 2, 41))
    CO_DACCU_MWh_heat = np.zeros((3, 2, 41))
    FT_DACCU_MWh = np.zeros((3, 2, 41))

    DAC_diesel_Gt = np.zeros((3, 2, 41))
    DAC_diesel_MWh = np.zeros((3, 2, 41))
    DAC_diesel_MWh_heat = np.zeros((3, 2, 41))
    H2_diesel_Mt = np.zeros((3, 2, 41))
    H2_diesel_MWh = np.zeros((3, 2, 41))
    CO_diesel_Mt = np.zeros((3, 2, 41))
    CO_diesel_MWh = np.zeros((3, 2, 41))
    CO_diesel_MWh_heat = np.zeros((3, 2, 41))
    FT_diesel_MWh = np.zeros((3, 2, 41))

    if efficiency_increase is False:
        if configuration == "PEM+El.CO2+FT":
            H2_ELECTRICITY_NEED_FACTOR = 53  # H2 electricity need [kWh/KgH2produced]    % 2035 best estimate as mean of: IRENA, 2020;  Schmidt et al., 2017; Matute et al., 2019; Nosherwani and Neto, 2021  (Amir's previous assumption: 47 for 2050 from Becattini et al., 2021)
            CO_THERMAL_NEED_FACTOR = 0
            CO_ELECTRICITY_NEED_FACTOR = 6.34  # CO electricity need [kWh/KgCOproduced]  % 2020 best estimate as mean of: Becattini et al., 2021; Shin et al., 2021; Jouny et al., 2018 (Amir's previous assumption: 5.6 from Becattini et al., 2021)
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26 # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            KGCO2_KG_FUEL = 5.63 / TOT_FT_OUTPUT_PER_JETFUEL  # How much CO2 from DAC is needed to produce 1 ton of synthetic fuel Becattini et al., 2021
            KGH2_KG_FUEL = 0.54 / TOT_FT_OUTPUT_PER_JETFUEL  # How much H2 from electrolysis is needed to produce 1 ton of synthetic fuel Becattini et al., 2021
            KGCO_KG_FUEL = 3.58 / TOT_FT_OUTPUT_PER_JETFUEL  # How much CO from electrolysis is needed to produce 1 ton of synthetic fuel Becattini et al., 2021

        elif configuration == "AEC+El.CO2+FT":
            H2_ELECTRICITY_NEED_FACTOR = 51  # H2 electricity need [kWh/KgH2produced]    %%
            CO_THERMAL_NEED_FACTOR = 0
            CO_ELECTRICITY_NEED_FACTOR = 6.34  # CO electricity need [kWh/KgCOproduced]  %%
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26 # FT electricity need [kWh/Kgfuelproduced]

            KGCO2_KG_FUEL = 5.73 / TOT_FT_OUTPUT_PER_JETFUEL  # How much CO2 from DAC is needed to produce 1 ton of synthetic fuel
            KGH2_KG_FUEL = 0.54 / TOT_FT_OUTPUT_PER_JETFUEL  # How much H2 from electrolysis is needed to produce 1 ton of synthetic fuel
            KGCO_KG_FUEL = 3.58 / TOT_FT_OUTPUT_PER_JETFUEL  # How much CO from electrolysis is needed to produce 1 ton of synthetic fuel

        elif configuration == "PEM+RWGS+FT":
            H2_ELECTRICITY_NEED_FACTOR = 53  # H2 electricity need [kWh/KgH2produced]    % 2035 best estimate as mean of: IRENA, 2020;  Schmidt et al., 2017; Matute et al., 2019; Nosherwani and Neto, 2021  (Amir's previous assumption: 47 for 2050 from Becattini et al., 2021)
            CO_THERMAL_HEAT_NEED_FACTOR = 0.49  # CO thermal need [kWh/kgCOproduced] % today Doty et al., 2010
            CO_ELECTRICITY_NEED_FACTOR = 0.11  # CO electricity need [kWh/KgCOproduced]  % today Doty et al., 2010
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            H2_KG_CO = 0.071  # kg H2 involved in the CO production - Mean of: van der Giesen et al., Doty et al., 2010
            H2_KG_FT = 0.30   # kg H2 involved in FT synthesis - Mean of: Becattini et al, 2021; Kalavasta et al., 2018
            CO2_KG_CO = 1.57 # kg CO2 involved in the CO production - Mean of: Jouny et al., 2018, Becattini et al., 2021, Shin et al., 2021
            KGCO_KG_FUEL = 1.969 # kg CO involved in RWGS - Mean of : Becattini et al., 2021; Kalavasta et al., 2018

            KGCO2_KG_FUEL = CO2_KG_CO * KGCO_KG_FUEL
            KGH2_KG_FUEL = H2_KG_FT + H2_KG_CO*KGCO_KG_FUEL

        elif configuration == "AEC+RWGS+FT":
            H2_ELECTRICITY_NEED_FACTOR = 51  # H2 electricity need [kWh/KgH2produced]    %%
            CO_THERMAL_HEAT_NEED_FACTOR = 0.49  # CO thermal need [kWh/kgCOproduced] % today Doty et al., 2010
            CO_ELECTRICITY_NEED_FACTOR = 0.11  # CO electricity need [kWh/KgCOproduced]  % 2020 best estimate as mean of: Becattini et al., 2021; Shin et al., 2021; Jouny et al., 2018 (Amir's previous assumption: 5.6 from Becattini et al., 2021)
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            H2_KG_CO = 0.071  # kg H2 involved in the CO production - Mean of: van der Giesen et al., Doty et al., 2010
            H2_KG_FT = 0.30   # kg H2 involved in FT synthesis - Mean of: Becattini et al, 2021; Kalavasta et al., 2018
            CO2_KG_CO = 1.57 # kg CO2 involved in the CO production - Mean of: Jouny et al., 2018, Becattini et al., 2021, Shin et al., 2021
            KGCO_KG_FUEL = 1.969 # kg CO involved in RWGS - Mean of : Becattini et al., 2021; Kalavasta et al., 2018

            KGCO2_KG_FUEL = CO2_KG_CO * KGCO_KG_FUEL
            KGH2_KG_FUEL = H2_KG_FT + H2_KG_CO*KGCO_KG_FUEL

        elif configuration == "average":
            H2_ELECTRICITY_NEED_FACTOR = np.mean([51,53])  # H2 electricity need [kWh/KgH2produced]    %%
            CO_THERMAL_HEAT_NEED_FACTOR = np.mean([0.49, 0])  # CO thermal need [kWh/kgCOproduced] % today Doty et al., 2010
            CO_ELECTRICITY_NEED_FACTOR = np.mean([0.11, 6.34])  # CO electricity need [kWh/KgCOproduced]  % 2020 best estimate as mean of: Becattini et al., 2021; Shin et al., 2021; Jouny et al., 2018 (Amir's previous assumption: 5.6 from Becattini et al., 2021)
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            KGCO_KG_FUEL = np.mean([1.969, 1.967])
            KGCO2_KG_FUEL = np.mean([3.09, 3.15])
            KGH2_KG_FUEL = np.mean([0.44, 0.297])


    elif efficiency_increase is True:
        def interpolate(year, years, electricity_factors):
            return np.interp(year, years, electricity_factors)

        if configuration == "PEM+El.CO2+FT":
            H2_ELECTRICITY_NEED_FACTOR_2020                   = 57       # H2 electricity need [kWh/KgH2produced]    % Mean of: Sutter et al., 2019; IRENA, 2020; Schmidt et al., 2015; Yates et al., 2020; Kopp et al., 2017; Alfian et al., 2019
            H2_ELECTRICITY_NEED_FACTOR_2035                   = 53       # H2 electricity need [kWh/KgH2produced]    % Mean of Schmidt et al., 2017; Nosherwani and Neto, 2021
            H2_ELECTRICITY_NEED_FACTOR_2050                   = 47.5     # H2 electricity need [kWh/KgH2produced]    % Becattini et al., 2021

            years = np.array([2020, 2035, 2050])
            electricity_factors = np.array([H2_ELECTRICITY_NEED_FACTOR_2020, H2_ELECTRICITY_NEED_FACTOR_2035, H2_ELECTRICITY_NEED_FACTOR_2050])
            all_years = np.arange(years[0], 2061)
            H2_ELECTRICITY_NEED_FACTOR = interpolate(all_years, years, electricity_factors)

            CO_THERMAL_HEAT_NEED_FACTOR = 0
            CO_ELECTRICITY_NEED_FACTOR                  = 6.34        # CO electricity need [kWh/KgCOproduced]  % 2020 best estimate as mean of: Becattini et al., 2021; Shin et al., 2021; Jouny et al., 2018 (Amir's previous assumption: 5.6 from Becattini et al., 2021)
            FT_ELECTRICITY_NEED_KWH_KG_FUEL       = 0.26             # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            KGCO2_KG_FUEL                   = 5.63/TOT_FT_OUTPUT_PER_JETFUEL              # How much CO2 from DAC is needed to produce 1 ton of synthetic fuel Becattini et al., 2021
            KGH2_KG_FUEL                    = 0.54/TOT_FT_OUTPUT_PER_JETFUEL              # How much H2 from electrolysis is needed to produce 1 ton of synthetic fuel Becattini et al., 2021
            KGCO_KG_FUEL                    = 3.58/TOT_FT_OUTPUT_PER_JETFUEL              # How much CO from electrolysis is needed to produce 1 ton of synthetic fuel Becattini et al., 2021

        elif configuration == "AEC+El.CO2+FT":
            H2_ELECTRICITY_NEED_FACTOR_2020 = 53  # H2 electricity need [kWh/KgH2produced]    % Mean of: Rosenthal et al., 2020; Becattini et al., 2021; IRENA, 2020; Schmidt et al., 2017
            H2_ELECTRICITY_NEED_FACTOR_2035 = 51  # H2 electricity need [kWh/KgH2produced]    % Mean of Rosenthal et al., 2020;  IRENA, 2020; Schmidt et al., 2017; Nosherwani and Neto, 2021
            H2_ELECTRICITY_NEED_FACTOR_2050 = 45.5  # H2 electricity need [kWh/KgH2produced]    % Most optimistic estimate of 2035: Rosenthal et al., 2020

            years = np.array([2020, 2035, 2050])
            electricity_factors = np.array(
                [H2_ELECTRICITY_NEED_FACTOR_2020, H2_ELECTRICITY_NEED_FACTOR_2035, H2_ELECTRICITY_NEED_FACTOR_2050])
            all_years = np.arange(years[0], 2061)
            H2_ELECTRICITY_NEED_FACTOR = interpolate(all_years, years, electricity_factors)

            CO_THERMAL_HEAT_NEED_FACTOR = 0
            CO_ELECTRICITY_NEED_FACTOR = 6.34  # CO electricity need [kWh/KgCOproduced]  %%
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced]

            KGCO2_KG_FUEL = 5.63 / TOT_FT_OUTPUT_PER_JETFUEL  # How much CO2 from DAC is needed to produce 1 ton of synthetic fuel
            KGH2_KG_FUEL = 0.54 / TOT_FT_OUTPUT_PER_JETFUEL  # How much H2 from electrolysis is needed to produce 1 ton of synthetic fuel
            KGCO_KG_FUEL = 3.58 / TOT_FT_OUTPUT_PER_JETFUEL  # How much CO from electrolysis is needed to produce 1 ton of synthetic fuel

        elif configuration == "PEM+RWGS+FT":
            H2_ELECTRICITY_NEED_FACTOR_2020                   = 57       # H2 electricity need [kWh/KgH2produced]    % Mean of: Sutter et al., 2019; IRENA, 2020; Schmidt et al., 2015; Yates et al., 2020; Kopp et al., 2017; Alfian et al., 2019
            H2_ELECTRICITY_NEED_FACTOR_2035                   = 53       # H2 electricity need [kWh/KgH2produced]    % Mean of Schmidt et al., 2017; Nosherwani and Neto, 2021
            H2_ELECTRICITY_NEED_FACTOR_2050                   = 47.5     # H2 electricity need [kWh/KgH2produced]    % Becattini et al., 2021

            years = np.array([2020, 2035, 2050])
            electricity_factors = np.array([H2_ELECTRICITY_NEED_FACTOR_2020, H2_ELECTRICITY_NEED_FACTOR_2035, H2_ELECTRICITY_NEED_FACTOR_2050])
            all_years = np.arange(years[0], 2061)
            H2_ELECTRICITY_NEED_FACTOR= interpolate(all_years, years, electricity_factors)

            CO_THERMAL_HEAT_NEED_FACTOR = 0.49  # CO thermal need [kWh/kgCOproduced] % today Doty et al., 2010
            CO_ELECTRICITY_NEED_FACTOR = 0.11  # CO electricity need [kWh/KgCOproduced]  % 2020 best estimate as mean of: Becattini et al., 2021; Shin et al., 2021; Jouny et al., 2018 (Amir's previous assumption: 5.6 from Becattini et al., 2021)
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            H2_KG_CO = 0.071  # kg H2 involved in the CO production - Mean of: van der Giesen et al., Doty et al., 2010
            H2_KG_FT = 0.30  # kg H2 involved in FT synthesis - Mean of: Becattini et al, 2021; Kalavasta et al., 2018
            CO2_KG_CO = 1.57  # kg CO2 involved in the CO production - Mean of: Jouny et al., 2018, Becattini et al., 2021, Shin et al., 2021
            KGCO_KG_FUEL = 1.969  # kg CO involved in RWGS - Mean of : Becattini et al., 2021; Kalavasta et al., 2018

            KGCO2_KG_FUEL = CO2_KG_CO * KGCO_KG_FUEL
            KGH2_KG_FUEL = H2_KG_FT + H2_KG_CO * KGCO_KG_FUEL

        elif configuration == "AEC+RWGS+FT":
            H2_ELECTRICITY_NEED_FACTOR_2020 = 53  # H2 electricity need [kWh/KgH2produced]    % Mean of: Rosenthal et al., 2020; Becattini et al., 2021; IRENA, 2020; Schmidt et al., 2017
            H2_ELECTRICITY_NEED_FACTOR_2035 = 51  # H2 electricity need [kWh/KgH2produced]    % Mean of Rosenthal et al., 2020;  IRENA, 2020; Schmidt et al., 2017; Nosherwani and Neto, 2021
            H2_ELECTRICITY_NEED_FACTOR_2050 = 45.5  # H2 electricity need [kWh/KgH2produced]    % Most optimistic estimate of 2035: Rosenthal et al., 2020

            years = np.array([2020, 2035, 2050])
            electricity_factors = np.array(
                [H2_ELECTRICITY_NEED_FACTOR_2020, H2_ELECTRICITY_NEED_FACTOR_2035, H2_ELECTRICITY_NEED_FACTOR_2050])
            all_years = np.arange(years[0], 2061)
            H2_ELECTRICITY_NEED_FACTOR = interpolate(all_years, years, electricity_factors)

            CO_THERMAL_HEAT_NEED_FACTOR = 0.49  # CO thermal need [kWh/kgCOproduced] % today Doty et al., 2010
            CO_ELECTRICITY_NEED_FACTOR = 0.11  # CO electricity need [kWh/KgCOproduced]  % 2020 best estimate as mean of: Becattini et al., 2021; Shin et al., 2021; Jouny et al., 2018 (Amir's previous assumption: 5.6 from Becattini et al., 2021)
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            H2_KG_CO = 0.071  # kg H2 involved in the CO production - Mean of: van der Giesen et al., Doty et al., 2010
            H2_KG_FT = 0.30  # kg H2 involved in FT synthesis - Mean of: Becattini et al, 2021; Kalavasta et al., 2018
            CO2_KG_CO = 1.57  # kg CO2 involved in the CO production - Mean of: Jouny et al., 2018, Becattini et al., 2021, Shin et al., 2021
            KGCO_KG_FUEL = 1.969  # kg CO involved in RWGS - Mean of : Becattini et al., 2021; Kalavasta et al., 2018

            KGCO2_KG_FUEL = CO2_KG_CO * KGCO_KG_FUEL
            KGH2_KG_FUEL = H2_KG_FT + H2_KG_CO * KGCO_KG_FUEL

        elif configuration == "average":
            H2_ELECTRICITY_NEED_FACTOR_2020 = np.mean([53, 57])  # H2 electricity need [kWh/KgH2produced]    % Mean of: Rosenthal et al., 2020; Becattini et al., 2021; IRENA, 2020; Schmidt et al., 2017
            H2_ELECTRICITY_NEED_FACTOR_2035 = np.mean([51, 53])  # H2 electricity need [kWh/KgH2produced]    % Mean of Rosenthal et al., 2020;  IRENA, 2020; Schmidt et al., 2017; Nosherwani and Neto, 2021
            H2_ELECTRICITY_NEED_FACTOR_2050 = np.mean([45.5, 47.5])  # H2 electricity need [kWh/KgH2produced]    % Most optimistic estimate of 2035: Rosenthal et al., 2020

            years = np.array([2020, 2035, 2050])
            electricity_factors = np.array(
                [H2_ELECTRICITY_NEED_FACTOR_2020, H2_ELECTRICITY_NEED_FACTOR_2035, H2_ELECTRICITY_NEED_FACTOR_2050])
            all_years = np.arange(years[0], 2061)
            H2_ELECTRICITY_NEED_FACTOR = interpolate(all_years, years, electricity_factors)

            CO_THERMAL_HEAT_NEED_FACTOR = np.mean([0.49, 0])  # CO thermal need [kWh/kgCOproduced] % today Doty et al., 2010
            CO_ELECTRICITY_NEED_FACTOR = np.mean([0.11, 6.34])  # CO electricity need [kWh/KgCOproduced]  % 2020 best estimate as mean of: Becattini et al., 2021; Shin et al., 2021; Jouny et al., 2018 (Amir's previous assumption: 5.6 from Becattini et al., 2021)
            FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)

            KGCO_KG_FUEL = np.mean([1.969, 1.967])
            KGCO2_KG_FUEL = np.mean([3.09, 3.15])
            KGH2_KG_FUEL = np.mean([0.44, 0.297])

    #Electricity need per KGFuel produced (calculated from electricity need per products multiplied per products needed per unit fuel
    DAC_Electricity_tonFuel_MWh = DAC_electricity_need * KGCO2_KG_FUEL * (10**6)
    H2_Electricity_tonFuel_MWh  = H2_ELECTRICITY_NEED_FACTOR * KGH2_KG_FUEL * (10**6)
    CO_Electricity_tonFuel_MWh  = CO_ELECTRICITY_NEED_FACTOR * KGCO_KG_FUEL * (10**6)
    FT_Electricity_tonFuel_MWh = FT_ELECTRICITY_NEED_KWH_KG_FUEL * (10**6)
    CO_Heat_tonFuel_MWh = CO_THERMAL_HEAT_NEED_FACTOR*(10**6)
    DAC_heat_tonFuel_MWh = DAC_heat_need * KGCO2_KG_FUEL * (10**6)

    KGCO_KG_DIESEL = KGCO_KG_FUEL * DIESEL_IN_TOT_FT_OUTPUT
    KGH2_KG_DIESEL = KGH2_KG_FUEL * DIESEL_IN_TOT_FT_OUTPUT
    KGCO2_KG_DIESEL = KGCO2_KG_FUEL * DIESEL_IN_TOT_FT_OUTPUT

    # Electricity need per KGFuel produced (calculated from electricity need per products multiplied per products needed per unit fuel
    DAC_Electricity_ton_diesel_MWh = DAC_electricity_need * KGCO2_KG_DIESEL * (10 ** 6)
    H2_Electricity_ton_diesel_MWh = H2_ELECTRICITY_NEED_FACTOR * KGH2_KG_DIESEL * (10 ** 6)
    CO_Electricity_ton_diesel_MWh = CO_ELECTRICITY_NEED_FACTOR * KGCO_KG_DIESEL * (10 ** 6)
    FT_Electricity_ton_diesel_MWh = FT_ELECTRICITY_NEED_KWH_KG_FUEL * DIESEL_IN_TOT_FT_OUTPUT * (10 ** 6)
    CO_Heat_ton_diesel_MWh = CO_THERMAL_HEAT_NEED_FACTOR * KGCO_KG_DIESEL * (10 ** 6)
    DAC_Heat_ton_diesel_MWh = DAC_heat_need * KGCO2_KG_DIESEL * (10 ** 6)
    for i in range(0, 3):
        for j in range(0, 2):

            # DAC for DACCU: Capacity & Electricity requirements

            DAC_DACCU_Gt[i,j] = DACCU_Tg_vector[i,j] * KGCO2_KG_FUEL/1000
            DAC_DACCU_MWh[i,j] = DACCU_Tg_vector[i,j] * DAC_Electricity_tonFuel_MWh
            DAC_DACCU_MWh_heat[i,j] = DACCU_Tg_vector[i,j] * DAC_heat_tonFuel_MWh
            # H2 for DACCU: Capacity & Electricity requirements

            H2_DACCU_Mt[i,j] = DACCU_Tg_vector[i,j] * KGH2_KG_FUEL
            H2_DACCU_MWh[i,j] = DACCU_Tg_vector[i,j] * H2_Electricity_tonFuel_MWh

            # CO for DACCU: Capacity & Electricity requirements

            CO_DACCU_Mt[i,j] = DACCU_Tg_vector[i,j] * KGCO_KG_FUEL
            CO_DACCU_MWh[i,j] = DACCU_Tg_vector[i,j] * CO_Electricity_tonFuel_MWh
            CO_DACCU_MWh_heat[i,j] = DACCU_Tg_vector[i,j] * CO_Heat_tonFuel_MWh

            Tot_DACCU_MWh_heat = DAC_DACCU_MWh_heat + CO_DACCU_MWh_heat

            # Fischer-Tropsch for DACCU: Electricity requirements (non-electricity costs for FT process are neglected - as they are negligible)

            FT_DACCU_MWh[i,j] = DACCU_Tg_vector[i,j] * FT_Electricity_tonFuel_MWh

            # DAC for diesel: Capacity & Electricity requirements

            DAC_diesel_Gt[i, j] = DACCU_Tg_vector[i, j] * KGCO2_KG_DIESEL/ 1000
            DAC_diesel_MWh[i, j] = DACCU_Tg_vector[i, j] * DAC_Electricity_ton_diesel_MWh
            DAC_diesel_MWh_heat[i,j] = DACCU_Tg_vector[i,j] * DAC_Heat_ton_diesel_MWh

            # H2 for diesel: Capacity & Electricity requirements

            H2_diesel_Mt[i, j] = DACCU_Tg_vector[i, j] * KGH2_KG_DIESEL
            H2_diesel_MWh[i, j] = DACCU_Tg_vector[i, j] * H2_Electricity_ton_diesel_MWh

            # CO for diesel: Capacity & Electricity requirements

            CO_diesel_Mt[i, j] = DACCU_Tg_vector[i, j] * KGCO_KG_DIESEL
            CO_diesel_MWh[i, j] = DACCU_Tg_vector[i, j] * CO_Electricity_ton_diesel_MWh
            CO_diesel_MWh_heat[i, j] = DACCU_Tg_vector[i, j] * CO_Heat_ton_diesel_MWh

            Tot_diesel_MWh_heat = DAC_diesel_MWh_heat + CO_diesel_MWh_heat
            # Fischer-Tropsch for diesel: Electricity requirements (non-electricity costs for FT process are neglected - as they are negligible)

            FT_diesel_MWh[i, j] = DACCU_Tg_vector[i, j] * FT_Electricity_ton_diesel_MWh

    output_variables = [DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh,
                        DAC_diesel_Gt, DAC_diesel_MWh, H2_diesel_Mt, H2_diesel_MWh, CO_diesel_Mt, CO_diesel_MWh,
                        FT_diesel_MWh, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat, CO_diesel_MWh_heat, DAC_diesel_MWh_heat]

    # Filter out None values if the corresponding variable is not requested
    output_variables = [var for var in output_variables if var is not None]

    return tuple(output_variables)

    #return DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh, DAC_diesel_Gt, DAC_diesel_MWh, H2_diesel_Mt, H2_diesel_MWh, CO_diesel_Mt, CO_diesel_MWh, FT_diesel_MWh

#================================================================================================================================================

### DACCU Production: Material and electricity indirect emissions 

def make_DACCU_indirect_emissions_II(df_input, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh):

    gridFootprint_WORLD = df_input.loc[:, 'GRID_CARBON_INTENSITY_WORLD'].values  
    gridFootprint = gridFootprint_WORLD #Should I parametrize this? it seems that the wind power option doesn't matter for the results discussion.
    
    DAC_MATERIAL_EMISSION_FACTOR                 = 0.0435             # DAC material emissions [KgCO2e/KgCO2captured]
    H2_MATERIAL_EMISSION_FACTOR                  = 0.01546            # H2 material emissions [KgCO2e/KgH2produced]
    CO_MATERIAL_EMISSION_FACTOR                  = 0.025080123        # CO material emissions [KgCO2e/KgCOproduced]
        
    
    DAC_DACCU_MaterialFootprint  = np.zeros((3,2,41))
    DAC_DACCU_ElectricityFootprint = np.zeros((3,2,41))
    H2_DACCU_MaterialFootprint = np.zeros((3,2,41))
    H2_DACCU_ElectricityFootprint = np.zeros((3,2,41))
    CO_DACCU_MaterialFootprint = np.zeros((3,2,41))
    CO_DACCU_ElectricityFootprint = np.zeros((3,2,41))
    FT_DACCU_ElectricityFootprint = np.zeros((3,2,41))

    totalDACCUMaterialFootprint = np.zeros((3,2,41))
    totalDACCUElectricitryFootprint = np.zeros((3,2,41))
    totalDACCUFootprint = np.zeros((3,2,41))


    for i in range(0, 3): 
        for j in range(0, 2):


            #DACCU Production: Material and electricity indirect emissions 

            DAC_DACCU_MaterialFootprint[i,j]    =  DAC_DACCU_Gt[i,j] * DAC_MATERIAL_EMISSION_FACTOR  #Gt
            DAC_DACCU_ElectricityFootprint[i,j] =  DAC_DACCU_MWh[i,j] * gridFootprint / (10**9)      #Gt

            H2_DACCU_MaterialFootprint[i,j]     =  H2_DACCU_Mt[i,j] * H2_MATERIAL_EMISSION_FACTOR /1000    #Mt
            H2_DACCU_ElectricityFootprint[i,j]  =  H2_DACCU_MWh[i,j] * gridFootprint / (10**9)             #Mt

            CO_DACCU_MaterialFootprint[i,j]     =  CO_DACCU_Mt[i,j] * CO_MATERIAL_EMISSION_FACTOR /1000    #Mt
            CO_DACCU_ElectricityFootprint[i,j]  =  CO_DACCU_MWh[i,j] * gridFootprint / (10**9)             #Mt

            FT_DACCU_ElectricityFootprint[i,j]  =  FT_DACCU_MWh[i,j] * gridFootprint / (10**9)     

            totalDACCUMaterialFootprint[i,j] = DAC_DACCU_MaterialFootprint[i,j] + H2_DACCU_MaterialFootprint[i,j] + CO_DACCU_MaterialFootprint[i,j] 
            totalDACCUElectricitryFootprint[i,j] = DAC_DACCU_ElectricityFootprint[i,j] + H2_DACCU_ElectricityFootprint[i,j]  + CO_DACCU_ElectricityFootprint[i,j] + FT_DACCU_ElectricityFootprint[i,j] 
            totalDACCUFootprint[i,j] = totalDACCUMaterialFootprint[i,j] + totalDACCUElectricitryFootprint[i,j]
    ###
    ### I could add some subtotals here (total material, electric DACCU footprint)
    
    return DAC_DACCU_MaterialFootprint, DAC_DACCU_ElectricityFootprint, H2_DACCU_MaterialFootprint, \
        H2_DACCU_ElectricityFootprint, CO_DACCU_MaterialFootprint, CO_DACCU_ElectricityFootprint,\
        FT_DACCU_ElectricityFootprint, totalDACCUMaterialFootprint, totalDACCUElectricitryFootprint, totalDACCUFootprint

    
#==========================================================================================================

### DAC CDR CO2 & non-CO2 emissions: Capacity & electricity requirements
### + Material & Electricity footprint

def make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated, flying_nonCO2_abated,
                                                           flying_nonCO2_abated_MIN = None, flying_nonCO2_abated_MAX = None, grid = 'world' ):

    DAC_electricity_need = df_input.loc[:, 'ELECTRICITY_DAC_KWH_KGCO2'].values
    DAC_heat_need = 1.75 #MWhth/tCO2 - Gabrielli et al., 2020
    gridFootprint_WORLD = df_input.loc[:, 'GRID_CARBON_INTENSITY_WORLD'].values
    gridFootprint_WIND  = df_input.loc[:, 'GRID_CARBON_INTENSITY_WIND'].values
    if grid == 'world':
        gridFootprint = gridFootprint_WORLD #Should I parametrize this? it seems that the wind power option doesn't matter for the results discussion.
    else:
        gridFootprint = gridFootprint_WIND
    
    DAC_MATERIAL_EMISSION_FACTOR                = 0.0435             # DAC material emissions [KgCO2e/KgCO2captured]

    DAC_CDR_CO2_Gt = np.zeros((3,2,41))
    DAC_CDR_CO2_MWh = np.zeros((3,2,41))
    DAC_CDR_CO2_MWhth = np.zeros((3, 2, 41))
    DAC_CDR_nonCO2_Gt = np.zeros((3,2,41))
    DAC_CDR_nonCO2_MWh = np.zeros((3,2,41))
    DAC_CDR_nonCO2_MWhth = np.zeros((3, 2, 41))
    DAC_CDR_CO2_MaterialFootprint = np.zeros((3,2,41))
    DAC_CDR_CO2_ElectricityFootprint = np.zeros((3,2,41))
    DAC_CDR_nonCO2_MaterialFootprint = np.zeros((3,2,41))
    DAC_CDR_nonCO2_ElectricityFootprint = np.zeros((3,2,41))
    totalDAC_CDRFootprint = np.zeros((3,2,41))

    if flying_nonCO2_abated_MIN is not None:
        DAC_CDR_nonCO2_Gt_MIN = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_MWh_MIN = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_MWhth_MIN = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_Gt_MAX = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_MWh_MAX = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_MWhth_MAX = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_MaterialFootprint_MIN = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_ElectricityFootprint_MIN = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_MaterialFootprint_MAX = np.zeros((3, 2, 41))
        DAC_CDR_nonCO2_ElectricityFootprint_MAX = np.zeros((3, 2, 41))
        totalDAC_CDRFootprint_MIN = np.zeros((3, 2, 41))
        totalDAC_CDRFootprint_MAX = np.zeros((3, 2, 41))

    for i in range(0, 3): 
        for j in range(0, 2):
            DAC_CDR_CO2_Gt[i,j] = flying_CO2_abated[i,j]
            DAC_CDR_CO2_MWh[i,j] = DAC_CDR_CO2_Gt[i,j] * DAC_electricity_need * (10**12) / (10**3)
            DAC_CDR_CO2_MWhth[i, j] = DAC_CDR_CO2_Gt[i, j] * DAC_heat_need  * (10**12) / (10**3)

            DAC_CDR_nonCO2_Gt[i,j] = flying_nonCO2_abated[i,j]
            DAC_CDR_nonCO2_MWh[i,j] = DAC_CDR_nonCO2_Gt[i,j] * DAC_electricity_need * (10**12) / (10**3)
            DAC_CDR_nonCO2_MWhth[i, j] = DAC_CDR_nonCO2_Gt[i, j] * DAC_heat_need  * (10**12) / (10**3)
            ## DAC CDR CO2 & non-CO2 emissions: Material & Electricity footprint

            DAC_CDR_CO2_MaterialFootprint[i,j] = DAC_CDR_CO2_Gt[i,j] * DAC_MATERIAL_EMISSION_FACTOR
            DAC_CDR_CO2_ElectricityFootprint[i,j] = DAC_CDR_CO2_MWh[i,j] * gridFootprint / (10**9)

            DAC_CDR_nonCO2_MaterialFootprint[i,j] = DAC_CDR_nonCO2_Gt[i,j] * DAC_MATERIAL_EMISSION_FACTOR
            DAC_CDR_nonCO2_ElectricityFootprint[i,j] = DAC_CDR_nonCO2_MWh[i,j] * gridFootprint / (10**9)


            totalDAC_CDRFootprint[i,j] = DAC_CDR_CO2_MaterialFootprint[i,j] + DAC_CDR_CO2_ElectricityFootprint[i,j] + DAC_CDR_nonCO2_MaterialFootprint[i,j] + DAC_CDR_nonCO2_ElectricityFootprint[i,j]

            if flying_nonCO2_abated_MIN is not None:
                DAC_CDR_nonCO2_Gt_MIN[i, j] = flying_nonCO2_abated_MIN[i, j]
                DAC_CDR_nonCO2_Gt_MAX[i,j]  = flying_nonCO2_abated_MAX[i,j]
                DAC_CDR_nonCO2_MWh_MIN[i, j] = DAC_CDR_nonCO2_Gt_MIN[i, j] * DAC_electricity_need * (10 ** 12) / (10 ** 3)
                DAC_CDR_nonCO2_MWh_MAX[i, j] = DAC_CDR_nonCO2_Gt_MAX[i, j] * DAC_electricity_need * (10 ** 12) / (10 ** 3)
                DAC_CDR_nonCO2_MWhth_MIN[i, j] = DAC_CDR_nonCO2_Gt_MIN[i, j] * DAC_heat_need * (10 ** 12) / (10 ** 3)
                DAC_CDR_nonCO2_MWhth_MAX[i, j] = DAC_CDR_nonCO2_Gt_MAX[i, j] * DAC_heat_need * (10 ** 12) / (10 ** 3)

                DAC_CDR_nonCO2_MaterialFootprint_MIN[i,j] = DAC_CDR_nonCO2_Gt_MIN[i,j] * DAC_MATERIAL_EMISSION_FACTOR
                DAC_CDR_nonCO2_ElectricityFootprint_MIN[i,j] = DAC_CDR_nonCO2_MWh_MIN[i,j] * gridFootprint / (10**9)
                DAC_CDR_nonCO2_MaterialFootprint_MAX[i,j] = DAC_CDR_nonCO2_Gt_MAX[i,j] * DAC_MATERIAL_EMISSION_FACTOR
                DAC_CDR_nonCO2_ElectricityFootprint_MAX[i,j] = DAC_CDR_nonCO2_MWh_MAX[i,j] * gridFootprint / (10**9)
                totalDAC_CDRFootprint_MIN[i,j] = DAC_CDR_CO2_MaterialFootprint[i,j] + \
                                                 DAC_CDR_CO2_ElectricityFootprint[i,j] + DAC_CDR_nonCO2_MaterialFootprint_MIN[i,j] + \
                                                 DAC_CDR_nonCO2_ElectricityFootprint_MIN[i,j]
                totalDAC_CDRFootprint_MAX[i,j] = DAC_CDR_CO2_MaterialFootprint[i,j] + DAC_CDR_CO2_ElectricityFootprint[i,j] + \
                                                 DAC_CDR_nonCO2_MaterialFootprint_MAX[i,j] + DAC_CDR_nonCO2_ElectricityFootprint_MAX[i,j]

    if flying_nonCO2_abated_MIN is not None:
        return DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_Gt_MIN, DAC_CDR_nonCO2_Gt_MAX, \
                DAC_CDR_nonCO2_MWh, DAC_CDR_nonCO2_MWh_MIN, DAC_CDR_nonCO2_MWh_MAX, \
                DAC_CDR_nonCO2_MWhth, DAC_CDR_nonCO2_MWhth_MIN, DAC_CDR_nonCO2_MWhth_MAX, \
                DAC_CDR_CO2_MaterialFootprint, DAC_CDR_CO2_ElectricityFootprint, \
                DAC_CDR_nonCO2_MaterialFootprint, DAC_CDR_nonCO2_MaterialFootprint_MIN, DAC_CDR_nonCO2_MaterialFootprint_MAX, \
                DAC_CDR_nonCO2_ElectricityFootprint, DAC_CDR_nonCO2_ElectricityFootprint_MIN, DAC_CDR_nonCO2_ElectricityFootprint_MAX, \
                totalDAC_CDRFootprint, totalDAC_CDRFootprint_MIN, totalDAC_CDRFootprint_MAX
    else:
        return DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_MWh, \
            DAC_CDR_nonCO2_MWhth, DAC_CDR_CO2_MaterialFootprint, \
                DAC_CDR_CO2_ElectricityFootprint, DAC_CDR_nonCO2_MaterialFootprint, DAC_CDR_nonCO2_ElectricityFootprint, \
                totalDAC_CDRFootprint


#========================================================================================================================

# total indirect emissions, Delta indirect emissions and net emissions calculations

def make_indirect_delta_net_emissions_II(yearlyWTT, totalDACCUFootprint, totalDAC_CDRFootprint, flying_CO2_emissions,
                                         flying_nonCO2_emissions,flying_CO2_abated,flying_nonCO2_abated,
                                         total_DAC_CDRFootprint_MIN = None, total_DAC_CDRFootprint_MAX = None,
                                         flying_nonCO2_emissions_MIN = None, flying_nonCO2_emissions_MAX = None,
                                         flying_nonCO2_abated_MIN = None, flying_nonCO2_abated_MAX = None):


    totalIndirectEmissions = np.zeros((3,2,41))

    if total_DAC_CDRFootprint_MIN is not None:
        totalIndirectEmissions_MIN = np.zeros((3, 2, 41))
        totalIndirectEmissions_MAX = np.zeros((3, 2, 41))

    #Total indirect emissions (WTT + all electricity & material footprints)
    for i in range(0, 3): 
        for j in range(0, 2):
            totalIndirectEmissions[i,j] = yearlyWTT[i,j] + totalDACCUFootprint[i,j] + totalDAC_CDRFootprint[i,j]
            if total_DAC_CDRFootprint_MIN is not None:
                totalIndirectEmissions_MIN[i, j] = yearlyWTT[i, j] + totalDACCUFootprint[i, j] + total_DAC_CDRFootprint_MIN[i, j]
                totalIndirectEmissions_MAX[i, j] = yearlyWTT[i, j] + totalDACCUFootprint[i, j] + total_DAC_CDRFootprint_MAX[i, j]
                ### Here compare A & B (DACCU vs DAC-CCS)

    #deltaDifference computation
    Delta_totalIndirectEmissions = np.zeros((3,2,41))

    if total_DAC_CDRFootprint_MIN is not None:
        Delta_totalIndirectEmissions_MIN = np.zeros((3, 2, 41))
        Delta_totalIndirectEmissions_MAX = np.zeros((3, 2, 41))

    for i in range(0, 3): 
        for j in range(0, 2):
            for y in range(0,41):

                if j == 0:

                    if (totalIndirectEmissions[i,0,y] > totalIndirectEmissions[i,1,y]): 

                        Delta_totalIndirectEmissions[i,0,y] = totalIndirectEmissions[i,0,y] - totalIndirectEmissions[i,1,y] 

                        if total_DAC_CDRFootprint_MIN is not None:
                            Delta_totalIndirectEmissions_MIN[i, 0, y] = totalIndirectEmissions_MIN[i, 0, y] - \
                                                                    totalIndirectEmissions_MIN[i, 1, y]
                            Delta_totalIndirectEmissions_MAX[i, 0, y] = totalIndirectEmissions_MAX[i, 0, y] - \
                                                                        totalIndirectEmissions_MAX[i, 1, y]


                if j == 1:
                    if (totalIndirectEmissions[i,1,y] > totalIndirectEmissions[i,0,y]):

                        Delta_totalIndirectEmissions[i,1,y] = totalIndirectEmissions[i,1,y] - totalIndirectEmissions[i,0,y]

                        if total_DAC_CDRFootprint_MIN is not None:
                            Delta_totalIndirectEmissions_MIN[i, 1, y] = totalIndirectEmissions_MIN[i, 1, y] - \
                                                                    totalIndirectEmissions_MIN[i, 0, y]
                            Delta_totalIndirectEmissions_MAX[i, 1, y] = totalIndirectEmissions_MAX[i, 1, y] - \
                                                                    totalIndirectEmissions_MAX[i, 0, y]

                            # Calculating total emissions pathways and BAU emissions

    # This is the BAU case, identical for all pathays 
    #(based on flying emissions of using only fossil jetfuel + WTT emissions)
    BAU_EmissionsGt = (flying_CO2_emissions[0,1] + flying_nonCO2_emissions[0,1])

    if total_DAC_CDRFootprint_MIN is not None:
        BAU_EmissionsGt_MIN = (flying_CO2_emissions[0, 1] + flying_nonCO2_emissions_MIN[0, 1])
        BAU_EmissionsGt_MAX = (flying_CO2_emissions[0, 1] + flying_nonCO2_emissions_MAX[0, 1])

    totalNetEmissions = np.zeros((3,2,41))
    if total_DAC_CDRFootprint_MIN is not None:
        totalNetEmissions_MIN = np.zeros((3, 2, 41))
        totalNetEmissions_MAX = np.zeros((3, 2, 41))

    for i in range(0, 3): 
        for j in range(0, 2):
            totalNetEmissions[i,j]  = ((flying_CO2_emissions[i,j] + flying_nonCO2_emissions[i,j]) - (flying_CO2_abated[i,j] + flying_nonCO2_abated[i,j])) + totalIndirectEmissions[i,j] - Delta_totalIndirectEmissions[i,j] 

            if total_DAC_CDRFootprint_MIN is not None:
                totalNetEmissions_MIN[i, j] = ((flying_CO2_emissions[i, j] + flying_nonCO2_emissions_MIN[i, j]) - (
                            flying_CO2_abated[i, j] + flying_nonCO2_abated_MIN[i, j])) + totalIndirectEmissions_MIN[i, j] - \
                                          Delta_totalIndirectEmissions_MIN[i, j]
                totalNetEmissions_MAX[i, j] = ((flying_CO2_emissions[i, j] + flying_nonCO2_emissions_MAX[i, j]) - (
                            flying_CO2_abated[i, j] + flying_nonCO2_abated_MAX[i, j])) + totalIndirectEmissions_MAX[i, j] - \
                                          Delta_totalIndirectEmissions_MAX[i, j]

    if total_DAC_CDRFootprint_MIN is not None:
        return totalIndirectEmissions, totalIndirectEmissions_MIN, totalIndirectEmissions_MAX,\
            Delta_totalIndirectEmissions, Delta_totalIndirectEmissions_MIN, Delta_totalIndirectEmissions_MAX, \
            BAU_EmissionsGt, BAU_EmissionsGt_MIN, BAU_EmissionsGt_MAX, \
            totalNetEmissions, totalNetEmissions_MIN, totalNetEmissions_MAX
    else:
        return totalIndirectEmissions, Delta_totalIndirectEmissions, BAU_EmissionsGt, totalNetEmissions


#============================================================

#### DAC Capacity & learning curves

def make_learning_curve_DAC_II(LEARNING_RATE, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt, DAC_CDR_CO2_Gt,
                               DAC_CDR_nonCO2_Gt, Delta_totalIndirectEmissions, DAC_CDR_nonCO2_Gt_MIN = None,
                               DAC_CDR_nonCO2_Gt_MAX = None, Delta_totalIndirectEmissions_MIN = None,
                               Delta_totalIndirectEmissions_MAX = None):

    b = - math.log(1 - LEARNING_RATE) / math.log(2) # calculating b factor for learning effects calculations
    #totalYearlyDAC is DAC for DACCU production + DAC for CDR of CO2 and non-CO2 emissions + DAC DeltaIndirectEmissions
    DAC_CDR_nonCO2_Gt[DAC_CDR_nonCO2_Gt < 0] = 0
    totalYearlyDAC_need = np.zeros((3,2,41))
    total_DAC_InstalledCapacity = np.zeros((3,2,41))
    yearlyAddedCapacityDAC = np.zeros((3,2,41))
    cost_DAC_ct = np.zeros((3,2,41))

    if DAC_CDR_nonCO2_Gt_MIN is not None:
        DAC_CDR_nonCO2_Gt_MIN[DAC_CDR_nonCO2_Gt_MIN < 0] = 0
        DAC_CDR_nonCO2_Gt_MAX[DAC_CDR_nonCO2_Gt_MAX < 0] = 0
        totalYearlyDAC_need_MIN = np.zeros((3,2,41))
        total_DAC_InstalledCapacity_MIN = np.zeros((3,2,41))
        yearlyAddedCapacityDAC_MIN = np.zeros((3,2,41))
        cost_DAC_ct_MIN = np.zeros((3,2,41))
        totalYearlyDAC_need_MAX = np.zeros((3,2,41))
        total_DAC_InstalledCapacity_MAX = np.zeros((3,2,41))
        yearlyAddedCapacityDAC_MAX = np.zeros((3,2,41))
        cost_DAC_ct_MAX = np.zeros((3,2,41))

    #Initiating inital DAC capacity and cost for year 2020
    for i in range(0, 3): 
        for j in range(0, 2):
            total_DAC_InstalledCapacity[i,j,0] = DAC_q0_Gt_2020
            cost_DAC_ct[i,j,0] = DAC_c0_2020
            totalYearlyDAC_need[i,j] = DAC_DACCU_Gt[i,j] + DAC_CDR_CO2_Gt[i,j] + DAC_CDR_nonCO2_Gt[i,j] + Delta_totalIndirectEmissions[i,j]

            if DAC_CDR_nonCO2_Gt_MIN is not None:
                total_DAC_InstalledCapacity_MIN[i, j, 0] = DAC_q0_Gt_2020
                cost_DAC_ct_MIN[i, j, 0] = DAC_c0_2020
                totalYearlyDAC_need_MIN[i, j] = DAC_DACCU_Gt[i, j] + DAC_CDR_CO2_Gt[i, j] + DAC_CDR_nonCO2_Gt_MIN[i, j] + \
                                            Delta_totalIndirectEmissions_MIN[i, j]

                total_DAC_InstalledCapacity_MAX[i, j, 0] = DAC_q0_Gt_2020
                cost_DAC_ct_MAX[i, j, 0] = DAC_c0_2020
                totalYearlyDAC_need_MAX[i, j] = DAC_DACCU_Gt[i, j] + DAC_CDR_CO2_Gt[i, j] + DAC_CDR_nonCO2_Gt_MAX[i, j] + \
                                            Delta_totalIndirectEmissions_MAX[i, j]

            for y in range(1,41):
                if totalYearlyDAC_need[i,j,y] + DAC_q0_Gt_2020 > total_DAC_InstalledCapacity[i,j,y-1]:
                    total_DAC_InstalledCapacity[i,j,y] = totalYearlyDAC_need[i,j,y] + DAC_q0_Gt_2020
                    if DAC_CDR_nonCO2_Gt_MIN is not None:
                        total_DAC_InstalledCapacity_MIN[i, j, y] = totalYearlyDAC_need_MIN[i, j, y] + DAC_q0_Gt_2020
                        total_DAC_InstalledCapacity_MAX[i, j, y] = totalYearlyDAC_need_MAX[i, j, y] + DAC_q0_Gt_2020
                else:
                    total_DAC_InstalledCapacity[i,j,y] = total_DAC_InstalledCapacity[i,j,y-1]
                    if DAC_CDR_nonCO2_Gt_MIN is not None:
                        total_DAC_InstalledCapacity_MIN[i, j, y] = total_DAC_InstalledCapacity_MIN[i, j, y - 1]
                        total_DAC_InstalledCapacity_MAX[i, j, y] = total_DAC_InstalledCapacity_MAX[i, j, y - 1]

                yearlyAddedCapacityDAC[i,j,y] = total_DAC_InstalledCapacity[i,j,y] - total_DAC_InstalledCapacity[i,j,y-1]
                cost_DAC_ct[i,j,y] = DAC_c0_2020 * ((total_DAC_InstalledCapacity[i,j,y]/DAC_q0_Gt_2020)**(-b))
                #cost_DAC_ct[i,j,y] = cost_DAC_ct[i,j,y-1] * ((total_DAC_InstalledCapacity[i,j,y]/total_DAC_InstalledCapacity[i,j,y-1])**(-b))
                if DAC_CDR_nonCO2_Gt_MIN is not None:
                    yearlyAddedCapacityDAC_MIN[i, j, y] = total_DAC_InstalledCapacity_MIN[i, j, y] - \
                                                      total_DAC_InstalledCapacity_MIN[i, j, y - 1]
                    cost_DAC_ct_MIN[i, j, y] = DAC_c0_2020 * (
                                (total_DAC_InstalledCapacity_MIN[i, j, y] / DAC_q0_Gt_2020) ** (-b))
                    yearlyAddedCapacityDAC_MAX[i, j, y] = total_DAC_InstalledCapacity_MAX[i, j, y] - \
                                                          total_DAC_InstalledCapacity_MAX[i, j, y - 1]
                    cost_DAC_ct_MAX[i, j, y] = DAC_c0_2020 * (
                            (total_DAC_InstalledCapacity_MAX[i, j, y] / DAC_q0_Gt_2020) ** (-b))

    output = [totalYearlyDAC_need, total_DAC_InstalledCapacity, yearlyAddedCapacityDAC, cost_DAC_ct]

    if DAC_CDR_nonCO2_Gt_MIN is not None:
        output= [totalYearlyDAC_need, totalYearlyDAC_need_MIN, totalYearlyDAC_need_MAX, \
            total_DAC_InstalledCapacity, total_DAC_InstalledCapacity_MIN, total_DAC_InstalledCapacity_MAX,\
            yearlyAddedCapacityDAC, yearlyAddedCapacityDAC_MIN, yearlyAddedCapacityDAC_MAX, \
            cost_DAC_ct, cost_DAC_ct_MIN, cost_DAC_ct_MAX]
    return tuple(output)

#========================================================================================================

#### H2 and CO Capacity & learning curves

def make_learning_curve_H2_CO_II(LEARNING_RATE_H2, LEARNING_RATE_CO, H2_q0_Mt_2020, H2_c0_2020, H2_DACCU_Mt, CO_q0_Mt_2020,
                                 CO_c0_2020, CO_DACCU_Mt, operating_capacity = 1, configuration = 'average'):
    
    b_H2 = math.log(1 - LEARNING_RATE_H2,2) * (-1) # calculating b factor for learning effects calculations
    b_CO = math.log(1 - LEARNING_RATE_CO, 2) * (-1)  # calculating b factor for learning effects calculations

    totalYearlyH2_need = np.zeros((3,2,41))
    total_H2_InstalledCapacity = np.zeros((3,2,41))
    yearlyAddedCapacityH2 = np.zeros((3,2,41))
    yearlyH2_need_increase = np.zeros((3,2,41))
    cost_H2_ct = np.zeros((3,2,41))

    #Initiating inital H2 capacity and cost for year 2020
    for i in range(0, 3): 
        for j in range(0, 2):

            total_H2_InstalledCapacity[i,j,0] = H2_q0_Mt_2020
            cost_H2_ct[i,j,0] = H2_c0_2020

    #Calculating learning curve: how much H2 capacity is added and it's cost in €/tH2 produced
    for i in range(0, 3): 
        for j in range(0, 2):

            totalYearlyH2_need[i,j] = H2_DACCU_Mt[i,j]

            for y in range(1,41):
                if totalYearlyH2_need[i,j,y] + H2_q0_Mt_2020 > total_H2_InstalledCapacity[i,j,y-1]:
                    total_H2_InstalledCapacity[i,j,y] = totalYearlyH2_need[i,j,y] + H2_q0_Mt_2020 
                else: 
                    total_H2_InstalledCapacity[i,j,y] = total_H2_InstalledCapacity[i,j,y-1]
                yearlyAddedCapacityH2[i,j,y] = total_H2_InstalledCapacity[i,j,y] - total_H2_InstalledCapacity[i,j,y-1]
                cost_H2_ct[i,j,y] = H2_c0_2020 * ((total_H2_InstalledCapacity[i,j,y]/H2_q0_Mt_2020)**(-b_H2))
                yearlyH2_need_increase[i,j,y] = totalYearlyH2_need[i,j,y] - totalYearlyH2_need[i,j,y-1]
    ####
    #### CO Capacity & learning curves

    totalYearlyCO_need = np.zeros((3,2,41))
    total_CO_InstalledCapacity = np.zeros((3,2,41))
    yearlyAddedCapacityCO = np.zeros((3,2,41))
    yearlyCO_need_increase = np.zeros((3,2,41))

    cost_CO_ct = np.zeros((3,2,41))

    #Initiating inital CO capacity and cost for year 2020
    for i in range(0, 3): 
        for j in range(0, 2):

            total_CO_InstalledCapacity[i,j,0] = CO_q0_Mt_2020
            cost_CO_ct[i,j,0] = CO_c0_2020

    #Calculating learning curve: how much CO capacity is added and its cost in €/tCO produced
    for i in range(0, 3): 
        for j in range(0, 2):
            totalYearlyCO_need[i,j] = CO_DACCU_Mt[i,j]
            for y in range(1,41):
                if totalYearlyCO_need[i,j,y] + CO_q0_Mt_2020 > total_CO_InstalledCapacity[i,j,y-1]:
                    total_CO_InstalledCapacity[i,j,y] = totalYearlyCO_need[i,j,y] + CO_q0_Mt_2020
                else: 
                    total_CO_InstalledCapacity[i,j,y] = total_CO_InstalledCapacity[i,j,y-1]

                yearlyAddedCapacityCO[i,j,y] = total_CO_InstalledCapacity[i,j,y] - total_CO_InstalledCapacity[i,j,y-1]
                cost_CO_ct[i,j,y] = CO_c0_2020 * ((total_CO_InstalledCapacity[i,j,y]/CO_q0_Mt_2020)**(-b_CO))
                yearlyCO_need_increase[i,j,y] = totalYearlyCO_need[i,j,y] - totalYearlyCO_need[i,j,y-1]
                
    return totalYearlyH2_need, total_H2_InstalledCapacity, yearlyAddedCapacityH2, yearlyH2_need_increase, cost_H2_ct, totalYearlyCO_need, total_CO_InstalledCapacity, yearlyAddedCapacityCO, yearlyCO_need_increase, cost_CO_ct    

#==================================================================================================================

##### DAC COSTS 
### Calculating cost and capacity matrix: DAC

### Each row correspond to the the capacity added at year y at it's given fixed cost during it's lifetime
### Each column sums up to all DAC costs used to fulfill this year's DAC need

def calculate_DAC_costs_II(cost_DAC_ct, yearlyAddedCapacityDAC, totalYearlyDAC_need, DAC_DACCU_Gt, DAC_CDR_CO2_Gt,
                           DAC_CDR_nonCO2_Gt, Delta_totalIndirectEmissions, cost_DAC_ct_MIN = None, cost_DAC_ct_MAX = None,
                           yearlyAddedCapacityDAC_MIN = None, yearlyAddedCapacityDAC_MAX = None, totalYearlyDAC_need_MIN = None,
                           totalYearlyDAC_need_MAX = None, DAC_CDR_nonCO2_Gt_MIN = None, DAC_CDR_nonCO2_Gt_MAX = None,
                           Delta_totalIndirectEmissions_MIN = None, Delta_totalIndirectEmissions_MAX = None):
    
    DAC_LIFETIME = 20
    current_year = 0
    total_years = 0
    total_years_2 = 0

    cost_matrix_DAC = np.zeros((3,2,41,41))
    if DAC_CDR_nonCO2_Gt_MIN is not None:
        cost_matrix_DAC_MIN = np.zeros((3, 2, 41, 41))
        cost_matrix_DAC_MAX = np.zeros((3, 2, 41, 41))
    for i in range(0, 3): 
        for j in range(0, 2):
            current_year = 0
            for y in range(0,41):
                if (41 - current_year) > DAC_LIFETIME: 
                    total_years = DAC_LIFETIME
                else: 
                    total_years = (41 - current_year)
                for t in range(0, total_years): #41-current year
                    cost_matrix_DAC[i,j,y,current_year + t] = cost_DAC_ct[i,j,current_year] * yearlyAddedCapacityDAC[i,j,current_year] * (10**9)
                    if DAC_CDR_nonCO2_Gt_MIN is not None:
                        cost_matrix_DAC_MIN[i, j, y, current_year + t] = cost_DAC_ct_MIN[i, j, current_year] * \
                                                                     yearlyAddedCapacityDAC_MIN[i, j, current_year] * (
                                                                                 10 ** 9)
                        cost_matrix_DAC_MAX[i, j, y, current_year + t] = cost_DAC_ct_MAX[i, j, current_year] * \
                                                                         yearlyAddedCapacityDAC_MAX[
                                                                             i, j, current_year] * (
                                                                                 10 ** 9)

                if (current_year + DAC_LIFETIME) < 41:  #We check if there are empty cells after all the capacity is built for a given year    
                        total_years_2 = (41 - (current_year + DAC_LIFETIME) ) 
                        for t2 in range(0, total_years_2): 

                            cost_matrix_DAC[i,j,y,current_year + DAC_LIFETIME + t2] = cost_DAC_ct[i,j,current_year + DAC_LIFETIME] * yearlyAddedCapacityDAC[i,j,current_year] * (10**9)

                            if DAC_CDR_nonCO2_Gt_MIN is not None:
                                cost_matrix_DAC_MIN[i, j, y, current_year + DAC_LIFETIME + t2] = cost_DAC_ct_MIN[
                                                                                                 i, j, current_year + DAC_LIFETIME] * \
                                                                                             yearlyAddedCapacityDAC_MIN[
                                                                                                 i, j, current_year] * (
                                                                                                         10 ** 9)
                                cost_matrix_DAC_MAX[i,j,y,current_year + DAC_LIFETIME + t2] = cost_DAC_ct_MAX[
                                                                                              i,j,current_year + DAC_LIFETIME] * \
                                                                                          yearlyAddedCapacityDAC_MAX[i,j,current_year] * (10**9)


                #For hydrogen and CO there is a 3rd pass (as 2020 + 15 + 15 i 2050, there are 10 years left)

                current_year = current_year + 1
    ###Summing all columns of the DAC cost matrix

    total_yearly_DAC_COST = np.zeros((3,2,41))
    yearly_DAC_DACCU_Cost = np.zeros((3,2,41))
    yearly_DAC_CDR_CO2_Cost = np.zeros((3,2,41))
    yearly_DAC_CDR_nonCO2_Cost = np.zeros((3,2,41))
    yearly_DAC_DeltaEmissions_Cost = np.zeros((3,2,41))
    if DAC_CDR_nonCO2_Gt_MIN is not None:
        total_yearly_DAC_COST_MIN = np.zeros((3, 2, 41))
        total_yearly_DAC_COST_MAX = np.zeros((3,2,41))
        yearly_DAC_CDR_nonCO2_Cost_MIN = np.zeros((3, 2, 41))
        yearly_DAC_CDR_nonCO2_Cost_MAX = np.zeros((3, 2, 41))
        yearly_DAC_DeltaEmissions_Cost_MIN = np.zeros((3, 2, 41))
        yearly_DAC_DeltaEmissions_Cost_MAX = np.zeros((3, 2, 41))

    for i in range(0, 3): 
        for j in range(0, 2):
            for t in range(0, 41):
                total_yearly_DAC_COST[i,j,t] = np.sum(cost_matrix_DAC[i,j,:,t])
                if DAC_CDR_nonCO2_Gt_MIN is not None:
                    total_yearly_DAC_COST_MIN[i, j, t] = np.sum(cost_matrix_DAC_MIN[i, j, :, t])
                    total_yearly_DAC_COST_MAX[i, j, t] = np.sum(cost_matrix_DAC_MAX[i, j, :, t])

                ### Separating DAC cost per category
                if (DAC_DACCU_Gt[i,j,t] == 0):
                    yearly_DAC_DACCU_Cost[i,j,t] = 0
                else:
                    yearly_DAC_DACCU_Cost[i,j,t] = total_yearly_DAC_COST[i,j,t] * (DAC_DACCU_Gt[i,j,t]/totalYearlyDAC_need[i,j,t])

                if (DAC_CDR_CO2_Gt[i,j,t] == 0):
                    yearly_DAC_CDR_CO2_Cost[i,j,t] = 0
                else:
                    yearly_DAC_CDR_CO2_Cost[i,j,t] = total_yearly_DAC_COST[i,j,t] * (DAC_CDR_CO2_Gt[i,j,t]/totalYearlyDAC_need[i,j,t])

                if (DAC_CDR_nonCO2_Gt[i,j,t] == 0):
                    yearly_DAC_CDR_nonCO2_Cost[i,j,t] = 0
                    if DAC_CDR_nonCO2_Gt_MIN is not None:
                        yearly_DAC_CDR_nonCO2_Cost_MIN[i, j, t] = 0
                        yearly_DAC_CDR_nonCO2_Cost_MAX[i,j,t] = 0
                else:
                    yearly_DAC_CDR_nonCO2_Cost[i,j,t] = total_yearly_DAC_COST[i,j,t] * (DAC_CDR_nonCO2_Gt[i,j,t]/totalYearlyDAC_need[i,j,t])
                    if DAC_CDR_nonCO2_Gt_MIN is not None:
                        yearly_DAC_CDR_nonCO2_Cost_MIN[i, j, t] = total_yearly_DAC_COST_MIN[i, j, t] * (
                                    DAC_CDR_nonCO2_Gt_MIN[i, j, t] / totalYearlyDAC_need_MIN[i, j, t])
                        yearly_DAC_CDR_nonCO2_Cost_MAX[i,j,t] = total_yearly_DAC_COST_MAX[i,j,t] * \
                                                                (DAC_CDR_nonCO2_Gt_MAX[i,j,t]/totalYearlyDAC_need_MAX[i,j,t])


                if (Delta_totalIndirectEmissions[i,j,t] == 0):
                    yearly_DAC_DeltaEmissions_Cost[i,j,t] = 0
                    if DAC_CDR_nonCO2_Gt_MIN is not None:
                        yearly_DAC_DeltaEmissions_Cost_MIN[i, j, t] = 0
                        yearly_DAC_DeltaEmissions_Cost_MAX[i, j, t] = 0
                else:
                    yearly_DAC_DeltaEmissions_Cost[i,j,t] = total_yearly_DAC_COST[i,j,t] * (Delta_totalIndirectEmissions[i,j,t]/totalYearlyDAC_need[i,j,t])
                    if DAC_CDR_nonCO2_Gt_MIN is not None:
                        yearly_DAC_DeltaEmissions_Cost_MIN[i, j, t] = total_yearly_DAC_COST_MIN[i, j, t] * (
                                    Delta_totalIndirectEmissions_MIN[i, j, t] / totalYearlyDAC_need_MIN[i, j, t])
                        yearly_DAC_DeltaEmissions_Cost_MAX[i, j, t] = total_yearly_DAC_COST_MAX[i, j, t] * (
                                    Delta_totalIndirectEmissions_MAX[i, j, t] / totalYearlyDAC_need_MAX[i, j, t])

    if DAC_CDR_nonCO2_Gt_MIN is not None:
        return total_yearly_DAC_COST, total_yearly_DAC_COST_MIN, total_yearly_DAC_COST_MAX, \
            yearly_DAC_DACCU_Cost,  yearly_DAC_CDR_CO2_Cost, \
            yearly_DAC_CDR_nonCO2_Cost, yearly_DAC_CDR_nonCO2_Cost_MIN, yearly_DAC_CDR_nonCO2_Cost_MAX, \
            yearly_DAC_DeltaEmissions_Cost, yearly_DAC_DeltaEmissions_Cost_MIN, yearly_DAC_DeltaEmissions_Cost_MIN
    else:
        return total_yearly_DAC_COST, yearly_DAC_DACCU_Cost,  yearly_DAC_CDR_CO2_Cost, yearly_DAC_CDR_nonCO2_Cost, yearly_DAC_DeltaEmissions_Cost

#================================================================================

##### H2 and CO Costs

### Calculating cost and capacity matrix: CO and H2

def calculate_H2_CO_costs_II(cost_H2_ct, yearlyH2_need_increase, cost_CO_ct, yearlyCO_need_increase):

    H2_LIFETIME = 20
    CO_LIFETIME = 20

    cost_matrix_H2 = np.zeros((3,2,41,41))
    current_year = 0
    total_years = 0
    total_years_2 = 0
    total_years_3 = 0

    for i in range(0, 3): 
        for j in range(0, 2):
            current_year = 0
            for y in range(0,41):

                if (41 - current_year) > H2_LIFETIME: 
                    total_years = H2_LIFETIME 


                else: 
                    total_years = (41 - current_year)

                for t in range(0, total_years): #41-current year

                    cost_matrix_H2[i,j,y,current_year + t] = cost_H2_ct[i,j,current_year] * yearlyH2_need_increase[i,j,current_year] * (10**6)

                if (current_year + H2_LIFETIME) < 41:  #We check if there are empty cells after all the capacity is built for a given year    
                        total_years_2 = (41 - (current_year + H2_LIFETIME) ) 
                        for t2 in range(0, total_years_2): 

                            cost_matrix_H2[i,j,y,current_year + H2_LIFETIME + t2] = cost_H2_ct[i,j,current_year + H2_LIFETIME] * yearlyH2_need_increase[i,j,current_year] * (10**6)

                #For hydrogen and CO there is a 3rd pass (as 2020 + 15 + 15 i 2050, there are 10 years left)

                if (current_year + H2_LIFETIME + H2_LIFETIME ) < 41:  #We check if there are empty cells after all the capacity is built for a given year    
                        total_years_3 = (41 - (current_year + H2_LIFETIME + H2_LIFETIME) ) 
                        for t3 in range(0, total_years_3): 

                            cost_matrix_H2[i,j,y,current_year + H2_LIFETIME + H2_LIFETIME + t3] = cost_H2_ct[i,j,current_year + H2_LIFETIME + H2_LIFETIME] * yearlyH2_need_increase[i,j,current_year] * (10**6)

                current_year = current_year + 1


    ### Calculating cost and capacity matrix: CO

    cost_matrix_CO = np.zeros((3,2,41,41))
    current_year = 0
    total_years = 0
    total_years_2 = 0
    total_years_3 = 0

    for i in range(0, 3): 
        for j in range(0, 2):
            current_year = 0
            for y in range(0,41):

                if (41 - current_year) > CO_LIFETIME: 
                    total_years = CO_LIFETIME 


                else: 
                    total_years = (41 - current_year)

                for t in range(0, total_years): #41-current year

                    cost_matrix_CO[i,j,y,current_year + t] = cost_CO_ct[i,j,current_year] * yearlyCO_need_increase[i,j,current_year] * (10**6)

                if (current_year + CO_LIFETIME) < 41:  #We check if there are empty cells after all the capacity is built for a given year    
                        total_years_2 = (41 - (current_year + CO_LIFETIME) ) 
                        for t2 in range(0, total_years_2): 

                            cost_matrix_CO[i,j,y,current_year + CO_LIFETIME + t2] = cost_CO_ct[i,j,current_year + CO_LIFETIME] * yearlyCO_need_increase[i,j,current_year] * (10**6)

                #For hydrogen and CO there is a 3rd pass (as 2020 + 15 + 15 i 2050, there are 10 years left)

                if (current_year + CO_LIFETIME + CO_LIFETIME ) < 41:  #We check if there are empty cells after all the capacity is built for a given year    
                        total_years_3 = (41 - (current_year + CO_LIFETIME + CO_LIFETIME) ) 
                        for t3 in range(0, total_years_3): 

                            cost_matrix_CO[i,j,y,current_year + CO_LIFETIME + CO_LIFETIME + t3] = cost_CO_ct[i,j,current_year + CO_LIFETIME + CO_LIFETIME] * yearlyCO_need_increase[i,j,current_year] * (10**6)

                current_year = current_year + 1
                
    ###Summing all columns of the H2 and CO cost matrix

    total_yearly_H2_COST = np.zeros((3,2,41))
    total_yearly_CO_COST = np.zeros((3,2,41))

    for i in range(0, 3): 
        for j in range(0, 2):
            for t in range(0, 41):
                total_yearly_H2_COST[i,j,t] = np.sum(cost_matrix_H2[i,j,:,t])
                total_yearly_CO_COST[i,j,t] = np.sum(cost_matrix_CO[i,j,:,t])


    return total_yearly_H2_COST, total_yearly_CO_COST

#==========================================================================================


###_______________________FINAL COST CALCULATIONS_______________________________________________


def calculate_final_cost_II(ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                            DAC_DACCU_MWh,H2_DACCU_MWh,CO_DACCU_MWh,FT_DACCU_MWh,DAC_CDR_CO2_MWh,
                            DAC_CDR_nonCO2_MWh, Delta_totalIndirectEmissions, DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt,
                            FF_EJ_vector,yearly_DAC_DACCU_Cost, total_yearly_H2_COST,
                            total_yearly_CO_COST, total_yearly_DAC_COST,
                            df_input, CO_DACCU_MWh_heat = None, DAC_DACCU_MWh_heat= None, DAC_CDR_CO2_MWhth = None, DAC_CDR_nonCO2_MWhth = None,
                            emissions_cost = None,
                            DAC_CDR_nonCO2_MWh_MIN = None, DAC_CDR_nonCO2_MWh_MAX = None,
                            Delta_totalIndirectEmissions_MIN = None, Delta_totalIndirectEmissions_MAX = None,
                            DAC_CDR_nonCO2_Gt_MIN = None, DAC_CDR_nonCO2_Gt_MAX = None,
                            total_yearly_DAC_COST_MIN = None, total_yearly_DAC_COST_MAX = None,
                            DAC_CDR_nonCO2_MWhth_MIN = None, DAC_CDR_nonCO2_MWhth_MAX = None):

    DAC_electricity_need = df_input.loc[:, 'ELECTRICITY_DAC_KWH_KGCO2'].values

    FT_cost = 108/10**3 # €/kgfuel  # CAPEX of FT from Becattini et al. 2018
    FT_ELECTRICITY_NEED_KWH_KG_FUEL = 0.26  # FT electricity need [kWh/Kgfuelproduced (Brazzola et al., 2024) 2035 estimate by Terwel and Kerkhoven, 2022 (Amir's previous assumption: 0.45 from Falter et al., 2021)
    FT_DACCU_cost =  FT_DACCU_MWh*10**3*FT_cost/FT_ELECTRICITY_NEED_KWH_KG_FUEL # Cost of FT in €

    ENERGY_DENSITY_FUEL = 0.0288184438040346 ## [L/MJ]
    COST_HIGH_TEMPERATURE_HEAT = 40 # €/MWhth - Moretti et al., 2024
    COST_LOW_TEMPERATURE_HEAT = 10 # €/MWhth - Moretti et al., 2024

    finalcost_electricity = np.zeros((3,2,41))
    finalcost_heat = np.zeros((3,2,41))
    finalcost_transport_storageCO2  = np.zeros((3,2,41))
    finalcost_fossilfuel = np.zeros((3,2,41))
    finalcost = np.zeros((3,2,41))

    total_DACCU_electricity_cost = np.zeros((3,2,41))
    total_DACCS_electricity_cost = np.zeros((3, 2, 41))
    total_DACCU_production_cost = np.zeros((3,2,41))
    total_DACCS_cost = np.zeros((3, 2, 41))
    total_DACCU_heat_cost = np.zeros((3,2,41))
    total_DACCS_heat_cost = np.zeros((3, 2, 41))

    total_MWh_DACCU = np.zeros((3,2,41))
    total_MWh_DAC_CDR = np.zeros((3,2,41))
    total_MWh_DAC_DeltaEmissions = np.zeros((3,2,41))

    if DAC_CDR_nonCO2_Gt_MIN is not None:
        total_DACCS_electricity_cost_MIN = np.zeros((3, 2, 41))
        total_DACCS_cost_MIN = np.zeros((3, 2, 41))
        total_MWh_DAC_CDR_MIN = np.zeros((3, 2, 41))
        total_MWh_DAC_DeltaEmissions_MIN = np.zeros((3, 2, 41))
        total_DACCS_heat_cost_MIN = np.zeros((3, 2, 41))

        finalcost_electricity_MIN = np.zeros((3, 2, 41))
        finalcost_transport_storageCO2_MIN = np.zeros((3, 2, 41))
        finalcost_MIN = np.zeros((3, 2, 41))
        finalcost_heat_MIN = np.zeros((3, 2, 41))

        total_DACCS_electricity_cost_MAX = np.zeros((3, 2, 41))
        total_DACCS_cost_MAX = np.zeros((3, 2, 41))
        total_MWh_DAC_CDR_MAX = np.zeros((3, 2, 41))
        total_MWh_DAC_DeltaEmissions_MAX = np.zeros((3, 2, 41))
        total_DACCS_heat_cost_MAX = np.zeros((3, 2, 41))

        finalcost_heat_MAX = np.zeros((3, 2, 41))
        finalcost_electricity_MAX = np.zeros((3, 2, 41))
        finalcost_transport_storageCO2_MAX = np.zeros((3, 2, 41))
        finalcost_MAX = np.zeros((3, 2, 41))

        # eliminate all 'negative' CDR rates by simply not doing CDR
        DAC_CDR_nonCO2_MWhth[DAC_CDR_nonCO2_MWhth < 0] = 0
        DAC_CDR_nonCO2_Gt[DAC_CDR_nonCO2_Gt < 0] = 0

    for i in range(0, 3): 
        for j in range(0, 2):

            total_MWh_DACCU[i,j] = DAC_DACCU_MWh[i,j] + H2_DACCU_MWh[i,j] + CO_DACCU_MWh[i,j] + FT_DACCU_MWh[i,j]
            total_MWh_DAC_CDR[i,j] = DAC_CDR_CO2_MWh[i,j] + DAC_CDR_nonCO2_MWh[i,j]
            total_MWh_DAC_DeltaEmissions[i,j] = Delta_totalIndirectEmissions[i,j] * DAC_electricity_need * (10**12) / (10**3)
            total_DACCS_heat_cost[i,j] = (DAC_CDR_CO2_MWhth[i,j] + DAC_CDR_nonCO2_MWhth[i,j]) * COST_LOW_TEMPERATURE_HEAT
            total_DACCU_heat_cost[i,j] = CO_DACCU_MWh_heat[i,j] * COST_HIGH_TEMPERATURE_HEAT + DAC_DACCU_MWh_heat[i,j]*COST_LOW_TEMPERATURE_HEAT

            finalcost_electricity[i,j] = ELECTRICITY_COST_KWH * 1000 * (total_MWh_DACCU[i,j] + total_MWh_DAC_CDR[i,j] + total_MWh_DAC_DeltaEmissions[i,j])
            finalcost_transport_storageCO2[i,j]  = CO2_TRANSPORT_STORAGE_COST * (10**9) *  (DAC_CDR_CO2_Gt[i,j] + DAC_CDR_nonCO2_Gt[i,j] + Delta_totalIndirectEmissions[i,j])
            finalcost_fossilfuel[i,j] = FF_MARKET_COST * FF_EJ_vector[i,j] * ENERGY_DENSITY_FUEL * (10**12)
            finalcost_heat[i,j] = total_DACCS_heat_cost[i,j]+total_DACCU_heat_cost[i,j]

            total_DACCU_electricity_cost[i,j] = ELECTRICITY_COST_KWH * 1000 * total_MWh_DACCU[i,j]
            total_DACCS_electricity_cost[i, j] = ELECTRICITY_COST_KWH * 1000 * ( total_MWh_DAC_CDR[i,j] + total_MWh_DAC_DeltaEmissions[i,j])

            total_DACCU_production_cost[i,j] = (yearly_DAC_DACCU_Cost[i,j] + total_yearly_H2_COST[i,j] + total_yearly_CO_COST[i,j]
                                                + total_DACCU_electricity_cost[i,j] + total_DACCU_heat_cost[i,j] + FT_DACCU_cost[i,j])
            total_DACCS_cost[i, j] = (
                        total_yearly_DAC_COST[i, j] + total_DACCS_electricity_cost[i, j] + total_DACCS_heat_cost[i,j])

            finalcost[i,j] = (total_yearly_DAC_COST[i,j] + total_yearly_CO_COST[i,j] + total_yearly_H2_COST[i,j] + FT_DACCU_cost[i,j] +
                                finalcost_electricity[i,j] + finalcost_transport_storageCO2[i,j] + finalcost_fossilfuel[i,j] + finalcost_heat[i,j])
            if DAC_CDR_nonCO2_Gt_MIN is not None:
                total_MWh_DAC_CDR_MIN[i, j] = DAC_CDR_CO2_MWh[i, j] + DAC_CDR_nonCO2_MWh_MIN[i, j]
                total_MWh_DAC_DeltaEmissions_MIN[i, j] = Delta_totalIndirectEmissions_MIN[i, j] * DAC_electricity_need * (
                            10 ** 12) / (10 ** 3)
                total_DACCS_electricity_cost_MIN[i, j] = ELECTRICITY_COST_KWH * 1000 * (
                            total_MWh_DAC_CDR_MIN[i, j] + total_MWh_DAC_DeltaEmissions_MIN[i, j])
                total_DACCS_heat_cost_MIN[i, j] = (DAC_CDR_CO2_MWhth[
                                                   i, j] + DAC_CDR_nonCO2_MWhth_MIN[i,j]) * COST_LOW_TEMPERATURE_HEAT
                total_DACCS_cost_MIN[i, j] = (
                        total_yearly_DAC_COST_MIN[i, j] + total_DACCS_electricity_cost[i, j] + total_DACCS_heat_cost_MIN[i,j])

                finalcost_electricity_MIN[i, j] = ELECTRICITY_COST_KWH * 1000 * (
                            total_MWh_DACCU[i, j] + total_MWh_DAC_CDR_MIN[i, j] + total_MWh_DAC_DeltaEmissions_MIN[i, j])
                finalcost_heat_MIN[i, j] = total_DACCS_heat_cost_MIN[i,j] + total_DACCU_heat_cost[i,j]
                finalcost_transport_storageCO2_MIN[i, j] = CO2_TRANSPORT_STORAGE_COST * (10 ** 9) * (
                            DAC_CDR_CO2_Gt[i, j] + DAC_CDR_nonCO2_Gt_MIN[i, j] + Delta_totalIndirectEmissions_MIN[i, j])
                finalcost_MIN[i, j] = (total_yearly_DAC_COST_MIN[i, j] + total_yearly_CO_COST[i, j] + total_yearly_H2_COST[i, j]
                                   + finalcost_electricity_MIN[i, j] + finalcost_transport_storageCO2_MIN[i, j] +
                                   finalcost_fossilfuel[i, j] + finalcost_heat_MIN[i,j])

                total_MWh_DAC_CDR_MAX[i, j] = DAC_CDR_CO2_MWh[i, j] + DAC_CDR_nonCO2_MWh_MAX[i, j]
                total_MWh_DAC_DeltaEmissions_MAX[i, j] = Delta_totalIndirectEmissions_MAX[
                                                             i, j] * DAC_electricity_need * (
                                                                 10 ** 12) / (10 ** 3)
                total_DACCS_electricity_cost_MAX[i, j] = ELECTRICITY_COST_KWH * 1000 * (
                        total_MWh_DAC_CDR_MAX[i, j] + total_MWh_DAC_DeltaEmissions_MAX[i, j])
                total_DACCS_heat_cost_MAX[i,j] = (DAC_CDR_CO2_MWhth[
                                                   i, j] + DAC_CDR_nonCO2_MWhth_MAX[i,j]) * COST_LOW_TEMPERATURE_HEAT
                total_DACCS_cost_MAX[i, j] = (
                        total_yearly_DAC_COST_MAX[i, j] + total_DACCS_electricity_cost[i, j] + total_DACCS_heat_cost_MAX[i,j])

                finalcost_electricity_MAX[i, j] = ELECTRICITY_COST_KWH * 1000 * (
                        total_MWh_DACCU[i, j] + total_MWh_DAC_CDR_MAX[i, j] + total_MWh_DAC_DeltaEmissions_MAX[i, j])
                finalcost_heat_MIN[i, j] = total_DACCS_heat_cost_MAX[i, j] + total_DACCU_heat_cost[i, j]
                finalcost_transport_storageCO2_MAX[i, j] = CO2_TRANSPORT_STORAGE_COST * (10 ** 9) * (
                        DAC_CDR_CO2_Gt[i, j] + DAC_CDR_nonCO2_Gt_MAX[i, j] + Delta_totalIndirectEmissions_MAX[i, j])
                finalcost_MAX[i, j] = (
                            total_yearly_DAC_COST_MAX[i, j] + total_yearly_CO_COST[i, j] + total_yearly_H2_COST[i, j]
                            + finalcost_electricity_MAX[i, j] + finalcost_transport_storageCO2_MAX[i, j] +
                            finalcost_fossilfuel[i, j] + finalcost_heat_MAX[i,j])

    finalcost_BAU = finalcost_fossilfuel[0,1]

    if emissions_cost is not None:

        finalcost += emissions_cost
        finalcost_BAU += emissions_cost[0,1]

    if DAC_CDR_nonCO2_Gt_MIN is not None:
        return finalcost, finalcost_MIN, finalcost_MAX, finalcost_BAU, \
            finalcost_electricity, finalcost_electricity_MIN, finalcost_electricity_MAX, \
            finalcost_transport_storageCO2, finalcost_transport_storageCO2_MIN, finalcost_transport_storageCO2_MAX, \
            finalcost_fossilfuel, finalcost_heat, finalcost_heat_MIN, finalcost_heat_MAX,\
            total_DACCU_electricity_cost, total_DACCS_electricity_cost, total_DACCS_electricity_cost_MIN, total_DACCS_electricity_cost_MAX, \
            total_DACCU_production_cost, total_DACCS_cost, total_DACCS_cost_MIN, total_DACCS_cost_MAX, \
            total_DACCU_heat_cost, total_DACCS_heat_cost, total_DACCS_heat_cost_MIN, total_DACCS_heat_cost_MAX
    else:
        return finalcost, finalcost_BAU, finalcost_electricity, finalcost_transport_storageCO2, finalcost_fossilfuel, finalcost_heat, \
            total_DACCU_electricity_cost, total_DACCS_electricity_cost, total_DACCU_production_cost, total_DACCS_cost, \
            total_DACCU_heat_cost, total_DACCS_heat_cost

#==================================================================================================


# SUPER FUNCTION (run the whole model, from all inputs parameters, export most output intermediate and final variable)


def run_model_II(df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY, LEARNING_RATE_H2, LEARNING_RATE_CO, LEARNING_RATE_DAC, ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_q0_Gt_2020, DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020, JETFUEL_ALLOCATION_SHARE,
                 CC_EFFICACY = 1, configuration_PtL = 'PEM+El.CO2+FT', efficiency_increase = False,
                 scenario = "Baseline", full_output = False):
    
    ### making base demand


    baseDemandEJ, baseDemandKM = make_base_demand_EJ_KM_II(df_input,GROWTH_RATE_AVIATION_FUEL_DEMAND,EFFICIENCY_INCREASE_YEARLY)


    DACCU_EJ_vector, FF_EJ_vector, DACCU_Tg_vector, FF_Tg_vector = make_DACCU_FF_EJ_Tg_II(df_input, baseDemandEJ) # Make DACCU and FF share of demand in EJ and KM for the 4 pathways

    # Creating historical demand for FF and DACCU Tg of fuel used and KM flown

    FF_KM_vector_1990, DACCU_KM_vector_1990, FF_Tg_vector_1990, DACCU_Tg_vector_1990 = make_historic_demand_II(df_input, df_emissions_input, baseDemandEJ, baseDemandKM)

    # Calculating total CO2 and non-CO2 emissions, as well as CO2 and non-CO2 emissios to abate for the 4 pathways

    emissions_Tg_ff, emissions_Tg_DACCU, emissions_Tg_total, ERF_ff, \
        ERF_DACCU,  \
        ERF_total,  = make_emissions_and_ERF(df_input, df_emissions_input, FF_KM_vector_1990,
                                                                         DACCU_KM_vector_1990, FF_Tg_vector_1990, DACCU_Tg_vector_1990,
                                                                         scenario=scenario, CC_efficacy = CC_EFFICACY
                                                                         )

    flying_CO2_emissions, flying_CO2_abated, flying_nonCO2_emissions, flying_nonCO2_abated  = \
        make_emissions_CO2equivalent_star(df_input, ERF_total, emissions_Tg_total)

    ### WTT Emission computing arrays
    # Vector Upstream + Vector Distribution + Vector Refining (df_Input) * yearly energy demand

    yearlyWTT = make_WTT_emissions_II(df_input, FF_EJ_vector)


    ### DACCU Production: DAC, H2, CO capacity + electricity requirements 
    ### + Fischer-Tropsch electricity only


    DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh, \
        DAC_Diesel_Gt, DAC_Diesel_MWh, H2_Diesel_Mt, H2_Diesel_MWh, CO_Diesel_Mt, CO_Diesel_MWh, FT_Diesel_MWh, \
        CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat, CO_Diesel_MWh_heat, DAC_Diesel_MWh_heat = \
        make_DACCU_need_DAC_H2_CO_FT_electricity_II(df_input, DACCU_Tg_vector, JETFUEL_ALLOCATION_SHARE,
                                                    configuration=configuration_PtL, efficiency_increase=efficiency_increase)

    
    ### DACCU Production: Material and electricity indirect emissions 

    (
        DAC_DACCU_MaterialFootprint, DAC_DACCU_ElectricityFootprint, H2_DACCU_MaterialFootprint, H2_DACCU_ElectricityFootprint,
     CO_DACCU_MaterialFootprint, CO_DACCU_ElectricityFootprint, FT_DACCU_ElectricityFootprint, totalDACCUMaterialFootprint,
     totalDACCUElectricitryFootprint,  totalDACCUFootprint

    ) =     make_DACCU_indirect_emissions_II(df_input, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh)
    
    
        ### DAC CDR CO2 & non-CO2 emissions: Capacity & electricity requirements
    ### + Material & Electricity footprint

    (
        DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_MWh, DAC_CDR_nonCO2_MWhth, DAC_CDR_CO2_MaterialFootprint, \
        DAC_CDR_CO2_ElectricityFootprint, DAC_CDR_nonCO2_MaterialFootprint, DAC_CDR_nonCO2_ElectricityFootprint, \
        total_DAC_CDR_Footprint
    ) = make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated, flying_nonCO2_abated)
    
    
    
        # total indirect emissions, Delta indirect emissions and net emissions calculations



    (
        totalIndirectEmissions, Delta_totalIndirectEmissions, BAU_EmissionsGt, 
         totalNetEmissions
    ) = make_indirect_delta_net_emissions_II(yearlyWTT, totalDACCUFootprint, total_DAC_CDR_Footprint, flying_CO2_emissions,flying_nonCO2_emissions,flying_CO2_abated,flying_nonCO2_abated)
    
    
    #### DAC Capacity & learning curves


    (
    totalYearlyDAC_need, total_DAC_InstalledCapacity, yearlyAddedCapacityDAC, cost_DAC_ct
    ) = make_learning_curve_DAC_II(LEARNING_RATE_DAC, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt, DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt, Delta_totalIndirectEmissions)
    
    #### H2 and CO Capacity & learning curves

    (
        totalYearlyH2_need, total_H2_InstalledCapacity, yearlyAddedCapacityH2, yearlyH2_need_increase,
     cost_H2_ct, totalYearlyCO_need, total_CO_InstalledCapacity, 
     yearlyAddedCapacityCO, yearlyCO_need_increase, cost_CO_ct

    ) = make_learning_curve_H2_CO_II(LEARNING_RATE_H2, LEARNING_RATE_CO, H2_q0_Mt_2020, H2_c0_2020, H2_DACCU_Mt, CO_q0_Mt_2020, CO_c0_2020, CO_DACCU_Mt)
    
    ##### DAC Costs



    (
        total_yearly_DAC_COST, yearly_DAC_DACCU_Cost, yearly_DAC_CDR_CO2_Cost,
     yearly_DAC_CDR_nonCO2_Cost, yearly_DAC_DeltaEmissions_Cost
    ) = calculate_DAC_costs_II(cost_DAC_ct, yearlyAddedCapacityDAC, totalYearlyDAC_need, DAC_DACCU_Gt, DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt, Delta_totalIndirectEmissions)
    
    
        ##### H2 and CO Costs


    total_yearly_H2_COST, total_yearly_CO_COST = calculate_H2_CO_costs_II(cost_H2_ct, yearlyH2_need_increase, cost_CO_ct, yearlyCO_need_increase)
    
    
        ###_______________________FINAL COST CALCULATIONS_______________________________________________

    (
    finalcost, finalcost_BAU, finalcost_electricity, finalcost_transport_storageCO2, finalcost_fossilfuel, finalcost_heat,
     total_DACCU_electricity_cost, total_DACCS_electricity_cost, total_DACCU_production_cost, total_DACCS_cost, total_DACCU_heat_cost, total_DACCS_heat_cost
    ) = calculate_final_cost_II(ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_DACCU_MWh,H2_DACCU_MWh,CO_DACCU_MWh,FT_DACCU_MWh,DAC_CDR_CO2_MWh,DAC_CDR_nonCO2_MWh, Delta_totalIndirectEmissions, DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt, FF_EJ_vector,yearly_DAC_DACCU_Cost, total_yearly_H2_COST, total_yearly_CO_COST , total_yearly_DAC_COST, df_input, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat, DAC_CDR_CO2_MWhth,
                            DAC_CDR_nonCO2_MWhth,)

    if full_output is False:
        return (finalcost, finalcost_BAU, finalcost_electricity, finalcost_transport_storageCO2, finalcost_fossilfuel, finalcost_heat, total_DACCU_electricity_cost, total_DACCU_production_cost, total_yearly_H2_COST, total_yearly_CO_COST, total_yearly_DAC_COST, yearly_DAC_DACCU_Cost, yearly_DAC_CDR_CO2_Cost, yearly_DAC_CDR_nonCO2_Cost, yearly_DAC_DeltaEmissions_Cost,totalIndirectEmissions, Delta_totalIndirectEmissions, BAU_EmissionsGt, totalNetEmissions, DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_MWh, DAC_CDR_CO2_MaterialFootprint, DAC_CDR_nonCO2_ElectricityFootprint, total_DAC_CDR_Footprint, DACCU_EJ_vector, FF_EJ_vector, DACCU_Tg_vector, FF_Tg_vector, flying_CO2_emissions, flying_CO2_abated, flying_nonCO2_emissions, flying_nonCO2_abated, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh,
                )
    else:
        return (finalcost, finalcost_BAU, finalcost_electricity, finalcost_transport_storageCO2, finalcost_fossilfuel,
                finalcost_heat, total_DACCU_electricity_cost, total_DACCU_production_cost, total_yearly_H2_COST,
                total_yearly_CO_COST, total_yearly_DAC_COST, yearly_DAC_DACCU_Cost, yearly_DAC_CDR_CO2_Cost,
                yearly_DAC_CDR_nonCO2_Cost, yearly_DAC_DeltaEmissions_Cost, totalIndirectEmissions,
                Delta_totalIndirectEmissions, BAU_EmissionsGt, totalNetEmissions, DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh,
                DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_MWh, DAC_CDR_CO2_MaterialFootprint,
                DAC_CDR_nonCO2_ElectricityFootprint, total_DAC_CDR_Footprint, DACCU_EJ_vector, FF_EJ_vector,
                DACCU_Tg_vector, FF_Tg_vector, flying_CO2_emissions, flying_CO2_abated, flying_nonCO2_emissions,
                flying_nonCO2_abated, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh,
                FT_DACCU_MWh, DAC_CDR_CO2_MWhth, DAC_CDR_CO2_MWhth, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat
                )


# SENSITIVITY ANALYSIS function

def run_sensitivity_analysis_III(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                             learning_rate_co, learning_rate_dac,
                             electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020, dac_c0_2020,
                             h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share, cc_efficacy, var_para_1, var1, var_para_2, var2, what,
                                scenario = "Baseline", configuration_PtL = 'PEM+El.CO2+FT', electrolysis_efficiency_increase = True):
    # make matrix to save the final cost (of what?) matrix with varying electricity price and learning rate
    Matrix_SENS_ANALYSIS = np.zeros((3, 2, 10, 10))

    if what == "BAU":
        # save BAU FF cost with varying prices
        SAdata_BAU_FF = np.zeros((10, 10))

    for i in range(0, 3):
        for j in range(0, 2):
            for x_sens in range(0, 10):
                for y_sens in range(0, 10):
                    if var_para_1 == "electricity cost":
                        tmp1 = electricity_cost
                        electricity_cost = var1[x_sens]
                    elif var_para_1 == "learning rate":
                        tmp1 = learning_rate_h2
                        tmpa = learning_rate_co
                        tmpb = learning_rate_dac
                        learning_rate_h2 = var1[x_sens]
                        learning_rate_co = var1[x_sens]
                        learning_rate_dac = var1[x_sens]
                    elif var_para_1 == "learning rate H2":
                        tmp1 = learning_rate_h2
                        learning_rate_h2 = var1[x_sens]
                    elif var_para_1 == "learning rate CO":
                        tmp1 = learning_rate_co
                        learning_rate_co = var1[x_sens]
                    elif var_para_1 == "learning rate DAC":
                        tmp1 = learning_rate_dac
                        learning_rate_dac = var1[x_sens]
                    elif var_para_1 == "growth rate":
                        tmp1 = growth_rate
                        growth_rate = var1[x_sens]
                    elif var_para_1 == "fossil fuel cost":
                        tmp1 = ff_market_cost
                        ff_market_cost = var1[x_sens]
                    elif var_para_1 == "DAC initial cost":
                        tmp1 = dac_c0_2020
                        dac_c0_2020 = var1[x_sens]
                    elif var_para_1 == "jetfuel allocation":
                        tmp1 = jetfuel_allocation_share
                        jetfuel_allocation_share = var1[x_sens]
                    elif var_para_1 == "cc efficacy":
                        tmp1 = cc_efficacy
                        cc_efficacy = var1[x_sens]

                    if var_para_2 == "electricity cost":
                        tmp2 = electricity_cost
                        electricity_cost = var2[y_sens]
                    elif var_para_2 == "learning rate":
                        tmp2 = learning_rate_h2
                        tmpx = learning_rate_co
                        tmpz = learning_rate_dac
                        learning_rate_h2 = var2[y_sens]
                        learning_rate_co = var2[y_sens]
                        learning_rate_dac = var2[y_sens]
                    elif var_para_2 == "learning rate H2":
                        tmp2 = learning_rate_h2
                        learning_rate_h2 = var2[y_sens]
                    elif var_para_2 == "learning rate CO":
                        tmp2 = learning_rate_co
                        learning_rate_co = var2[y_sens]
                    elif var_para_2 == "learning rate DAC":
                        tmp2 = learning_rate_dac
                        learning_rate_dac = var2[y_sens]
                    elif var_para_2 == "growth rate":
                        tmp2 = growth_rate
                        growth_rate = var2[y_sens]
                    elif var_para_2 == "fossil fuel cost":
                        tmp2 = ff_market_cost
                        ff_market_cost = var2[y_sens]
                    elif var_para_2 == "DAC initial cost":
                        tmp2 = dac_c0_2020
                        dac_c0_2020 = var2[y_sens]
                    elif var_para_2 == "jetfuel allocation":
                        tmp2 = jetfuel_allocation_share
                        jetfuel_allocation_share = var2[y_sens]
                    elif var_para_2 == "cc efficacy":
                        tmp2 = cc_efficacy
                        cc_efficacy = var2[y_sens]

                    finalcost = run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                          learning_rate_co, learning_rate_dac,
                                          electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                          dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share,
                                             cc_efficacy,
                                             configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase,
                                             scenario = scenario)[0]


                    if what == "BAU":
                        finalcost_fossilfuel = \
                        run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                  learning_rate_co, learning_rate_dac,
                                  electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                  dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share,
                                     cc_efficacy,
                                     configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase,
                                     scenario = scenario)[4]

                        SAdata_BAU_FF[x_sens, y_sens] = finalcost_fossilfuel[0, 1, 40]

                    Matrix_SENS_ANALYSIS[i, j, x_sens, y_sens] = finalcost[i, j, 40]

    # Restore the original variables for the next iteration
    if var_para_1 == "electricity cost":
        electricity_cost = tmp1
    elif var_para_1 == "learning rate":
        learning_rate_h2 = tmp1
        learning_rate_co = tmpa
        learning_rate_dac = tmpb
    elif var_para_1 == "learning rate H2":
        learning_rate_h2 = tmp1
    elif var_para_1 == "learning rate CO":
        learning_rate_co = tmp1
    elif var_para_1 == "learning rate DAC":
        learning_rate_DAC = tmp1
    elif var_para_1 == "growth rate":
        growth_rate = tmp1
    elif var_para_1 == "fossil fuel cost":
        ff_market_cost = tmp1
    elif var_para_1 == "DAC initial cost":
        dac_c0_2020 = tmp1
    elif var_para_1 == "jetfuel allocation":
        jetfuel_allocation_share = tmp1
    elif var_para_1 == "cc efficacy":
        cc_efficacy = tmp1


    if var_para_2 == "electricity cost":
        electricity_cost = tmp2
    elif var_para_2 == "learning rate":
        learning_rate_h2 = tmp2
        learning_rate_co = tmpx
        learning_rate_dac = tmpz
    elif var_para_2 == "learning rate H2":
        learning_rate_h2 = tmp2
    elif var_para_2 == "learning rate CO":
        learning_rate_co = tmp2
    elif var_para_2 == "learning rate DAC":
        learning_rate_DAC = tmp2
    elif var_para_2 == "growth rate":
        growth_rate = tmp2
    elif var_para_2 == "fossil fuel cost":
        ff_market_cost = tmp2
    elif var_para_2 == "DAC initial cost":
        dac_c0_2020 = tmp2
    elif var_para_2 == "jetfuel allocation":
        jetfuel_allocation_share = tmp2
    elif var_para_2 == "cc efficacy":
        cc_efficacy = tmp2

    if what == "BAU":
        SAdata_DACCU_VS_BAU_BASELINE = (Matrix_SENS_ANALYSIS[0, 0] - SAdata_BAU_FF) / (10 ** 12)
        SAdata_DACCU_VS_BAU_NEUTRAL = (Matrix_SENS_ANALYSIS[2, 0] - SAdata_BAU_FF) / (10 ** 12)
        SAdata_DACCU_VS_BAU_CARBON = (Matrix_SENS_ANALYSIS[2, 0] - SAdata_BAU_FF) / (10 ** 12)
        df_SA_DACCU_vs_BAU_BASELINE = pd.DataFrame(SAdata_DACCU_VS_BAU_BASELINE.transpose(),
                                          columns=var1.round(3),
                                          index=var2.round(3))
        df_SA_DACCU_vs_BAU_NEUTRAL = pd.DataFrame(SAdata_DACCU_VS_BAU_NEUTRAL.transpose(),
                                               columns=var1.round(3),
                                               index=var2.round(3))
        df_SA_DACCU_vs_BAU_CARBON = pd.DataFrame(SAdata_DACCU_VS_BAU_CARBON.transpose(),
                                                 columns=var1.round(3),
                                                 index=var2.round(3))
        return df_SA_DACCU_vs_BAU_BASELINE, df_SA_DACCU_vs_BAU_NEUTRAL, df_SA_DACCU_vs_BAU_CARBON


    else:
        SAdata_DACCU_BASELINE = (Matrix_SENS_ANALYSIS[0, 0] - Matrix_SENS_ANALYSIS[0, 1]) / (
                10 ** 12)

        SAdata_DACCU_NEUTRAL = (Matrix_SENS_ANALYSIS[1, 0] - Matrix_SENS_ANALYSIS[1, 1]) / (10 ** 12)

        SAdata_DACCU_CARBON = (Matrix_SENS_ANALYSIS[2, 0] - Matrix_SENS_ANALYSIS[2, 1]) / (10 ** 12)

        df_SA_baseline = pd.DataFrame(SAdata_DACCU_BASELINE.transpose(),
                                   columns=var1.round(3),
                                   index=var2.round(3))

        df_SA_neutrality = pd.DataFrame(SAdata_DACCU_NEUTRAL.transpose(),
                                     columns=var1.round(3),
                                     index=var2.round(3))
        df_SA_carbon = pd.DataFrame(SAdata_DACCU_CARBON.transpose(),
                                        columns=var1.round(3),
                                        index=var2.round(3))

        return df_SA_baseline, df_SA_neutrality, df_SA_carbon


def run_sensitivity_analysis_single_factors(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                             learning_rate_co, learning_rate_dac,
                             electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020, dac_c0_2020,
                             h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share, cc_efficacy,
                                            var_para_1, var1, what,
                                scenario = "Baseline", configuration_PtL = 'PEM+El.CO2+FT', electrolysis_efficiency_increase = True):
    # make matrix to save the final cost (of what?) matrix with varying electricity price and learning rate
    Matrix_SENS_ANALYSIS = np.zeros((3, 2, 10, 41))

    if what == "BAU":
        # save BAU FF cost with varying prices
        SAdata_BAU_FF = np.zeros((10, 41))

    for i in range(0, 3):
        for j in range(0, 2):
            for x_sens in range(0, 10):
                    if var_para_1 == "electricity cost":
                        tmp1 = electricity_cost
                        electricity_cost = var1[x_sens]
                    elif var_para_1 == "learning rate":
                        tmp1 = learning_rate_h2
                        tmpa = learning_rate_co
                        tmpb = learning_rate_dac
                        learning_rate_h2 = var1[x_sens]
                        learning_rate_co = var1[x_sens]
                        learning_rate_dac = var1[x_sens]
                    elif var_para_1 == "learning rate H2":
                        tmp1 = learning_rate_h2
                        learning_rate_h2 = var1[x_sens]
                    elif var_para_1 == "learning rate CO":
                        tmp1 = learning_rate_co
                        learning_rate_co = var1[x_sens]
                    elif var_para_1 == "learning rate DAC":
                        tmp1 = learning_rate_dac
                        learning_rate_dac = var1[x_sens]
                    elif var_para_1 == "growth rate":
                        tmp1 = growth_rate
                        growth_rate = var1[x_sens]
                    elif var_para_1 == "fossil fuel cost":
                        tmp1 = ff_market_cost
                        ff_market_cost = var1[x_sens]
                    elif var_para_1 == "DAC initial cost":
                        tmp1 = dac_c0_2020
                        dac_c0_2020 = var1[x_sens]
                    elif var_para_1 == "jetfuel allocation":
                        tmp1 = jetfuel_allocation_share
                        jetfuel_allocation_share = var1[x_sens]
                    elif var_para_1 == "cc efficacy":
                        tmp1 = cc_efficacy
                        cc_efficacy = var1[x_sens]
                    elif var_para_1 == "fuel efficiency":
                        tmp1 = efficiency_increase
                        efficiency_increase = var1[x_sens]


                    finalcost = run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                          learning_rate_co, learning_rate_dac,
                                          electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                          dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share,
                                             cc_efficacy, configuration_PtL, electrolysis_efficiency_increase, scenario)[0]


                    if what == "BAU":
                        finalcost_fossilfuel = \
                        run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                  learning_rate_co, learning_rate_dac,
                                  electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                  dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share, cc_efficacy,
                                       configuration_PtL, electrolysis_efficiency_increase, scenario)[4]

                        SAdata_BAU_FF[x_sens,:] = finalcost_fossilfuel[0, 1]

                    Matrix_SENS_ANALYSIS[i, j, x_sens] = finalcost[i, j]

    # Restore the original variables for the next iteration
    if var_para_1 == "electricity cost":
        electricity_cost = tmp1
    elif var_para_1 == "learning rate":
        learning_rate_h2 = tmp1
        learning_rate_co = tmpa
        learning_rate_dac = tmpb
    elif var_para_1 == "learning rate H2":
        learning_rate_h2 = tmp1
    elif var_para_1 == "learning rate CO":
        learning_rate_co = tmp1
    elif var_para_1 == "learning rate DAC":
        learning_rate_DAC = tmp1
    elif var_para_1 == "growth rate":
        growth_rate = tmp1
    elif var_para_1 == "fossil fuel cost":
        ff_market_cost = tmp1
    elif var_para_1 == "DAC initial cost":
        dac_c0_2020 = tmp1
    elif var_para_1 == "jetfuel allocation":
        jetfuel_allocation_share = tmp1
    elif var_para_1 == "cc efficacy":
        cc_efficacy = tmp1
    elif var_para_1 == "fuel efficiency":
        efficiency_increase = tmp1

    dates = np.arange(2020, 2061)

    if what == "BAU":
        SAdata_DACCU_VS_BAU_BASELINE = (Matrix_SENS_ANALYSIS[0, 0] - SAdata_BAU_FF) / (10 ** 12)
        SAdata_DACCU_VS_BAU_NEUTRAL = (Matrix_SENS_ANALYSIS[1, 0] - SAdata_BAU_FF) / (10 ** 12)
        SAdata_DACCU_VS_BAU_CARBON = (Matrix_SENS_ANALYSIS[2, 0] - SAdata_BAU_FF) / (10 ** 12)
        df_SA_DACCU_vs_BAU_BASELINE = pd.DataFrame(SAdata_DACCU_VS_BAU_BASELINE.transpose(),
                                          columns=var1.round(3),
                                          index=dates)
        df_SA_DACCU_vs_BAU_NEUTRAL = pd.DataFrame(SAdata_DACCU_VS_BAU_NEUTRAL.transpose(),
                                               columns=var1.round(3),
                                               index=dates)
        df_SA_DACCU_vs_BAU_CARBON = pd.DataFrame(SAdata_DACCU_VS_BAU_CARBON.transpose(),
                                                 columns=var1.round(3),
                                                 index=dates)
        return df_SA_DACCU_vs_BAU_BASELINE, df_SA_DACCU_vs_BAU_NEUTRAL, df_SA_DACCU_vs_BAU_CARBON


    else:
        SAdata_DACCU_BASELINE = (Matrix_SENS_ANALYSIS[0, 0] - Matrix_SENS_ANALYSIS[0, 1]) / (
                10 ** 12)

        SAdata_DACCU_NEUTRAL = (Matrix_SENS_ANALYSIS[1, 0] - Matrix_SENS_ANALYSIS[1, 1]) / (10 ** 12)

        SAdata_DACCU_CARBON = (Matrix_SENS_ANALYSIS[2, 0] - Matrix_SENS_ANALYSIS[2, 1]) / (10 ** 12)

        df_SA_baseline = pd.DataFrame(SAdata_DACCU_BASELINE.transpose(),
                                   columns=var1.round(3),
                                   index=dates)

        df_SA_neutrality = pd.DataFrame(SAdata_DACCU_NEUTRAL.transpose(),
                                     columns=var1.round(3),
                                     index=dates)
        df_SA_carbon = pd.DataFrame(SAdata_DACCU_CARBON.transpose(),
                                        columns=var1.round(3),
                                        index=dates)

        return df_SA_baseline, df_SA_neutrality, df_SA_carbon


def run_uncertainty_analysis(df_input, df_emissions_input, growth_rate_min, growth_rate_max, efficiency_increase_min,
                             efficiency_increase_max, learning_rate_h2_min, learning_rate_h2_max, learning_rate_co_min,
                             learning_rate_co_max, learning_rate_dac_min, learning_rate_dac_max, electricity_cost_min,
                             electricity_cost_max, ff_market_cost_min, ff_market_cost_max, co2_transport_storage_cost_min,
                             co2_transport_storage_cost_max, dac_q0_2020_min, dac_q0_2020_max, dac_c0_2020_min, dac_c0_2020_max,
                             h2_q0_2020_min, h2_q0_2020_max, h2_c0_2020_min, h2_c0_2020_max, co_q0_2020_min, co_q0_2020_max,
                             co_c0_2020_min, co_c0_2020_max, jetfuel_allocation_share_min, jetfuel_allocation_share_max,
                             cc_efficacy_min, cc_efficacy_max):
    # make matrix to save the final cost (of what?) matrix with varying electricity price and learning rate


    variables = {
        "growth_rate": (growth_rate_min, growth_rate_max),
        "efficiency_increase": (efficiency_increase_min, efficiency_increase_max),
        "learning_rate_H2": (learning_rate_h2_min, learning_rate_h2_max),
        "learning_rate_CO": ( learning_rate_co_min,learning_rate_co_max),
        "learning_rate_DAC": ( learning_rate_dac_min, learning_rate_dac_max),
        "electricity_cost": ( electricity_cost_min,electricity_cost_max),
        "ff_market_cost": ( ff_market_cost_min, ff_market_cost_max),
        "CO2_transport_storage_cost": ( co2_transport_storage_cost_min, co2_transport_storage_cost_max),
        "DAC_Q0_2020": ( dac_q0_2020_min, dac_q0_2020_max),
        "DAC_C0_2020": ( dac_c0_2020_min, dac_c0_2020_max),
        "H2_Q0_2020": ( h2_q0_2020_min, h2_q0_2020_max),
        "H2_C0_2020": ( h2_c0_2020_min, h2_c0_2020_max),
        "CO_Q0_2020": ( co_q0_2020_min, co_q0_2020_max),
        "CO_C0_2020": ( co_c0_2020_min, co_c0_2020_max),
        "jetfuel_allocation_share": ( jetfuel_allocation_share_min, jetfuel_allocation_share_max),
        "cc_efficacy": ( cc_efficacy_min, cc_efficacy_max)
    }

    # Remove keys with identical min and max values (single values)
    variables = {key: value for key, value in variables.items() if value[0] != value[1]}

    # Generate all possible combinations
    combinations = [dict(zip(variables.keys(), values)) for values in itertools.product(*variables.values())]

    # Prepare a list to store the results
    results_finalcost = []
    results_totalNetEmissions = []
    results_DAC_tot = []

    # Iterate through each combination and run the function
    # Iterate through each combination and run the function
    for combination in combinations:
        finalcost, finalcost_BAU, finalcost_electricity, finalcost_transport_storageCO2, finalcost_fossilfuel, \
            finalcost_heat, total_DACCU_electricity_cost, total_DACCU_production_cost, total_yearly_H2_COST, \
            total_yearly_CO_COST, total_yearly_DAC_COST, yearly_DAC_DACCU_Cost, yearly_DAC_CDR_CO2_Cost, \
            yearly_DAC_CDR_nonCO2_Cost, yearly_DAC_DeltaEmissions_Cost,totalIndirectEmissions, \
            Delta_totalIndirectEmissions, BAU_EmissionsGt, totalNetEmissions, DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, \
            DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_MWh, DAC_CDR_CO2_MaterialFootprint, DAC_CDR_nonCO2_ElectricityFootprint, \
            total_DAC_CDR_Footprint, DACCU_EJ_vector, FF_EJ_vector, DACCU_Tg_vector, FF_Tg_vector, flying_CO2_emissions, \
            flying_CO2_abated, flying_nonCO2_emissions, flying_nonCO2_abated, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, \
            H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh  = \
            run_model_II(df_input, df_emissions_input, combination['growth_rate'],
                                  combination['efficiency_increase'], combination['learning_rate_H2'],
                                  combination['learning_rate_CO'], combination['learning_rate_DAC'],
                                  combination['electricity_cost'], combination['ff_market_cost'],
                                  combination['CO2_transport_storage_cost'], combination['DAC_Q0_2020'],
                                  combination['DAC_C0_2020'], combination['H2_Q0_2020'], combination['H2_C0_2020'],
                                  combination['CO_Q0_2020'], combination['CO_C0_2020'],
                                  combination['jetfuel_allocation_share'], combination['cc_efficacy'],
                                  scenario = 'Both')  # Assuming 'scenario' is 'Both' for all combinations
        results_finalcost.append((combination, finalcost[:,:,40]))
        results_totalNetEmissions.append((combination, totalNetEmissions[:,:,40]))
        results_DAC_tot.append((combination, DAC_CDR_CO2_Gt[:,:,40]+DAC_CDR_nonCO2_Gt[:,:,40]+DAC_DACCU_Gt[:,:,40]))

    return results_finalcost, results_totalNetEmissions, results_DAC_tot


def make_min_max_combination(df_uncertain, what = 'Final cost'):
    columns = df_uncertain.columns
    # Find the index where 'Final cost DACCU baseline' is minimal
    min_index_DACCU_baseline = df_uncertain[columns[1]].idxmin()
    min_index_DACCS_baseline = df_uncertain[columns[2]].idxmin()
    min_index_DACCU_neutrality = df_uncertain[columns[3]].idxmin()
    min_index_DACCS_neutrality = df_uncertain[columns[4]].idxmin()
    min_index_DACCU_carbon = df_uncertain[columns[5]].idxmin()
    min_index_DACCS_carbon = df_uncertain[columns[6]].idxmin()
    # Retrieve the corresponding combination using the found index
    min_combination_DACCU_baseline = df_uncertain.loc[min_index_DACCU_baseline, 'Combination']
    min_combination_DACCS_baseline = df_uncertain.loc[min_index_DACCS_baseline, 'Combination']
    min_combination_DACCU_neutrality = df_uncertain.loc[min_index_DACCU_neutrality, 'Combination']
    min_combination_DACCS_neutrality = df_uncertain.loc[min_index_DACCS_neutrality, 'Combination']
    min_combination_DACCU_carbon = df_uncertain.loc[min_index_DACCU_carbon, 'Combination']
    min_combination_DACCS_carbon = df_uncertain.loc[min_index_DACCS_carbon, 'Combination']
    # Retrieve the corresponding cost using the found index
    min_value_DACCU_baseline = df_uncertain.loc[min_index_DACCU_baseline, columns[1]]
    min_value_DACCS_baseline = df_uncertain.loc[min_index_DACCS_baseline,  columns[2]]
    min_value_DACCU_neutrality = df_uncertain.loc[min_index_DACCU_neutrality,  columns[3]]
    min_value_DACCS_neutrality = df_uncertain.loc[min_index_DACCS_neutrality,  columns[4]]
    min_value_DACCU_carbon = df_uncertain.loc[min_index_DACCU_carbon,  columns[5]]
    min_value_DACCS_carbon = df_uncertain.loc[min_index_DACCS_carbon,  columns[6]]
    # Create a DataFrame with the desired information
    min_df = pd.DataFrame({
        'Scenario': ['DACCU baseline', 'DACCS baseline', 'DACCU neutrality', 'DACCS neutrality', 'DACCU carbon', 'DACCS carbon'],
        'Combination': [
            min_combination_DACCU_baseline,
            min_combination_DACCS_baseline,
            min_combination_DACCU_neutrality,
            min_combination_DACCS_neutrality,
            min_combination_DACCU_carbon,
            min_combination_DACCS_carbon
        ],
        'Min '+ what : [
            min_value_DACCU_baseline,
            min_value_DACCS_baseline,
            min_value_DACCU_neutrality,
            min_value_DACCS_neutrality,
            min_value_DACCU_carbon,
            min_value_DACCS_carbon
        ]
    })

    # Find the index where 'Final cost DACCU baseline' is maximal
    max_index_DACCU_baseline = df_uncertain[columns[1]].idxmax()
    max_index_DACCS_baseline = df_uncertain[columns[2]].idxmax()
    max_index_DACCU_neutrality = df_uncertain[columns[3]].idxmax()
    max_index_DACCS_neutrality = df_uncertain[columns[4]].idxmax()
    max_index_DACCU_carbon = df_uncertain[columns[5]].idxmax()
    max_index_DACCS_carbon = df_uncertain[columns[6]].idxmax()
    # Retrieve the corresponding combination using the found index
    max_combination_DACCU_baseline = df_uncertain.loc[max_index_DACCU_baseline, 'Combination']
    max_combination_DACCS_baseline = df_uncertain.loc[max_index_DACCS_baseline, 'Combination']
    max_combination_DACCU_neutrality = df_uncertain.loc[max_index_DACCU_neutrality, 'Combination']
    max_combination_DACCS_neutrality = df_uncertain.loc[max_index_DACCS_neutrality, 'Combination']
    max_combination_DACCU_carbon = df_uncertain.loc[max_index_DACCU_carbon, 'Combination']
    max_combination_DACCS_carbon = df_uncertain.loc[max_index_DACCS_carbon, 'Combination']
    # Retrieve the corresponding cost using the found index
    max_value_DACCU_baseline = df_uncertain.loc[max_index_DACCU_baseline, columns[1]]
    max_value_DACCS_baseline = df_uncertain.loc[max_index_DACCS_baseline,  columns[2]]
    max_value_DACCU_neutrality = df_uncertain.loc[max_index_DACCU_neutrality,  columns[3]]
    max_value_DACCS_neutrality = df_uncertain.loc[max_index_DACCS_neutrality,  columns[4]]
    max_value_DACCU_carbon = df_uncertain.loc[max_index_DACCU_carbon,  columns[5]]
    max_value_DACCS_carbon = df_uncertain.loc[max_index_DACCS_carbon,  columns[6]]
    # Create a DataFrame with the desired information
    max_df = pd.DataFrame({
        'Scenario': ['DACCU baseline', 'DACCS baseline', 'DACCU neutrality', 'DACCS neutrality', 'DACCU carbon', 'DACCS carbon'],
        'Combination': [
            max_combination_DACCU_baseline,
            max_combination_DACCS_baseline,
            max_combination_DACCU_neutrality,
            max_combination_DACCS_neutrality,
            max_combination_DACCU_carbon,
            max_combination_DACCS_carbon
        ],
        'Max '+ what : [
            max_value_DACCU_baseline,
            max_value_DACCS_baseline,
            max_value_DACCU_neutrality,
            max_value_DACCS_neutrality,
            max_value_DACCU_carbon,
            max_value_DACCS_carbon
        ]
    })

    return min_df, max_df


def find_varying_and_constant_variables(cluster_data):
    # Convert 'Combination' column from JSON strings to dictionaries
    cluster_data['Combination'] = cluster_data['Combination'].apply(json.loads)
    # Select any row from the DataFrame
    sample_row = cluster_data.iloc[0]

    # Extract the 'Combination' value from the selected row
    combination_sample = sample_row['Combination']

    # Initialize dictionaries to store varying and constant variables
    varying_variables = {}
    constant_variables = {}

    # Iterate through the keys of the 'Combination' dictionary
    for key in combination_sample.keys():
        values = set()
        is_varying = False

        # Iterate through the rows to check if the values vary or stay constant
        for index, row in cluster_data.iterrows():
            if row['Combination'][key] not in values:
                values.add(row['Combination'][key])
                if len(values) > 1:
                    is_varying = True
                    break

        if is_varying:
            varying_variables[key] = values
        else:
            constant_variables[key] = values.pop()

    variables_to_remove = ['DAC_Q0_2020', 'H2_Q0_2020', 'CO_Q0_2020', 'jetfuel_allocation_share']

    # Remove variables from constant_vars
    for var in variables_to_remove:
        if var in constant_variables:
            constant_variables.pop(var)

    return varying_variables, constant_variables


def make_kmeans_categories(df_uncertain, k = 50):
    values = df_uncertain.drop(columns=['Combination']).values
    # apply K-Means clustering to your data
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(values)
    # Add the cluster labels to your DataFrame
    df_cluster_uncertain = df_uncertain.copy()
    df_cluster_uncertain['Cluster'] = clusters

    # Find varying and constant variables for each cluster
    for cluster_id in df_cluster_uncertain['Cluster'].unique():
        cluster_data = df_cluster_uncertain[df_cluster_uncertain['Cluster'] == cluster_id]
        varying_vars, constant_vars = find_varying_and_constant_variables(cluster_data)

        # If there are constant variables, use them as cluster name
        if constant_vars:
            cluster_name = "_".join([f"{key}_{value}" for key, value in constant_vars.items()])
        else:
            cluster_name = f"Cluster_{cluster_id}"
        df_cluster_uncertain['Constant'] = cluster_name
        #df_cluster_uncertain.loc[df_cluster_uncertain['Cluster'] == cluster_id, 'Cluster'] = cluster_name

    return df_cluster_uncertain

def add_emissions_prices(emissions, price, type = 'constant', growth_rate = None):
    if type == 'constant':
        cost_emissions = emissions*10**9*price # Emissions (GtCO2) * 10^^ --> tCO2 x €/tCO2 --> €
    elif type == 'exponential':
        t = np.arange(0, len(emissions[0,1]))
        price_in_t = [price*(1+growth_rate)**i for i in t]
        cost_emissions = emissions*1e9*price_in_t
    else:
        raise ValueError("Invalid type. Use 'constant' or 'exponential'.")
    return price_in_t, cost_emissions #np.concatenate(cost_emissions, axis = 0)

# Define the objective function to minimize
def objective_function(var_name, var_value, what, scenario):
    # Call your function with the provided growth_rate and other parameters
    # Importing necessary input files
    df_input = pd.read_csv('base_input.csv')
    df_emissions_input = pd.read_csv('Input_emissions_indices_Lee.csv')
    if scenario == 'Baseline':
        index = 0
    elif scenario == 'Carbon neutrality':
        index = 2
    elif scenario == 'Climate neutrality':
        index = 1
    df_SA = run_sensitivity_analysis_single_factors(df_input, df_emissions_input, growth_rate = 0.02, efficiency_increase = 0.02, learning_rate_h2 = 0.08,
                             learning_rate_co = 0.05, learning_rate_dac = 0.012,
                             electricity_cost = 0.03, ff_market_cost = 0.06, co2_transport_storage_cost = 20, dac_q0_2020 = 0.003, dac_c0_2020 = 850,
                             h2_q0_2020 = 7, h2_c0_2020 = 915, co_q0_2020 = 0.002, co_c0_2020 = 108, jetfuel_allocation_share = 1/1.82,
                                                    cc_efficacy = 1,
                                                    var_para_1 = var_name, var1 = var_value, what = what,
                                scenario="Baseline")[index]  # Assuming you want the first value returned

    # Extract the last value of df_SA_baseline and negate it because we want to minimize
    last_value = df_SA.iloc[-1, -1]  # Assuming the last value is at the bottom-right corner
    return last_value  # Negate to minimize

#def objective_function_2(var_value):
    # Call your function with the provided var_name, var_value, what, and scenario
#    return -objective_function(var_name, var_value, what, scenario)


def run_min_max_uncertainty_analysis(df_input, df_emissions_input, growth_rate_min, growth_rate_max, efficiency_increase_min,
                             efficiency_increase_max, learning_rate_h2_min, learning_rate_h2_max, learning_rate_co_min,
                             learning_rate_co_max, learning_rate_dac_min, learning_rate_dac_max, electricity_cost_min,
                             electricity_cost_max, ff_market_cost_min, ff_market_cost_max, co2_transport_storage_cost_min,
                             co2_transport_storage_cost_max, dac_q0_2020_min, dac_q0_2020_max, dac_c0_2020_min, dac_c0_2020_max,
                             h2_q0_2020_min, h2_q0_2020_max, h2_c0_2020_min, h2_c0_2020_max, co_q0_2020_min, co_q0_2020_max,
                             co_c0_2020_min, co_c0_2020_max, jetfuel_allocation_share_min, jetfuel_allocation_share_max,
                             cc_efficacy_min, cc_efficacy_max):
    # make matrix to save the final cost (of what?) matrix with varying electricity price and learning rate


    variables = {
        "growth_rate": (growth_rate_min, growth_rate_max),
        "efficiency_increase": (efficiency_increase_min, efficiency_increase_max),
        "learning_rate_H2": (learning_rate_h2_min, learning_rate_h2_max),
        "learning_rate_CO": ( learning_rate_co_min,learning_rate_co_max),
        "learning_rate_DAC": ( learning_rate_dac_min, learning_rate_dac_max),
        "electricity_cost": ( electricity_cost_min,electricity_cost_max),
        "ff_market_cost": ( ff_market_cost_min, ff_market_cost_max),
        "CO2_transport_storage_cost": ( co2_transport_storage_cost_min, co2_transport_storage_cost_max),
        "DAC_Q0_2020": ( dac_q0_2020_min, dac_q0_2020_max),
        "DAC_C0_2020": ( dac_c0_2020_min, dac_c0_2020_max),
        "H2_Q0_2020": ( h2_q0_2020_min, h2_q0_2020_max),
        "H2_C0_2020": ( h2_c0_2020_min, h2_c0_2020_max),
        "CO_Q0_2020": ( co_q0_2020_min, co_q0_2020_max),
        "CO_C0_2020": ( co_c0_2020_min, co_c0_2020_max),
        "jetfuel_allocation_share": ( jetfuel_allocation_share_min, jetfuel_allocation_share_max),
        "cc_efficacy": ( cc_efficacy_min, cc_efficacy_max)
    }


    # Prepare a list to store the results
    results_finalcost = []
    results_totalNetEmissions = []
    results_DAC_tot = []

    # Run the model with minimum values for all variables
    finalcost_min, finalcost_BAU_min, finalcost_electricity_min, finalcost_transport_storageCO2_min_min, finalcost_fossilfuel_min, \
        finalcost_heat_min, total_DACCU_electricity_cost_min, total_DACCU_production_cost_min, total_yearly_H2_COST_min, total_yearly_CO_COST_min, \
        total_yearly_DAC_COST_min, yearly_DAC_DACCU_Cost_min, yearly_DAC_CDR_CO2_Cost_min, yearly_DAC_CDR_nonCO2_Cost_min, \
        yearly_DAC_DeltaEmissions_Cost,totalIndirectEmissions_min, Delta_totalIndirectEmissions_min, BAU_EmissionsGt_min, \
        totalNetEmissions_min, DAC_CDR_CO2_Gt_min, DAC_CDR_CO2_MWh_min, DAC_CDR_nonCO2_Gt_min, DAC_CDR_nonCO2_MWh_min, \
        DAC_CDR_CO2_MaterialFootprint_min, DAC_CDR_nonCO2_ElectricityFootprint_min, total_DAC_CDR_Footprint_min, DACCU_EJ_vector_min, \
        FF_EJ_vector_min, DACCU_Tg_vector_min, FF_Tg_vector_min, flying_CO2_emissions_min, flying_CO2_abated_min, flying_nonCO2_emissions_min, \
        flying_nonCO2_abated_min, DAC_DACCU_Gt_min, DAC_DACCU_MWh_min, H2_DACCU_Mt_min, H2_DACCU_MWh_min, CO_DACCU_Mt_min, CO_DACCU_MWh_min, FT_DACCU_MWh_min = \
        run_model_II(df_input, df_emissions_input, variables['growth_rate'][0],
                      variables['efficiency_increase'][0], variables['learning_rate_H2'][0],
                      variables['learning_rate_CO'][0], variables['learning_rate_DAC'][0],
                      variables['electricity_cost'][0], variables['ff_market_cost'][0],
                      variables['CO2_transport_storage_cost'][0], variables['DAC_Q0_2020'][0],
                      variables['DAC_C0_2020'][0], variables['H2_Q0_2020'][0], variables['H2_C0_2020'][0],
                      variables['CO_Q0_2020'][0], variables['CO_C0_2020'][0],
                      variables['jetfuel_allocation_share'][0], variables['cc_efficacy'][0],
                      scenario='Both')  # Assuming 'scenario' is 'Both' for all combinations

    finalcost_max, finalcost_BAU_max, finalcost_electricity_max, finalcost_transport_storageCO2_max_max, finalcost_fossilfuel_max, \
        finalcost_heat_max, total_DACCU_electricity_cost_max, total_DACCU_production_cost_max, total_yearly_H2_COST_max, total_yearly_CO_COST_max, \
        total_yearly_DAC_COST_max, yearly_DAC_DACCU_Cost_max, yearly_DAC_CDR_CO2_Cost_max, yearly_DAC_CDR_nonCO2_Cost_max, \
        yearly_DAC_DeltaEmissions_Cost,totalIndirectEmissions_max, Delta_totalIndirectEmissions_max, BAU_EmissionsGt_max, \
        totalNetEmissions_max, DAC_CDR_CO2_Gt_max, DAC_CDR_CO2_MWh_max, DAC_CDR_nonCO2_Gt_max, DAC_CDR_nonCO2_MWh_max, \
        DAC_CDR_CO2_MaterialFootprint_max, DAC_CDR_nonCO2_ElectricityFootprint_max, total_DAC_CDR_Footprint_max, DACCU_EJ_vector_max, \
        FF_EJ_vector_max, DACCU_Tg_vector_max, FF_Tg_vector_max, flying_CO2_emissions_max, flying_CO2_abated_max, flying_nonCO2_emissions_max, \
        flying_nonCO2_abated_max, DAC_DACCU_Gt_max, DAC_DACCU_MWh_max, H2_DACCU_Mt_max, H2_DACCU_MWh_max, CO_DACCU_Mt_max, CO_DACCU_MWh_max, FT_DACCU_MWh_max = \
        run_model_II(df_input, df_emissions_input, variables['growth_rate'][1],
                      variables['efficiency_increase'][1], variables['learning_rate_H2'][1],
                      variables['learning_rate_CO'][1], variables['learning_rate_DAC'][1],
                      variables['electricity_cost'][1], variables['ff_market_cost'][1],
                      variables['CO2_transport_storage_cost'][1], variables['DAC_Q0_2020'][1],
                      variables['DAC_C0_2020'][1], variables['H2_Q0_2020'][1], variables['H2_C0_2020'][1],
                      variables['CO_Q0_2020'][1], variables['CO_C0_2020'][1],
                      variables['jetfuel_allocation_share'][1], variables['cc_efficacy'][1],
                      scenario='Both')  # Assuming 'scenario' is 'Both' for all combinations

    results_finalcost.append((finalcost_min[:, :, 40], finalcost_max[:,:,40]))
    results_totalNetEmissions.append((totalNetEmissions_min[:, :, 40], totalNetEmissions_max[:, :, 40]))
    results_DAC_tot.append((DAC_CDR_CO2_Gt_min[:, :, 40] + DAC_CDR_nonCO2_Gt_min[:, :, 40] + DAC_DACCU_Gt_min[:, :, 40],
                            DAC_CDR_CO2_Gt_max[:, :, 40] + DAC_CDR_nonCO2_Gt_max[:, :, 40] + DAC_DACCU_Gt_max[:, :, 40]))

    return results_finalcost, results_totalNetEmissions, results_DAC_tot

def run_sensitivity_analysis_policies(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                             learning_rate_co, learning_rate_dac,
                             electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020, dac_c0_2020,
                             h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share, cc_efficacy,
                             policy_name, percentage_change, what, emissions_price = None, subsidy = None,
                                      subsidy_type = None, excess_electricity_price = None, growth_rate_capped = None,
                                      configuration_PtL = 'PEM+El.CO2+FT', electrolysis_efficiency_increase = True):
    # make matrix to save the final cost (of what?) matrix with varying electricity price and learning rate
    Matrix_SENS_ANALYSIS = np.zeros((3, 2, 10, 41))

    if what == "BAU":
        # save BAU FF cost with varying prices
        SAdata_BAU_FF = np.zeros((10, 41))

    for i in range(0, 3):
        for j in range(0, 2):
            if policy_name == "CO2 prices" or policy_name == 'prices on climate impacts':
                varying_emissions_prices = emissions_price * percentage_change
                var1 = varying_emissions_prices
                finalcost, finalcost_BAU, finalcost_electricity, finalcost_transport_storageCO2, finalcost_fossilfuel, \
                    finalcost_heat, total_DACCU_electricity_cost, total_DACCU_production_cost, total_yearly_H2_COST, total_yearly_CO_COST, \
                    total_yearly_DAC_COST, yearly_DAC_DACCU_Cost, yearly_DAC_CDR_CO2_Cost, yearly_DAC_CDR_nonCO2_Cost, \
                    yearly_DAC_DeltaEmissions_Cost,totalIndirectEmissions, Delta_totalIndirectEmissions, BAU_EmissionsGt, \
                    totalNetEmissions, DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_MWh, \
                    DAC_CDR_CO2_MaterialFootprint, DAC_CDR_nonCO2_ElectricityFootprint, total_DAC_CDR_Footprint, \
                    DACCU_EJ_vector, FF_EJ_vector, DACCU_Tg_vector, FF_Tg_vector, flying_CO2_emissions, flying_CO2_abated, \
                    flying_nonCO2_emissions, flying_nonCO2_abated, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, \
                    CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_MWhth, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat \
                    = run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                 learning_rate_co, learning_rate_dac,
                                 electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                 dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020,
                                 jetfuel_allocation_share, cc_efficacy, scenario = 'Both',
                                   configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase,
                                   full_output = True)
                # calculate cost of emissions if we have prices on CO2 or climate impacts
                for x_sens in np.arange(0, 10):
                    if policy_name == 'CO2 prices':
                        price_increasing, emissions_cost = add_emissions_prices(
                                flying_CO2_emissions + totalIndirectEmissions - flying_CO2_abated, varying_emissions_prices[x_sens], 'exponential',
                                growth_rate=0.01)
                        #print('Starting CO$_2$ price: '+ str(varying_emissions_prices[x_sens]) + "\n" + '2060 CO$_2$ price: '+ str(price_increasing[-1]))
                        price_increasing_BAU, emissions_cost_BAU = add_emissions_prices(
                                flying_CO2_emissions, varying_emissions_prices[x_sens], 'exponential',
                                growth_rate=0.01)
                    elif policy_name == "prices on climate impacts":
                        price_increasing, emissions_cost = add_emissions_prices(totalNetEmissions,
                                                                                varying_emissions_prices[x_sens],
                                                                                                      'exponential',
                                                                                                      growth_rate=0.01)
                        #print('Starting emissions price: ' + str(
                        #    varying_emissions_prices[x_sens]) + "\n" + '2060 emissions price: ' + str(
                        #    price_increasing[-1]))
                        price_increasing_BAU, emissions_cost_BAU = add_emissions_prices(
                            flying_CO2_emissions + flying_nonCO2_emissions, varying_emissions_prices[x_sens], 'exponential',
                            growth_rate=0.01)
                    # calculate final yearly costs including electricity, fossil fuel, CO2 transport and storage, etc.
                    finalcost_emissions_price = \
                        calculate_final_cost_II(electricity_cost, ff_market_cost, co2_transport_storage_cost,
                                                DAC_DACCU_MWh, H2_DACCU_MWh, CO_DACCU_MWh, FT_DACCU_MWh, DAC_CDR_CO2_MWh,
                                                DAC_CDR_nonCO2_MWh, Delta_totalIndirectEmissions, DAC_CDR_CO2_Gt,
                                                DAC_CDR_nonCO2_Gt, FF_EJ_vector, yearly_DAC_DACCU_Cost,
                                                total_yearly_H2_COST, total_yearly_CO_COST, total_yearly_DAC_COST,
                                                df_input, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_MWhth, emissions_cost)[0]

                    Matrix_SENS_ANALYSIS[i, j, x_sens] = finalcost_emissions_price[i, j]

                    if what == "BAU":
                        finalcost_BAU_emissions_price = \
                            calculate_final_cost_II(electricity_cost, ff_market_cost, co2_transport_storage_cost,
                                                    DAC_DACCU_MWh, H2_DACCU_MWh, CO_DACCU_MWh, FT_DACCU_MWh,
                                                    DAC_CDR_CO2_MWh,
                                                    DAC_CDR_nonCO2_MWh, Delta_totalIndirectEmissions, DAC_CDR_CO2_Gt,
                                                    DAC_CDR_nonCO2_Gt, FF_EJ_vector, yearly_DAC_DACCU_Cost,
                                                    total_yearly_H2_COST, total_yearly_CO_COST, total_yearly_DAC_COST,
                                                    df_input, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_MWhth, emissions_cost_BAU)[1]
                        SAdata_BAU_FF[x_sens, :] = finalcost_BAU_emissions_price

            elif policy_name == 'subsidies':
                varying_subsidy = percentage_change * subsidy
                var1 = varying_subsidy
                finalcost, finalcost_BAU, finalcost_electricity, finalcost_transport_storageCO2, finalcost_fossilfuel, finalcost_heat, \
                    total_DACCU_electricity_cost, total_DACCU_production_cost, total_yearly_H2_COST, total_yearly_CO_COST, \
                    total_yearly_DAC_COST, yearly_DAC_DACCU_Cost, yearly_DAC_CDR_CO2_Cost, yearly_DAC_CDR_nonCO2_Cost, \
                    yearly_DAC_DeltaEmissions_Cost,totalIndirectEmissions, Delta_totalIndirectEmissions, BAU_EmissionsGt, \
                    totalNetEmissions, DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_MWh, \
                    DAC_CDR_CO2_MaterialFootprint, DAC_CDR_nonCO2_ElectricityFootprint, total_DAC_CDR_Footprint, \
                    DACCU_EJ_vector, FF_EJ_vector, DACCU_Tg_vector, FF_Tg_vector, flying_CO2_emissions, flying_CO2_abated, \
                    flying_nonCO2_emissions, flying_nonCO2_abated, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, \
                    CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_MWhth, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat \
                    = run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                 learning_rate_co, learning_rate_dac,
                                 electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                 dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020,
                                 jetfuel_allocation_share, cc_efficacy, scenario = 'Both',
                                   configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase,
                                   full_output=True)
                for x_sens in np.arange(0,10):
                    t = np.arange(0, len(df_input.iloc[:,0]))
                    subsidy_change_rate = -0.01
                    subsidy_in_t = [varying_subsidy[x_sens] * (1 + subsidy_change_rate) ** i for i in t]
                    if subsidy_type == 'DACCS':
                        subsidised_DACCS_cost = subsidy_in_t*(DAC_CDR_CO2_Gt+DAC_CDR_nonCO2_Gt)*10**9 #subsidy needs to be in €/tCO2
                        Matrix_SENS_ANALYSIS[i, j, x_sens] = finalcost[i, j] - subsidised_DACCS_cost[i, j]
                    elif subsidy_type == 'DACCU':
                        subsidised_DACCU_cost = subsidy_in_t*DACCU_Tg_vector*10**9 #subsidy needs to be in €/kg DACCU fuel
                        Matrix_SENS_ANALYSIS[i, j, x_sens] = finalcost[i, j] - subsidised_DACCU_cost[i, j]

                    if what == "BAU":
                        finalcost_BAU_CO2_price = \
                            calculate_final_cost_II(electricity_cost, ff_market_cost, co2_transport_storage_cost,
                                                    DAC_DACCU_MWh, H2_DACCU_MWh, CO_DACCU_MWh, FT_DACCU_MWh,
                                                    DAC_CDR_CO2_MWh,
                                                    DAC_CDR_nonCO2_MWh, Delta_totalIndirectEmissions, DAC_CDR_CO2_Gt,
                                                    DAC_CDR_nonCO2_Gt, FF_EJ_vector, yearly_DAC_DACCU_Cost,
                                                    total_yearly_H2_COST, total_yearly_CO_COST, total_yearly_DAC_COST,
                                                    df_input, CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_MWhth)[1]
                        SAdata_BAU_FF[x_sens, :] = finalcost_BAU_CO2_price
            elif policy_name == 'excess electricity':
                varying_electricty_cost = percentage_change * excess_electricity_price
                var1 = varying_electricty_cost
                for x_sens in np.arange(0,10):
                    electricity_cost = var1[x_sens]
                    finalcost = run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                          learning_rate_co, learning_rate_dac,
                                          electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                          dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share,
                                             cc_efficacy, scenario = 'Both',
                                             configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase)[0]
                    Matrix_SENS_ANALYSIS[i, j, x_sens] = finalcost[i, j]
                    if what == "BAU":
                        finalcost_fossilfuel = \
                            run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase,
                                         learning_rate_h2,
                                         learning_rate_co, learning_rate_dac,
                                         electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                         dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020,
                                         jetfuel_allocation_share, cc_efficacy, scenario = 'Both',
                                         configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase)[4]
                        SAdata_BAU_FF[x_sens, :] = finalcost_fossilfuel[0, 1]
            elif policy_name == 'demand reduction':
                varying_demand_growth = percentage_change * growth_rate_capped
                var1 = varying_demand_growth
                for x_sens in np.arange(0,10):
                    growth_rate = var1[x_sens]
                    finalcost = run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase, learning_rate_h2,
                                          learning_rate_co, learning_rate_dac,
                                          electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                          dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share,
                                             cc_efficacy, scenario = 'Both',
                                             configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase)[0]
                    Matrix_SENS_ANALYSIS[i, j, x_sens] = finalcost[i, j]
                    if what == "BAU":
                        finalcost_fossilfuel = \
                            run_model_II(df_input, df_emissions_input, growth_rate, efficiency_increase,
                                         learning_rate_h2,
                                         learning_rate_co, learning_rate_dac,
                                         electricity_cost, ff_market_cost, co2_transport_storage_cost, dac_q0_2020,
                                         dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020,
                                         jetfuel_allocation_share, cc_efficacy, scenario = 'Both',
                                         configuration_PtL = configuration_PtL, efficiency_increase = electrolysis_efficiency_increase)[4]
                        SAdata_BAU_FF[x_sens, :] = finalcost_fossilfuel[0, 1]

    dates = np.arange(2020, 2061)
    if what == "BAU":
        SAdata_DACCU_VS_BAU_BASELINE = (Matrix_SENS_ANALYSIS[0, 0] - SAdata_BAU_FF) / (10 ** 12)
        SAdata_DACCU_VS_BAU_NEUTRAL = (Matrix_SENS_ANALYSIS[1, 0] - SAdata_BAU_FF) / (10 ** 12)
        SAdata_DACCU_VS_BAU_CARBON = (Matrix_SENS_ANALYSIS[2, 0] - SAdata_BAU_FF) / (10 ** 12)
        df_SA_DACCU_vs_BAU_BASELINE = pd.DataFrame(SAdata_DACCU_VS_BAU_BASELINE.transpose(),
                                                   columns=var1.round(3), index = dates)
        df_SA_DACCU_vs_BAU_NEUTRAL = pd.DataFrame(SAdata_DACCU_VS_BAU_NEUTRAL.transpose(),
                                                  columns=var1.round(3),
                                                  index=dates)
        df_SA_DACCU_vs_BAU_CARBON = pd.DataFrame(SAdata_DACCU_VS_BAU_CARBON.transpose(),
                                                 columns=var1.round(3),
                                                 index=dates)
        return df_SA_DACCU_vs_BAU_BASELINE, df_SA_DACCU_vs_BAU_NEUTRAL, df_SA_DACCU_vs_BAU_CARBON

    else:
        SAdata_DACCU_BASELINE = (Matrix_SENS_ANALYSIS[0, 0] - Matrix_SENS_ANALYSIS[0, 1]) / (
                10 ** 12)
        SAdata_DACCU_NEUTRAL = (Matrix_SENS_ANALYSIS[1, 0] - Matrix_SENS_ANALYSIS[1, 1]) / (10 ** 12)
        SAdata_DACCU_CARBON = (Matrix_SENS_ANALYSIS[2, 0] - Matrix_SENS_ANALYSIS[2, 1]) / (10 ** 12)
        df_SA_baseline = pd.DataFrame(SAdata_DACCU_BASELINE.transpose(),
                                   columns=var1.round(3),
                                   index=dates)
        df_SA_neutrality = pd.DataFrame(SAdata_DACCU_NEUTRAL.transpose(),
                                     columns=var1.round(3),
                                     index=dates)
        df_SA_carbon = pd.DataFrame(SAdata_DACCU_CARBON.transpose(),
                                        columns=var1.round(3),
                                        index=dates)
        return df_SA_baseline, df_SA_neutrality, df_SA_carbon



def calculate_cost_per_liter(cost,ff_Tg, DACCU_Tg, kg_to_liter = 1.22):
    return cost/((ff_Tg+DACCU_Tg)*10**9*kg_to_liter)

def calc_array_cost_per_liter(cost1, cost2, cost3, cost4, ff1, ff2, ff3, ff4, daccu1, daccu2, daccu3, daccu4,
                              cost5=None, cost6=None, cost7=None, ff5=None, ff6=None, ff7=None, daccu5=None, daccu6=None, daccu7=None):
    cost_per_liter1 = calculate_cost_per_liter(cost1, ff1, daccu1)
    cost_per_liter2 = calculate_cost_per_liter(cost2, ff2, daccu2)
    cost_per_liter3 = calculate_cost_per_liter(cost3, ff3, daccu3)
    cost_per_liter4 = calculate_cost_per_liter(cost4, ff4, daccu4)

    results = [cost_per_liter1, cost_per_liter2, cost_per_liter3, cost_per_liter4]

    if cost5 is not None:
        cost_per_liter5 = calculate_cost_per_liter(cost5, ff5, daccu5)
        results.append(cost_per_liter5)
    if cost6 is not None:
        cost_per_liter6 = calculate_cost_per_liter(cost6, ff6, daccu6)
        results.append(cost_per_liter6)
    if cost7 is not None:
        cost_per_liter7 = calculate_cost_per_liter(cost7, ff7, daccu7)
        results.append(cost_per_liter7)

    return tuple(results)

def calc_array_cost_per_liter_one_scenario(cost1, cost2, cost3, cost4, ff, daccu,
                              cost5=None, cost6=None, cost7=None):
    cost_per_liter1 = calculate_cost_per_liter(cost1, ff, daccu)
    cost_per_liter2 = calculate_cost_per_liter(cost2, ff, daccu)
    cost_per_liter3 = calculate_cost_per_liter(cost3, ff, daccu)
    cost_per_liter4 = calculate_cost_per_liter(cost4, ff, daccu)

    results = [cost_per_liter1, cost_per_liter2, cost_per_liter3, cost_per_liter4]

    if cost5 is not None:
        cost_per_liter5 = calculate_cost_per_liter(cost5, ff, daccu)
        results.append(cost_per_liter5)
    if cost6 is not None:
        cost_per_liter6 = calculate_cost_per_liter(cost6, ff, daccu)
        results.append(cost_per_liter6)
    if cost7 is not None:
        cost_per_liter7 = calculate_cost_per_liter(cost7, ff, daccu)
        results.append(cost_per_liter7)

    return tuple(results)

def prep_contrails_data(base, rerouting, noCC):
    prep_data = np.array(([[base[2,0,:],
                            base[2,1,:]],
                           [base[1,0,:],
                            base[1,1,:]],
                           [rerouting[1,0,:],
                            rerouting[1,1,:]],
                           [noCC[1,0,:],
                            noCC[1,1,:]]]))
    return prep_data


def prep_data_bardata(np1, np2, np3=None, np4=None, np5=None, np_bau=None):
    tot_scenarios = 2
    if np3 is not None:
        tot_scenarios += 1
    if np4 is not None:
        tot_scenarios += 1
    if np5 is not None:
        tot_scenarios += 1
    if np_bau is not None:
        tot_scenarios += 1
    prepped_data = np.zeros((tot_scenarios, 2, 41))
    idx = 0
    if np_bau is not None:
        prepped_data[idx, 0, :] = np_bau[0, :]
        prepped_data[idx, 1, :] = np_bau[1, :]
        idx += 1
    prepped_data[idx, 0, :] = np1[0, :]
    prepped_data[idx, 1, :] = np1[1, :]
    idx += 1
    prepped_data[idx, 0, :] = np2[0, :]
    prepped_data[idx, 1, :] = np2[1, :]
    idx += 1
    if np3 is not None:
        prepped_data[idx, 0, :] = np3[0, :]
        prepped_data[idx, 1, :] = np3[1, :]
        idx += 1
    if np4 is not None:
        prepped_data[idx, 0, :] = np4[0, :]
        prepped_data[idx, 1, :] = np4[1, :]
        idx += 1
    if np5 is not None:
        prepped_data[idx, 0, :] = np5[0, :]
        prepped_data[idx, 1, :] = np5[1, :]
        idx += 1
    return prepped_data


def export_prepped_data_to_csv(prepped_data, filename, scenario_names, BAU = False):
    """
    Export prepped data to a CSV file.

    Parameters:
    - prepped_data: The prepped data array.
    - filename: The name of the CSV file.
    - scenario_names: List of scenario names corresponding to the prepped data.
    """
    # Define years from 2020 to 2060
    years = np.arange(2020, 2061)

    # Define the column names
    columns = ['Year'] + scenario_names

    # Initialize a DataFrame with the years
    df = pd.DataFrame({'Year': years})

    # Populate the DataFrame with data from prepped_data
    for i, scenario in enumerate(scenario_names):
        if i == 0 and BAU is True:  # Business as Usual scenario
            df[scenario] = prepped_data[i, 1, :]
        else:  # Other scenarios
            df[f'DACCU {scenario}'] = prepped_data[i, 0, :]  # DACCU
            df[f'DACCS {scenario}'] = prepped_data[i, 1, :]  # DACCS

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

import numpy as np
import pandas as pd

def export_prepped_data_to_excel(prepped_data_dict, filename, scenario_names, BAU=False, unit=None):
    """
    Export prepped data to an Excel file with each dataset as a different sheet.

    Parameters:
    - prepped_data_dict: A dictionary where keys are sheet names and values are prepped data arrays.
    - filename: The name of the Excel file.
    - scenario_names: List of scenario names corresponding to the prepped data.
    - BAU: A boolean indicating if the first scenario is 'Business as Usual'.
    - unit: The unit to add as a column in the DataFrame, if any.
    """
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Iterate through the dictionary and save each prepped data to a separate sheet
        for sheet_name, prepped_data in prepped_data_dict.items():
            # Define years from 2020 to 2060
            years = np.arange(2020, 2061)

            # Initialize a DataFrame with the years
            df = pd.DataFrame({'Year': years})

            # Populate the DataFrame with data from prepped_data
            for i, scenario in enumerate(scenario_names):
                if scenario == 'Baseline':
                    continue  # Skip 'Baseline' scenario
                elif i == 0 and BAU:  # Business as Usual scenario
                    df[scenario] = prepped_data[i, 0, :]
                else:  # Other scenarios
                    df[f'DACCU {scenario}'] = prepped_data[i, 0, :]  # DACCU
                    df[f'DACCS {scenario}'] = prepped_data[i, 1, :]  # DACCS

            if unit is not None:
                df['Unit'] = unit

            # Write the DataFrame to the Excel sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def run_sensitivity_analysis_to_find_zero(df_input, df_emissions_input, growth_rate, efficiency_increase,
                                          learning_rate_h2, learning_rate_co, learning_rate_dac, electricity_cost,
                                          ff_market_cost, co2_transport_storage_cost, dac_q0_2020, dac_c0_2020,
                                          h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020, jetfuel_allocation_share,
                                          cc_efficacy, var_para_1, var1, what, reference, scenario="Baseline",
                                          configuration_PtL='PEM+El.CO2+FT', electrolysis_efficiency_increase=True):
    """
    Finds the optimal value of the specified variable that results in the minimum absolute difference based on specified conditions.

    Parameters:
    df_input (pd.DataFrame): Input DataFrame.
    df_emissions_input (pd.DataFrame): Emissions input DataFrame.
    growth_rate (float): Growth rate.
    efficiency_increase (float): Efficiency increase.
    learning_rate_h2 (float): Learning rate for H2.
    learning_rate_co (float): Learning rate for CO.
    learning_rate_dac (float): Learning rate for DAC.
    electricity_cost (float): Initial electricity cost.
    ff_market_cost (float): Fossil fuel market cost.
    co2_transport_storage_cost (float): CO2 transport storage cost.
    dac_q0_2020 (float): Initial DAC quantity in 2020.
    dac_c0_2020 (float): Initial DAC cost in 2020.
    h2_q0_2020 (float): Initial H2 quantity in 2020.
    h2_c0_2020 (float): Initial H2 cost in 2020.
    co_q0_2020 (float): Initial CO quantity in 2020.
    co_c0_2020 (float): Initial CO cost in 2020.
    jetfuel_allocation_share (float): Jetfuel allocation share.
    cc_efficacy (float): Carbon capture efficacy.
    var_para_1 (str): Parameter to vary (e.g., "electricity cost", "learning rate").
    var1 (array): Array of variation values.
    what (str): Description of what is being analyzed.
    reference (str): Reference for comparison (e.g., "DACCU").
    scenario (str): Scenario to run (default is "Baseline").
    configuration_PtL (str): Configuration for PtL (default is 'PEM+El.CO2+FT').
    electrolysis_efficiency_increase (bool): Flag for electrolysis efficiency increase (default is True).

    Returns:
    float: Optimal value that results in the minimum absolute difference based on specified conditions, or 'NA' if not found.
    float: The minimized absolute difference.
    """

    def objective(value, var_para):
        # Set default values
        ec = electricity_cost
        lr_h2 = learning_rate_h2
        lr_co = learning_rate_co
        lr_dac = learning_rate_dac
        gr = growth_rate
        ff_cost = ff_market_cost
        dac_cost = dac_c0_2020
        eff_increase = efficiency_increase
        cc_eff = cc_efficacy

        if var_para == "electricity cost":
            ec = value
        elif var_para == "learning rate":
            lr_h2 = lr_co = lr_dac = value
        elif var_para == "learning rate H2":
            lr_h2 = value
        elif var_para == "learning rate CO":
            lr_co = value
        elif var_para == "learning rate DAC":
            lr_dac = value
        elif var_para == "growth rate":
            gr = value
        elif var_para == "fossil fuel cost":
            ff_cost = value
        elif var_para == "DAC initial cost":
            dac_cost = value
        elif var_para == "fuel efficiency":
            eff_increase = value
        elif var_para == "cc efficacy":
            cc_eff = value
        else:
            raise ValueError("Invalid parameter to vary.")

        try:
            finalcost_baseline = run_model_II(df_input, df_emissions_input, gr, eff_increase, lr_h2,
                                              lr_co, lr_dac, ec, ff_cost, co2_transport_storage_cost,
                                              dac_q0_2020, dac_cost, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020,
                                              jetfuel_allocation_share, cc_eff, configuration_PtL,
                                              electrolysis_efficiency_increase, scenario="Baseline")[1]

            finalcost_neutrality = run_model_II(df_input, df_emissions_input, gr, eff_increase, lr_h2,
                                                lr_co, lr_dac, ec, ff_cost, co2_transport_storage_cost,
                                                dac_q0_2020, dac_cost, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020,
                                                jetfuel_allocation_share, cc_eff, configuration_PtL,
                                                electrolysis_efficiency_increase, scenario="Neutrality")[0]

            # Debug: Check specific indices
            if finalcost_baseline.shape[0] <= 31:
                print("Index 31 is out of bounds for finalcost_baseline")
                return np.inf

            if finalcost_neutrality.shape[2] <= 31:
                print("Index 31 is out of bounds for finalcost_neutrality")
                return np.inf

            if reference == 'DACCS':
                if what == 'BAU':
                    if scenario == 'carbon neutrality':
                        return abs(finalcost_baseline[31] - finalcost_neutrality[2, 1, 31])
                    elif scenario == 'climate neutrality':
                        return abs(finalcost_baseline[31] - finalcost_neutrality[1, 1, 31])
                elif what == 'DACCU':
                    if scenario == 'carbon neutrality':
                        return abs(finalcost_neutrality[2, 0, 31] - finalcost_neutrality[2, 1, 31])
                    elif scenario == 'climate neutrality':
                        return abs(finalcost_neutrality[2, 0, 31] - finalcost_neutrality[1, 1, 31])
            elif reference == 'DACCU':
                if what == 'BAU':
                    if scenario == 'carbon neutrality':
                        return abs(finalcost_baseline[31] - finalcost_neutrality[2, 0, 31])
                    elif scenario == 'climate neutrality':
                        return abs(finalcost_baseline[31] - finalcost_neutrality[1, 0, 31])
                elif what == 'DACCS':
                    if scenario == 'carbon neutrality':
                        return abs(finalcost_neutrality[2, 1, 31] - finalcost_neutrality[2, 0, 31])
                    elif scenario == 'climate neutrality':
                        return abs(finalcost_neutrality[2, 1, 31] - finalcost_neutrality[1, 0, 31])

        except ValueError as e:
            print(f"Error during function evaluation: {e}")
            return np.inf

        raise ValueError("Invalid combination of 'reference', 'what', and 'scenario'.")

    # Use minimize_scalar to find the optimal value that minimizes the absolute difference
    result = minimize_scalar(objective, bounds=(var1.min(), var1.max()), args=(var_para_1,), method='bounded')

    if result.success:
        optimal_value = result.x
        minimized_difference = result.fun
    else:
        optimal_value = 'NA'
        minimized_difference = 'NA'

    return optimal_value, minimized_difference

def run_sensitivity_analysis_all_variables(df_input, df_emissions_input, growth_rate, efficiency_increase,
                                           learning_rate_h2, learning_rate_co, learning_rate_dac,
                                           electricity_cost, ff_market_cost, co2_transport_storage_cost,
                                           dac_q0_2020, dac_c0_2020, h2_q0_2020, h2_c0_2020,
                                           co_q0_2020, co_c0_2020, jetfuel_allocation_share, cc_efficacy,
                                           vars_to_analyze, what, reference, scenario="Baseline",
                                           configuration_PtL='PEM+El.CO2+FT', electrolysis_efficiency_increase=True):

    results = pd.DataFrame(
        columns=["Variable", "Optimal Value", "Minimized Difference (€ Trillion)"])

    for var, var_range in vars_to_analyze.items():
        optimal_value, minimized_diff = run_sensitivity_analysis_to_find_zero(
            df_input, df_emissions_input, growth_rate, efficiency_increase,
            learning_rate_h2, learning_rate_co, learning_rate_dac,
            electricity_cost, ff_market_cost, co2_transport_storage_cost,
            dac_q0_2020, dac_c0_2020, h2_q0_2020, h2_c0_2020, co_q0_2020, co_c0_2020,
            jetfuel_allocation_share, cc_efficacy, var, var_range, what, reference, scenario, configuration_PtL,
            electrolysis_efficiency_increase)

        # Append the results to the DataFrame using concat
        new_row = pd.DataFrame({"Variable": [var],
                                "Optimal Value": [optimal_value],
                                "Minimized Difference (€ Trillion)": [
                                    minimized_diff / 10 ** 12 if minimized_diff != 'NA' else 'NA']})

        results = pd.concat([results, new_row], ignore_index=True)

    return results


