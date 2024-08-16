from running_analyses import *

#=======================#
#  PARAMETERS SETTINGS  #
#=======================#
# Scenarios for CO2-neutrality:
SCENARIO = "Carbon neutrality"  # Both, Baseline or Carbon neutrality
CONFIGURATION = 'average'
PROGRESSION_CURVE = 'power'
YEAR = 2050
# Create the palette list with RGB tuples
hex_colors = [ '#3277C5', '#244D99','#122071', '#FE8E42','#F8B238', '#FA5050', '#808080']
rgb_colors = [hex_to_rgb(hex_code) for hex_code in hex_colors]
PALETTE = "viridis"

SA_type = "Normal" # or "Exploratory"


# Demand growth & fuel use efficiency improvements over time
EFFICIENCY_INCREASE_YEARLY = 0.02 # 2% efficiency increase is the base assumptions
JETFUEL_ALLOCATION_SHARE = (1/1.82)

# Alternative scenarios
DECREASING_DEMAND_GROWTH_RATE = -0.02 # 2% annual decrease in km flown
STAGNATING_DEMAND_GROWTH_RATE = 0 # no change in km flown, only efficiency increases
HISTORIC_DEMAND_GROWTH_RATE = 0.04 # 4% growth rate (as observed historically)
GROWTH_RATE_AVIATION_FUEL_DEMAND = 0.02 # 2% growth rate is the base assumption


#Learning rate
LEARNING_RATE                    =   0.1  # Learning rate of 10%
LEARNING_RATE_DAC                    =   0.12  # Learning rate of DAC (Young et al., 2023; Sievert et al., 2024)
LEARNING_RATE_FT                    =   0.1  # Learning rate of FT (Becattini et al., 2021)
LEARNING_RATE_CO                    = 0.075 # Learning rate for CO with RWGS (Elsernagawy et al., 2020)
LEARNING_RATE_electrolysis          = 0.18 # +/- 0.13 (Schoots, Ferioli, Kramer and van der Zwaan, 2008; Schmidt et al., 2017)
LEARNING_RATE_electrolysis_endogenous = 0.08 # based on my CAPEX future estimates

# Electricity, Fossil fuel and CO2 Transport and storage cots
ELECTRICITY_COST_KWH     =   0.03 # Electricity cost for all years [$/KWh]
FF_MARKET_COST               =   0.6  # Fossil jet fuel for all years [$/litre]
CO2_TRANSPORT_STORAGE_COST    =   20  # Transport and storage cost per ton of CO2 for all years [$/tCO2]

# All capacity initial data from Becattini et al. 2021
DAC_q0_Gt_2020                        = 0.00001              # DAC initial quantity 2020 [GtCO2]
H2_q0_Mt_2020                         = 7                  # H2 production initial quantity 2020 [MtCO2]
CO_q0_Mt_2020                         = 0.002              # CO production initial quantity 2020 [MtCO2]

# Electrolysis operational values
hours_in_year = 24 * 365
lifetime_electrolyser = 20  # Terlouw et al. 2022
lifetime_stack_min = 7  # Terlouw et al. 2022
lifetime_stack_max = 10  # Terlouw et al. 2022

# Initial CAPEX cost
DAC_c0_2020 = 870  # DAC initial cost 2020 # Young et al., 2023
FT_c0_2020 = 108/10**3 # €/kgfuel  # CAPEX of FT from Becattini et al. 2018
# Contrail cirrus efficacy
CC_EFFICACY = 1.

# Price on emissions and other policy parameters
PRICE_CO2 = 100 # €/tCO2
PRICE_CC_IMPACTS = 100  # €/tCO2eq*
SUBSIDY_DACCS = 100 #€/tCO2
SUBSIDY_DACCU = 0.033 #€/kg
EXCESS_ELECTRICITY_COST = 0.003 #€/kWh
EXCESS_ELECTRICITY_COST_NEGATIVE = -0.001 #€/kWh
DEMAND_RATE_CAPPED = -0.001

#=========================================================================
## RUN ANALYSES ##
#=========================================================================

# define outputs:
running_demand = True
running_contrails = True
plot_main_figures = True
running_SA = False
running_optimization = True
running_singlefactor_SA = True
running_policy_SA = False
plotting_SA_figures = True
plotting_SI_figures = False
output_csv = True

# run
run_base_analysis(rgb_colors, YEAR, SCENARIO, CONFIGURATION, PROGRESSION_CURVE,
                      PALETTE, SA_type, EFFICIENCY_INCREASE_YEARLY,
                      JETFUEL_ALLOCATION_SHARE, DECREASING_DEMAND_GROWTH_RATE,
                      STAGNATING_DEMAND_GROWTH_RATE, HISTORIC_DEMAND_GROWTH_RATE,
                      GROWTH_RATE_AVIATION_FUEL_DEMAND,
                      LEARNING_RATE, LEARNING_RATE_DAC, LEARNING_RATE_FT, LEARNING_RATE_CO, LEARNING_RATE_electrolysis,
                      LEARNING_RATE_electrolysis_endogenous, ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                      DAC_q0_Gt_2020, H2_q0_Mt_2020, CO_q0_Mt_2020,  lifetime_electrolyser,
                      lifetime_stack_min, lifetime_stack_max, DAC_c0_2020, FT_c0_2020,
                      CC_EFFICACY, PRICE_CO2, PRICE_CC_IMPACTS, SUBSIDY_DACCS, SUBSIDY_DACCU,
                      EXCESS_ELECTRICITY_COST, EXCESS_ELECTRICITY_COST_NEGATIVE, DEMAND_RATE_CAPPED,
                      running_demand, running_contrails,
                      plot_main_figures, running_SA, running_optimization, running_singlefactor_SA,
                      running_policy_SA, plotting_SA_figures, plotting_SI_figures, output_csv)
