from plots_two_scenarios_presentation_uncertainty import *
from functions_two_scenarios_uncertainty import *


#================================= MODEL RUNS FOR DIFFERENT ANALYSES ======================================

def run_base_analysis(rgb_colors, YEAR = 2050, SCENARIO = 'Carbon neutrality', CONFIGURATION = 'average', PROGRESSION_CURVE = 'power',
                      PALETTE = 'viridis', SA_type = 'Normal', EFFICIENCY_INCREASE_YEARLY = 0.02,
                      JETFUEL_ALLOCATION_SHARE = (1/1.82), DECREASING_DEMAND_GROWTH_RATE = -0.02,
                      STAGNATING_DEMAND_GROWTH_RATE = 0, HISTORIC_DEMAND_GROWTH_RATE = 0.04, GROWTH_RATE_AVIATION_FUEL_DEMAND = 0.2,
                      LEARNING_RATE = 0.1, LEARNING_RATE_DAC = 0.12, LEARNING_RATE_FT = 0.1, LEARNING_RATE_CO = 0.075, LEARNING_RATE_electrolysis = 0.18,
                      LEARNING_RATE_electrolysis_endogenous = 0.08, ELECTRICITY_COST_KWH = 0.03, FF_MARKET_COST = 0.6, CO2_TRANSPORT_STORAGE_COST = 20,
                      DAC_q0_Gt_2020 = 0.00001, H2_q0_Mt_2020 = 7, CO_q0_Mt_2020 = 0.002,  lifetime_electrolyser = 20,
                      lifetime_stack_min = 7, lifetime_stack_max = 10, DAC_c0_2020 = 870, FT_c0_2020 = 108/10**3,
                      CC_EFFICACY = 1, PRICE_CO2 = 100, PRICE_CC_IMPACTS = 100, SUBSIDY_DACCS = 100, SUBSIDY_DACCU = 0.033,
                      EXCESS_ELECTRICITY_COST = 0.003, EXCESS_ELECTRICITY_COST_NEGATIVE = 0.001, DEMAND_RATE_CAPPED = -0.001,
                      running_demand = True, running_contrails = True,
                      plot_main_figures = True, running_SA = False, running_optimization = False, running_singlefactor_SA = False,
                      running_policy_SA = False, plotting_SA_figures = False, plotting_SI_figures = False, output_csv = True):


    if SCENARIO == "Baseline":
        SCENARIO1 = "DACCU \n baseline"
        SCENARIO3 = "DACCS \n baseline"
    elif SCENARIO == "Carbon neutrality":
        SCENARIO1 = "DACCU \n carbon \n neutrality"
        SCENARIO3 = "DACCS \n carbon \n neutrality"
    elif SCENARIO == "Both":
        SCENARIO1 = "DACCU \n baseline"
        SCENARIO3 = "DACCS \n baseline"
        SCENARIO5 = "DACCU \n carbon \n neutrality"
        SCENARIO6 = "DACCS \n carbon \n neutrality"

    hours_in_year = 24 * 365
    if 'PEM' in CONFIGURATION:
        H2_c0_2020_euro_kW = 920  # H2 initial cost 2020 [€/kW] # IRENA, 2021
        electrolysis_efficiency_2020 = 57  # H2 electricity need [kWh/KgH2produced]    % Mean of: Sutter et al., 2019; IRENA, 2020; Schmidt et al., 2015; Yates et al., 2020; Kopp et al., 2017; Alfian et al., 2019
        H2_c0_2020 = H2_c0_2020_euro_kW / hours_in_year / lifetime_electrolyser * electrolysis_efficiency_2020 * (
                    lifetime_electrolyser / lifetime_stack_min) * 10 ** 3
    if 'AEC' in CONFIGURATION:
        H2_c0_2020_euro_kW = 758  # H2 initial cost 2020 [€/kW] # IRENA, 2021
        electrolysis_efficiency_2020 = 53  # H2 electricity need [kWh/KgH2produced]    % Mean of: Rosenthal et al., 2020; Becattini et al., 2021; IRENA, 2020; Schmidt et al., 2017
        H2_c0_2020 = H2_c0_2020_euro_kW / hours_in_year / lifetime_electrolyser * electrolysis_efficiency_2020 * (
                lifetime_electrolyser / lifetime_stack_min) * 10 ** 3
    if 'average' in CONFIGURATION:
        H2_c0_2020_euro_kW = 839  # average H2 initial cost 2020 [€/kW] # IRENA, 2021
        electrolysis_efficiency_2020 = 55  # average H2 electricity need [kWh/KgH2produced]
        H2_c0_2020 = H2_c0_2020_euro_kW / hours_in_year / lifetime_electrolyser * electrolysis_efficiency_2020 * (
                lifetime_electrolyser / lifetime_stack_min) * 10 ** 3
    if 'RWGS' in CONFIGURATION:
        KGCO_KG_FUEL = 1.969  # kg CO involved in RWGS - Mean of : Becattini et al., 2021; Kalavasta et al., 2018
        CO_c0_2020_euro_kgfuel = 0.0098  # CO initial cost [€/kg fuel] - average of Terwel and Kerkhoven, 2018, Zang ert al., 2021
        CO_c0_2020 = CO_c0_2020_euro_kgfuel / KGCO_KG_FUEL * 10 ** 3  # CO initial cost 2020 [€/tCO]
    if 'El.CO2' in CONFIGURATION:
        CO_c0_2020_euro_kgfuel = 0.83  # CO initial cost 2020 [€/tCO] - Mean of: Shin et al., 2021, Jouny et al., 2018
        KGCO_KG_FUEL = 1.969  # kg CO involved in RWGS - Mean of : Becattini et al., 2021; Kalavasta et al., 2018
        CO_c0_2020 = CO_c0_2020_euro_kgfuel / KGCO_KG_FUEL * 10 ** 3  # CO initial cost 2020 [€/tCO]
    if 'average' in CONFIGURATION:
        CO_c0_2020_euro_kgfuel = 0.0464  # Average cost for CO per kg fuel
        KGCO_KG_FUEL = 1.969  # kg CO involved in RWGS - Mean of : Becattini et al., 2021; Kalavasta et al., 2018
        CO_c0_2020 = CO_c0_2020_euro_kgfuel / KGCO_KG_FUEL * 10 ** 3  # CO initial cost 2020 [€/tCO]

    # Importing necessary input files
    df_input = pd.read_csv('base_input.csv')

    # change progression curve if needed
    if PROGRESSION_CURVE == 'power':
        # Number of total time steps
        total_time_steps = YEAR-2020+(2061-YEAR)
        # Time steps for the first part of the curve
        initial_time_steps = np.linspace(0, 1, YEAR-2020+1)
        # Power function (e.g., cubic) for the initial growth
        progression_curve = initial_time_steps ** 3
        # Ensure the value reaches exactly 1 at the 31st time step
        progression_curve[-1] = 1.0
        # Append 1s for the remaining time steps
        progression_curve = np.append(progression_curve, np.ones(total_time_steps - len(initial_time_steps)))
        # Replace the 'PROGRESSION_CURVE' column in df_input with the new progression curve
        df_input.loc[:, 'PROGRESSION_CURVE'] = progression_curve


    df_emissions_input = pd.read_csv('Input_emissions_indices_Lee.csv')
    ERF_2018, aviation_2018, emissions_2018, ERF_factors = import_lee()
    ERF_factors_uncertain = calculate_ERF_uncertainty(df_emissions_input, ERF_factors)

    # make aviation demand (fuels = EJ) and in distance flown
    baseDemand_EJ, baseDemand_km = make_base_demand_EJ_KM_II(df_input, GROWTH_RATE_AVIATION_FUEL_DEMAND,
                                                             EFFICIENCY_INCREASE_YEARLY)

    # calculate volumes of DACCUs and FF both in EJ and Tg
    DACCU_volumes_EJ, FF_volumes_EJ, DACCU_volumes_Tg, FF_volumes_Tg = make_DACCU_FF_EJ_Tg_II(df_input, baseDemand_EJ)

    # caclulate distance flown by DACCUs and FF and in Tg fuel used from 1990
    FF_km_1990, DACCU_km_1990, FF_Tg_1990, DACCU_Tg_1990 = make_historic_demand_II(df_input, df_emissions_input,
                                                                                   baseDemand_EJ,
                                                                                   baseDemand_km)

    # calculate DAC (GtCO2 and MWh), H2 (Mt and MWh), CO (Mt), FT (MWh) and electricity needed to produce DACCUs
    DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt, CO_DACCU_MWh, FT_DACCU_MWh, \
        DAC_Diesel_Gt, DAC_Diesel_MWh, H2_Diesel_Mt, H2_Diesel_MWh, CO_Diesel_Mt, CO_Diesel_MWh, FT_Diesel_MWh, \
        CO_DACCU_MWh_heat, DAC_DACCU_MWh_heat, CO_Diesel_MWh_heat, DAC_Diesel_MWh_heat = make_DACCU_need_DAC_H2_CO_FT_electricity_II(
        df_input, DACCU_volumes_Tg, JETFUEL_ALLOCATION_SHARE,
        configuration=CONFIGURATION, efficiency_increase=True)

    # =================== LCA part with Uncertainty =========================================

    emissions_Tg_ff, emissions_Tg_DACCU, emissions_Tg_total, ERF_ff, ERF_ff_min, ERF_ff_max, \
        ERF_DACCU, ERF_DACCU_min, ERF_DACCU_max, \
        ERF_total, ERF_total_min, ERF_total_max = make_emissions_and_ERF(df_input, df_emissions_input, FF_km_1990,
                                                                         DACCU_km_1990, FF_Tg_1990, DACCU_Tg_1990,
                                                                         scenario=SCENARIO,
                                                                         erf_uncertain=ERF_factors_uncertain,
                                                                         uncertainty_daccu=True)

    flying_CO2_emissions, flying_CO2_abated, flying_nonCO2_emissions, flying_nonCO2_emissions_min, \
        flying_nonCO2_emissions_max, flying_nonCO2_abated, flying_nonCO2_abated_min, flying_nonCO2_abated_max = \
        make_emissions_CO2equivalent_star(df_input, ERF_total, emissions_Tg_total, ERF_total_min, ERF_total_max)

    # make well-to-tank emissions (driven by FF)
    yearlyWTT = make_WTT_emissions_II(df_input, FF_volumes_EJ)

    # calculate material and electricity indirect emissions in DACCU production
    DAC_DACCU_MaterialFootprint, DAC_DACCU_ElectricityFootprint, H2_DACCU_MaterialFootprint, \
        H2_DACCU_ElectricityFootprint, CO_DACCU_MaterialFootprint, CO_DACCU_ElectricityFootprint, \
        FT_DACCU_ElectricityFootprint, total_DACCU_MaterialFootprint, total_DACCU_ElectricitryFootprint, total_DACCU_Footprint = \
        make_DACCU_indirect_emissions_II(df_input, DAC_DACCU_Gt, DAC_DACCU_MWh, H2_DACCU_Mt, H2_DACCU_MWh, CO_DACCU_Mt,
                                         CO_DACCU_MWh, FT_DACCU_MWh)

    # calculate material and electricity indirect emissions in DACCS utilization
    DAC_CDR_CO2_Gt, DAC_CDR_CO2_MWh, DAC_CDR_CO2_MWhth, DAC_CDR_nonCO2_Gt, DAC_CDR_nonCO2_Gt_min, DAC_CDR_nonCO2_Gt_max, \
        DAC_CDR_nonCO2_MWh, DAC_CDR_nonCO2_MWh_min, DAC_CDR_nonCO2_MWh_max, \
        DAC_CDR_nonCO2_MWhth, DAC_CDR_nonCO2_MWhth_min, DAC_CDR_nonCO2_MWhth_max, \
        DAC_CDR_CO2_MaterialFootprint, DAC_CDR_CO2_ElectricityFootprint, \
        DAC_CDR_nonCO2_MaterialFootprint, DAC_CDR_nonCO2_MaterialFootprint_min, DAC_CDR_nonCO2_MaterialFootprint_max, \
        DAC_CDR_nonCO2_ElectricityFootprint, DAC_CDR_nonCO2_ElectricityFootprint_min, DAC_CDR_nonCO2_ElectricityFootprint_max, \
        total_DAC_CDR_Footprint, total_DAC_CDR_Footprint_min, total_DAC_CDR_Footprint_max = \
        make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated, flying_nonCO2_abated,
                                                               flying_nonCO2_abated_MIN=flying_nonCO2_abated_min,
                                                               flying_nonCO2_abated_MAX=flying_nonCO2_abated_max)

    # calculate all indirect emissions together (well-to-tank + indirect electricity and material footprint)
    totalIndirectEmissions, totalIndirectEmissions_min, totalIndirectEmissions_max, \
        Delta_totalIndirectEmissions, Delta_totalIndirectEmissions_min, Delta_totalIndirectEmissions_max, \
        BAU_EmissionsGt, BAU_EmissionsGt_min, BAU_EmissionsGt_max, \
        totalNetEmissions, totalNetEmissions_min, totalNetEmissions_max = \
        make_indirect_delta_net_emissions_II(yearlyWTT, total_DACCU_Footprint, total_DAC_CDR_Footprint,
                                             flying_CO2_emissions,
                                             flying_nonCO2_emissions, flying_CO2_abated, flying_nonCO2_abated,
                                             total_DAC_CDR_Footprint_min, total_DAC_CDR_Footprint_max,
                                             flying_nonCO2_emissions_min, flying_nonCO2_emissions_max,
                                             flying_nonCO2_abated_min, flying_nonCO2_abated_max
                                             )

    total_abated_emissions = BAU_EmissionsGt - totalNetEmissions
    total_abated_emissions_min = BAU_EmissionsGt_min - totalNetEmissions_min
    total_abated_emissions_max = BAU_EmissionsGt_max - totalNetEmissions_max

    total_absolute_emissions = totalIndirectEmissions + flying_CO2_emissions + flying_nonCO2_emissions
    total_absolute_emissions_min = totalIndirectEmissions_min + flying_CO2_emissions + flying_nonCO2_emissions_min
    total_absolute_emissions_max = totalIndirectEmissions_max + flying_CO2_emissions + flying_nonCO2_emissions_max

    # =============== COST Calculations =================================
    # make yearly DAC costs based on learning
    totalYearlyDAC_need, totalYearlyDAC_need_min, totalYearlyDAC_need_max, \
        total_DAC_InstalledCapacity, total_DAC_InstalledCapacity_min, total_DAC_InstalledCapacity_max, \
        yearlyAddedCapacityDAC, yearlyAddedCapacityDAC_min, yearlyAddedCapacityDAC_max, \
        cost_DAC_ct, cost_DAC_ct_min, cost_DAC_ct_max = \
        make_learning_curve_DAC_II(LEARNING_RATE_DAC, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt, DAC_CDR_CO2_Gt,
                                   DAC_CDR_nonCO2_Gt,
                                   Delta_totalIndirectEmissions, DAC_CDR_nonCO2_Gt_min,
                                   DAC_CDR_nonCO2_Gt_max, Delta_totalIndirectEmissions_min,
                                   Delta_totalIndirectEmissions_max)

    # make yearly H2 and CO costs based on learning
    totalYearlyH2_need, total_H2_InstalledCapacity, yearlyAddedCapacityH2, yearlyH2_need_increase, cost_H2_ct, \
        totalYearlyCO_need, total_CO_InstalledCapacity, yearlyAddedCapacityCO, yearlyCO_need_increase, cost_CO_ct = \
        make_learning_curve_H2_CO_II(LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, H2_q0_Mt_2020, H2_c0_2020,
                                     H2_DACCU_Mt, CO_q0_Mt_2020, CO_c0_2020, CO_DACCU_Mt)

    # calculate cost of DAC
    total_yearly_DAC_Cost, total_yearly_DAC_Cost_min, total_yearly_DAC_Cost_max, yearly_DAC_DACCU_Cost, yearly_DAC_CDR_CO2_Cost, \
        yearly_DAC_CDR_nonCO2_Cost, yearly_DAC_CDR_nonCO2_Cost_min, yearly_DAC_CDR_nonCO2_Cost_max, \
        yearly_DAC_DeltaEmissions_Cost, yearly_DAC_DeltaEmissions_Cost_min, yearly_DAC_DeltaEmissions_Cost_max = calculate_DAC_costs_II(
        cost_DAC_ct, yearlyAddedCapacityDAC, totalYearlyDAC_need,
        DAC_DACCU_Gt, DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt,
        Delta_totalIndirectEmissions, cost_DAC_ct_MIN=cost_DAC_ct_min, cost_DAC_ct_MAX=cost_DAC_ct_max,
        yearlyAddedCapacityDAC_MIN=yearlyAddedCapacityDAC_min, yearlyAddedCapacityDAC_MAX=yearlyAddedCapacityDAC_max,
        totalYearlyDAC_need_MIN=totalYearlyDAC_need_min,
        totalYearlyDAC_need_MAX=totalYearlyDAC_need_max, DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min,
        DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max,
        Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min,
        Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max)

    # calculate cost od H2 and CO
    total_yearly_H2_Cost, total_yearly_CO_Cost = calculate_H2_CO_costs_II(cost_H2_ct, yearlyH2_need_increase,
                                                                          cost_CO_ct, yearlyCO_need_increase)

    # calculate final yearly costs including electricity, fossil fuel, CO2 transport and storage, etc.
    finalcost, finalcost_min, finalcost_max, finalcost_BAU, \
        finalcost_electricity, finalcost_electricity_min, finalcost_electricity_max, \
        finalcost_transport_storageCO2, finalcost_transport_storageCO2_min, finalcost_transport_storageCO2_max, \
        finalcost_fossilfuel, finalcost_heat, finalcost_heat_min, finalcost_heat_max, total_DACCU_electricity_cost, \
        total_DACCS_electricity_cost, total_DACCS_electricity_cost_min, total_DACCS_electricity_cost_max, \
        total_DACCU_production_cost, total_DACCS_cost, total_DACCS_cost_min, total_DACCS_cost_max, \
        total_DACCU_heat_cost, total_DACCS_heat_cost, total_DACCS_heat_cost_min, total_DACCS_heat_cost_max = \
        calculate_final_cost_II(ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_DACCU_MWh,
                                H2_DACCU_MWh,
                                CO_DACCU_MWh, FT_DACCU_MWh, DAC_CDR_CO2_MWh, DAC_CDR_nonCO2_MWh,
                                Delta_totalIndirectEmissions,
                                DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt, FF_volumes_EJ, yearly_DAC_DACCU_Cost,
                                total_yearly_H2_Cost,
                                total_yearly_CO_Cost, total_yearly_DAC_Cost, df_input, CO_DACCU_MWh_heat,
                                DAC_DACCU_MWh_heat, DAC_CDR_CO2_MWhth,
                                DAC_CDR_nonCO2_MWhth,
                                DAC_CDR_nonCO2_MWh_MIN=DAC_CDR_nonCO2_MWh_min,
                                DAC_CDR_nonCO2_MWh_MAX=DAC_CDR_nonCO2_MWh_max,
                                Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min,
                                Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max,
                                DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min,
                                DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max,
                                total_yearly_DAC_COST_MIN=total_yearly_DAC_Cost_min,
                                total_yearly_DAC_COST_MAX=total_yearly_DAC_Cost_max,
                                DAC_CDR_nonCO2_MWhth_MIN=DAC_CDR_nonCO2_MWh_min,
                                DAC_CDR_nonCO2_MWhth_MAX=DAC_CDR_nonCO2_MWh_max)

    total_DACCU_CAPEX_cost = (total_DACCU_production_cost - total_DACCU_electricity_cost - total_DACCU_heat_cost)
    total_DACCS_CAPEX_cost = (yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost + yearly_DAC_DeltaEmissions_Cost)
    total_DACCS_CAPEX_cost_min = yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_min + yearly_DAC_DeltaEmissions_Cost_min
    total_DACCS_CAPEX_cost_max = yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_max + yearly_DAC_DeltaEmissions_Cost_max

    # ====================== simple calculations for additional visualizations =========================
    total_cost_per_CO2eq = finalcost / (totalNetEmissions * 10 ** 12)
    DAC_total_Gt = DAC_DACCU_Gt + DAC_CDR_CO2_Gt + DAC_CDR_nonCO2_Gt
    DAC_yearly_cost_per_ton = total_yearly_DAC_Cost / (DAC_total_Gt * 10 ** 9)

    cost_by_demand = finalcost / (baseDemand_km * 10 ** 6)
    cost_by_demand_BAU = finalcost_BAU / (baseDemand_km * 10 ** 6)

    LN_NY_distance = 5570  # km
    LN_NY_number_passengers = np.average((296, 317, 392,
                                          269)) * 0.8  # assuming 80% occupancy with an average of Boeing 787-9 Dreamliner, Boeing 777-200LR, Boeing 777-300LR, and Boeing 767-300
    LN_Perth_distance = 14500
    LN_Perth_number_passengers = 296 * 0.8  # assuming 80% occupancy with Boeing 787-9 Dreamliner
    LN_Berlin_distance = 964
    LN_Berlin_number_passengers = 178 * 0.8  # assuming 80% occupancy with Boeing 737-900

    cost_fuels_per_passenger_LN_NY_BAU = cost_by_demand_BAU * LN_NY_distance / LN_NY_number_passengers
    cost_neutrality_per_passenger_LN_NY = cost_by_demand * LN_NY_distance / LN_NY_number_passengers

    cost_fuels_per_passenger_LN_Perth_BAU = cost_by_demand_BAU * LN_Perth_distance / LN_Perth_number_passengers
    cost_neutrality_per_passenger_LN_Perth = cost_by_demand * LN_Perth_distance / LN_Perth_number_passengers

    cost_fuels_per_passenger_LN_Berlin_BAU = cost_by_demand_BAU * LN_Berlin_distance / LN_Berlin_number_passengers
    cost_neutrality_per_passenger_LN_Berlin = cost_by_demand * LN_Berlin_distance / LN_Berlin_number_passengers

    reference_2023_cost_LN_NY = np.average((220, 380))
    reference_2023_cost_LN_Berlin = np.average((30, 100))
    reference_2023_cost_LN_Perth = np.average((1300, 1700))

    fuel_share_2023_LN_NY_BAU = cost_fuels_per_passenger_LN_NY_BAU[2023 - 2020] / reference_2023_cost_LN_NY
    fuel_share_2023_LN_Berlin_BAU = cost_fuels_per_passenger_LN_Berlin_BAU[2023 - 2020] / reference_2023_cost_LN_Berlin
    fuel_share_2023_LN_Perth_BAU = cost_fuels_per_passenger_LN_Perth_BAU[2023 - 2020] / reference_2023_cost_LN_Perth

    future_flight_price_LN_NY_BAU = cost_fuels_per_passenger_LN_NY_BAU / fuel_share_2023_LN_NY_BAU
    future_flight_price_LN_Berlin_BAU = cost_fuels_per_passenger_LN_Berlin_BAU / fuel_share_2023_LN_Berlin_BAU
    future_flight_price_LN_Perth_BAU = cost_fuels_per_passenger_LN_Perth_BAU / fuel_share_2023_LN_Perth_BAU

    future_neutrality_flight_price_LN_NY = (
                                                       1 - fuel_share_2023_LN_NY_BAU) * future_flight_price_LN_NY_BAU + cost_neutrality_per_passenger_LN_NY
    future_neutrality_flight_price_LN_Berlin = (
                                                           1 - fuel_share_2023_LN_Berlin_BAU) * future_flight_price_LN_Berlin_BAU + cost_neutrality_per_passenger_LN_Berlin
    future_neutrality_flight_price_LN_Perth = (
                                                          1 - fuel_share_2023_LN_Perth_BAU) * future_flight_price_LN_Perth_BAU + cost_neutrality_per_passenger_LN_Perth

    increase_neutrality_flight_price_LN_NY = (
                                                         future_neutrality_flight_price_LN_NY - future_flight_price_LN_NY_BAU) / future_flight_price_LN_NY_BAU
    increase_neutrality_flight_price_LN_Berlin = (
                                                             future_neutrality_flight_price_LN_Berlin - future_flight_price_LN_Berlin_BAU) / future_flight_price_LN_Berlin_BAU
    increase_neutrality_flight_price_LN_Perth = (
                                                            future_neutrality_flight_price_LN_Perth - future_flight_price_LN_Perth_BAU) / future_flight_price_LN_Perth_BAU

    # cost per liter fuel
    cost_per_liter_DACCU_CAPEX, cost_per_liter_DACCS_CAPEX, \
        cost_per_liter_DACCS_CAPEX_min, cost_per_liter_DACCS_CAPEX_max = calc_array_cost_per_liter_one_scenario(total_DACCU_CAPEX_cost, total_DACCS_CAPEX_cost, total_DACCS_CAPEX_cost_min, total_DACCS_CAPEX_cost_min,
                                           FF_volumes_Tg, DACCU_volumes_Tg)

    cost_per_liter_electricity, cost_per_liter_fossilfuel, cost_per_liter_transport_storageCO2, cost_per_liter_heat = \
        calc_array_cost_per_liter_one_scenario(finalcost_electricity, finalcost_fossilfuel, finalcost_transport_storageCO2,
                                               finalcost_heat, FF_volumes_Tg, DACCU_volumes_Tg)
    cost_per_liter_electricity_min, cost_per_liter_electricity_max, \
        cost_per_liter_transport_storageCO2_min, cost_per_liter_transport_storageCO2_max, \
        cost_per_liter_heat_min, cost_per_liter_heat_max = calc_array_cost_per_liter_one_scenario(finalcost_electricity_min, finalcost_electricity_max,
                                               finalcost_transport_storageCO2_min, finalcost_transport_storageCO2_max, FF_volumes_Tg, DACCU_volumes_Tg,
                                               cost5= finalcost_heat_min, cost6 = finalcost_heat_max)



    if running_demand is True:

        # make aviation demand (fuels = EJ) and in distance flown - three scenarios of demand
        histgrowthDemand_EJ, histgrowthDemand_km = make_base_demand_EJ_KM_II(df_input, HISTORIC_DEMAND_GROWTH_RATE,
                                                                             EFFICIENCY_INCREASE_YEARLY)
        decreasingDemand_EJ, decreasingDemand_km = make_base_demand_EJ_KM_II(df_input, DECREASING_DEMAND_GROWTH_RATE,
                                                                             EFFICIENCY_INCREASE_YEARLY)
        # make aviation demand (fuels = EJ) and in distance flown
        stagnatingDemand_EJ, stagnatingDemand_km = make_base_demand_EJ_KM_II(df_input, STAGNATING_DEMAND_GROWTH_RATE,
                                                                             EFFICIENCY_INCREASE_YEARLY)

        # calculate volumes of DACCUs and FF both in EJ and Tg
        DACCU_volumes_EJ_decreasing, FF_volumes_EJ_decreasing, DACCU_volumes_Tg_decreasing, FF_volumes_Tg_decreasing = \
            make_DACCU_FF_EJ_Tg_II(df_input, decreasingDemand_EJ)
        DACCU_volumes_EJ_stagnating, FF_volumes_EJ_stagnating, DACCU_volumes_Tg_stagnating, FF_volumes_Tg_stagnating = \
            make_DACCU_FF_EJ_Tg_II(df_input, stagnatingDemand_EJ)
        DACCU_volumes_EJ_histgrowth, FF_volumes_EJ_histgrowth, DACCU_volumes_Tg_histgrowth, FF_volumes_Tg_histgrowth = \
            make_DACCU_FF_EJ_Tg_II(df_input, histgrowthDemand_EJ)


        # caclulate distance flown by DACCUs and FF and in Tg fuel used from 1990
        FF_km_1990_decreasing, DACCU_km_1990_decreasing, FF_Tg_1990_decreasing, DACCU_Tg_1990_decreasing = make_historic_demand_II(
            df_input, df_emissions_input, decreasingDemand_EJ,
            decreasingDemand_km)
        FF_km_1990_stagnating, DACCU_km_1990_stagnating, FF_Tg_1990_stagnating, DACCU_Tg_1990_stagnating = make_historic_demand_II(
            df_input, df_emissions_input, stagnatingDemand_EJ,
            stagnatingDemand_km)
        FF_km_1990_histgrowth, DACCU_km_1990_histgrowth, FF_Tg_1990_histgrowth, DACCU_Tg_1990_histgrowth = make_historic_demand_II(
            df_input, df_emissions_input, histgrowthDemand_EJ,
            histgrowthDemand_km)

        # calculate DAC (GtCO2 and MWh), H2 (Mt and MWh), CO (Mt), FT (MWh) and electricity needed to produce DACCUs
        DAC_DACCU_Gt_decreasing, DAC_DACCU_MWh_decreasing, H2_DACCU_Mt_decreasing, H2_DACCU_MWh_decreasing, CO_DACCU_Mt_decreasing, CO_DACCU_MWh_decreasing, FT_DACCU_MWh_decreasing, \
            DAC_Diesel_Gt_decreasing, DAC_Diesel_MWh_decreasing, H2_Diesel_Mt_decreasing, H2_Diesel_MWh_decreasing, CO_Diesel_Mt_decreasing, CO_Diesel_MWh_decreasing, FT_Diesel_MWh_decreasing, \
            CO_DACCU_MWh_heat_decreasing, DAC_DACCU_MWh_heat_decreasing, CO_Diesel_MWh_heat_decreasing, DAC_Diesel_MWh_heat_decreasing = make_DACCU_need_DAC_H2_CO_FT_electricity_II(
            df_input, DACCU_volumes_Tg_decreasing, JETFUEL_ALLOCATION_SHARE,
            configuration=CONFIGURATION, efficiency_increase=True)
        DAC_DACCU_Gt_stagnating, DAC_DACCU_MWh_stagnating, H2_DACCU_Mt_stagnating, H2_DACCU_MWh_stagnating, CO_DACCU_Mt_stagnating, CO_DACCU_MWh_stagnating, FT_DACCU_MWh_stagnating, \
            DAC_Diesel_Gt_stagnating, DAC_Diesel_MWh_stagnating, H2_Diesel_Mt_stagnating, H2_Diesel_MWh_stagnating, CO_Diesel_Mt_stagnating, CO_Diesel_MWh_stagnating, FT_Diesel_MWh_stagnating, \
            CO_DACCU_MWh_heat_stagnating, DAC_DACCU_MWh_heat_stagnating, CO_Diesel_MWh_heat_stagnating, DAC_Diesel_MWh_heat_stagnating = make_DACCU_need_DAC_H2_CO_FT_electricity_II(
            df_input, DACCU_volumes_Tg_stagnating, JETFUEL_ALLOCATION_SHARE,
            configuration=CONFIGURATION, efficiency_increase=True)
        DAC_DACCU_Gt_histgrowth, DAC_DACCU_MWh_histgrowth, H2_DACCU_Mt_histgrowth, H2_DACCU_MWh_histgrowth, CO_DACCU_Mt_histgrowth, CO_DACCU_MWh_histgrowth, FT_DACCU_MWh_histgrowth, \
            DAC_Diesel_Gt_histgrowth, DAC_Diesel_MWh_histgrowth, H2_Diesel_Mt_histgrowth, H2_Diesel_MWh_histgrowth, CO_Diesel_Mt_histgrowth, CO_Diesel_MWh_histgrowth, FT_Diesel_MWh_histgrowth, \
            CO_DACCU_MWh_heat_histgrowth, DAC_DACCU_MWh_heat_histgrowth, CO_Diesel_MWh_heat_histgrowth, DAC_Diesel_MWh_heat_histgrowth = make_DACCU_need_DAC_H2_CO_FT_electricity_II(
            df_input, DACCU_volumes_Tg_histgrowth, JETFUEL_ALLOCATION_SHARE,
            configuration=CONFIGURATION, efficiency_increase=True)

        # =================== LCA part with Uncertainty =========================================

        emissions_Tg_ff_decreasing, emissions_Tg_DACCU_decreasing, emissions_Tg_total_decreasing, ERF_ff_decreasing, ERF_ff_min_decreasing, ERF_ff_max_decreasing, \
            ERF_DACCU_decreasing, ERF_DACCU_min_decreasing, ERF_DACCU_max_decreasing, \
            ERF_total_decreasing, ERF_total_min_decreasing, ERF_total_max_decreasing = make_emissions_and_ERF(df_input,
                                                                                                              df_emissions_input,
                                                                                                              FF_km_1990_decreasing,
                                                                                                              DACCU_km_1990_decreasing,
                                                                                                              FF_Tg_1990_decreasing,
                                                                                                              DACCU_Tg_1990_decreasing,
                                                                                                              scenario=SCENARIO,
                                                                                                              erf_uncertain=ERF_factors_uncertain,
                                                                                                              uncertainty_daccu=True)
        emissions_Tg_ff_stagnating, emissions_Tg_DACCU_stagnating, emissions_Tg_total_stagnating, ERF_ff_stagnating, ERF_ff_min_stagnating, ERF_ff_max_stagnating, \
            ERF_DACCU_stagnating, ERF_DACCU_min_stagnating, ERF_DACCU_max_stagnating, \
            ERF_total_stagnating, ERF_total_min_stagnating, ERF_total_max_stagnating = make_emissions_and_ERF(df_input,
                                                                                                              df_emissions_input,
                                                                                                              FF_km_1990_stagnating,
                                                                                                              DACCU_km_1990_stagnating,
                                                                                                              FF_Tg_1990_stagnating,
                                                                                                              DACCU_Tg_1990_stagnating,
                                                                                                              scenario=SCENARIO,
                                                                                                              erf_uncertain=ERF_factors_uncertain,
                                                                                                              uncertainty_daccu=True)

        emissions_Tg_ff_histgrowth, emissions_Tg_DACCU_histgrowth, emissions_Tg_total_histgrowth, ERF_ff_histgrowth, ERF_ff_min_histgrowth, ERF_ff_max_histgrowth, \
            ERF_DACCU_histgrowth, ERF_DACCU_min_histgrowth, ERF_DACCU_max_histgrowth, \
            ERF_total_histgrowth, ERF_total_min_histgrowth, ERF_total_max_histgrowth = make_emissions_and_ERF(df_input,
                                                                                                              df_emissions_input,
                                                                                                              FF_km_1990_histgrowth,
                                                                                                              DACCU_km_1990_histgrowth,
                                                                                                              FF_Tg_1990_histgrowth,
                                                                                                              DACCU_Tg_1990_histgrowth,
                                                                                                              scenario=SCENARIO,
                                                                                                              erf_uncertain=ERF_factors_uncertain,
                                                                                                              uncertainty_daccu=True)

        flying_CO2_emissions_decreasing, flying_CO2_abated_decreasing, flying_nonCO2_emissions_decreasing, flying_nonCO2_emissions_min_decreasing, \
            flying_nonCO2_emissions_max_decreasing, flying_nonCO2_abated_decreasing, flying_nonCO2_abated_min_decreasing, flying_nonCO2_abated_max_decreasing = \
            make_emissions_CO2equivalent_star(df_input, ERF_total_decreasing, emissions_Tg_total_decreasing,
                                              ERF_total_min_decreasing, ERF_total_max_decreasing)
        flying_CO2_emissions_stagnating, flying_CO2_abated_stagnating, flying_nonCO2_emissions_stagnating, flying_nonCO2_emissions_min_stagnating, \
            flying_nonCO2_emissions_max_stagnating, flying_nonCO2_abated_stagnating, flying_nonCO2_abated_min_stagnating, flying_nonCO2_abated_max_stagnating = \
            make_emissions_CO2equivalent_star(df_input, ERF_total_stagnating, emissions_Tg_total_stagnating,
                                              ERF_total_min_stagnating, ERF_total_max_stagnating)
        flying_CO2_emissions_histgrowth, flying_CO2_abated_histgrowth, flying_nonCO2_emissions_histgrowth, flying_nonCO2_emissions_min_histgrowth, \
            flying_nonCO2_emissions_max_histgrowth, flying_nonCO2_abated_histgrowth, flying_nonCO2_abated_min_histgrowth, flying_nonCO2_abated_max_histgrowth = \
            make_emissions_CO2equivalent_star(df_input, ERF_total_histgrowth, emissions_Tg_total_histgrowth,
                                              ERF_total_min_histgrowth, ERF_total_max_histgrowth)


        # make well-to-tank emissions (driven by FF)
        yearlyWTT_decreasing = make_WTT_emissions_II(df_input, FF_volumes_EJ_decreasing)
        yearlyWTT_stagnating = make_WTT_emissions_II(df_input, FF_volumes_EJ_stagnating)
        yearlyWTT_histgrowth = make_WTT_emissions_II(df_input, FF_volumes_EJ_histgrowth)

        # calculate material and electricity indirect emissions in DACCU production
        DAC_DACCU_MaterialFootprint_decreasing, DAC_DACCU_ElectricityFootprint_decreasing, H2_DACCU_MaterialFootprint_decreasing, \
            H2_DACCU_ElectricityFootprint_decreasing, CO_DACCU_MaterialFootprint_decreasing, CO_DACCU_ElectricityFootprint_decreasing, \
            FT_DACCU_ElectricityFootprint_decreasing, total_DACCU_MaterialFootprint_decreasing, total_DACCU_ElectricitryFootprint_decreasing, total_DACCU_Footprint_decreasing = \
            make_DACCU_indirect_emissions_II(df_input, DAC_DACCU_Gt_decreasing, DAC_DACCU_MWh_decreasing,
                                             H2_DACCU_Mt_decreasing, H2_DACCU_MWh_decreasing, CO_DACCU_Mt_decreasing,
                                             CO_DACCU_MWh_decreasing, FT_DACCU_MWh_decreasing)
        DAC_DACCU_MaterialFootprint_stagnating, DAC_DACCU_ElectricityFootprint_stagnating, H2_DACCU_MaterialFootprint_stagnating, \
            H2_DACCU_ElectricityFootprint_stagnating, CO_DACCU_MaterialFootprint_stagnating, CO_DACCU_ElectricityFootprint_stagnating, \
            FT_DACCU_ElectricityFootprint_stagnating, total_DACCU_MaterialFootprint_stagnating, total_DACCU_ElectricitryFootprint_stagnating, total_DACCU_Footprint_stagnating = \
            make_DACCU_indirect_emissions_II(df_input, DAC_DACCU_Gt_stagnating, DAC_DACCU_MWh_stagnating,
                                             H2_DACCU_Mt_stagnating, H2_DACCU_MWh_stagnating, CO_DACCU_Mt_stagnating,
                                             CO_DACCU_MWh_stagnating, FT_DACCU_MWh_stagnating)
        DAC_DACCU_MaterialFootprint_histgrowth, DAC_DACCU_ElectricityFootprint_histgrowth, H2_DACCU_MaterialFootprint_histgrowth, \
            H2_DACCU_ElectricityFootprint_histgrowth, CO_DACCU_MaterialFootprint_histgrowth, CO_DACCU_ElectricityFootprint_histgrowth, \
            FT_DACCU_ElectricityFootprint_histgrowth, total_DACCU_MaterialFootprint_histgrowth, total_DACCU_ElectricitryFootprint_histgrowth, total_DACCU_Footprint_histgrowth = \
            make_DACCU_indirect_emissions_II(df_input, DAC_DACCU_Gt_histgrowth, DAC_DACCU_MWh_histgrowth,
                                             H2_DACCU_Mt_histgrowth, H2_DACCU_MWh_histgrowth, CO_DACCU_Mt_histgrowth,
                                             CO_DACCU_MWh_histgrowth, FT_DACCU_MWh_histgrowth)

        # calculate material and electricity indirect emissions in DACCS utilization
        DAC_CDR_CO2_Gt_decreasing, DAC_CDR_CO2_MWh_decreasing, DAC_CDR_CO2_MWhth_decreasing, DAC_CDR_nonCO2_Gt_decreasing, DAC_CDR_nonCO2_Gt_min_decreasing, DAC_CDR_nonCO2_Gt_max_decreasing, \
            DAC_CDR_nonCO2_MWh_decreasing, DAC_CDR_nonCO2_MWh_min_decreasing, DAC_CDR_nonCO2_MWh_max_decreasing, \
            DAC_CDR_nonCO2_MWhth_decreasing, DAC_CDR_nonCO2_MWhth_min_decreasing, DAC_CDR_nonCO2_MWhth_max_decreasing, \
            DAC_CDR_CO2_MaterialFootprint_decreasing, DAC_CDR_CO2_ElectricityFootprint_decreasing, \
            DAC_CDR_nonCO2_MaterialFootprint_decreasing, DAC_CDR_nonCO2_MaterialFootprint_min_decreasing, DAC_CDR_nonCO2_MaterialFootprint_max_decreasing, \
            DAC_CDR_nonCO2_ElectricityFootprint_decreasing, DAC_CDR_nonCO2_ElectricityFootprint_min_decreasing, DAC_CDR_nonCO2_ElectricityFootprint_max_decreasing, \
            total_DAC_CDR_Footprint_decreasing, total_DAC_CDR_Footprint_min_decreasing, total_DAC_CDR_Footprint_max_decreasing = \
            make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated_decreasing,
                                                                   flying_nonCO2_abated_decreasing,
                                                                   flying_nonCO2_abated_MIN=flying_nonCO2_abated_min_decreasing,
                                                                   flying_nonCO2_abated_MAX=flying_nonCO2_abated_max_decreasing)
        DAC_CDR_CO2_Gt_stagnating, DAC_CDR_CO2_MWh_stagnating, DAC_CDR_CO2_MWhth_stagnating, DAC_CDR_nonCO2_Gt_stagnating, DAC_CDR_nonCO2_Gt_min_stagnating, DAC_CDR_nonCO2_Gt_max_stagnating, \
            DAC_CDR_nonCO2_MWh_stagnating, DAC_CDR_nonCO2_MWh_min_stagnating, DAC_CDR_nonCO2_MWh_max_stagnating, \
            DAC_CDR_nonCO2_MWhth_stagnating, DAC_CDR_nonCO2_MWhth_min_stagnating, DAC_CDR_nonCO2_MWhth_max_stagnating, \
            DAC_CDR_CO2_MaterialFootprint_stagnating, DAC_CDR_CO2_ElectricityFootprint_stagnating, \
            DAC_CDR_nonCO2_MaterialFootprint_stagnating, DAC_CDR_nonCO2_MaterialFootprint_min_stagnating, DAC_CDR_nonCO2_MaterialFootprint_max_stagnating, \
            DAC_CDR_nonCO2_ElectricityFootprint_stagnating, DAC_CDR_nonCO2_ElectricityFootprint_min_stagnating, DAC_CDR_nonCO2_ElectricityFootprint_max_stagnating, \
            total_DAC_CDR_Footprint_stagnating, total_DAC_CDR_Footprint_min_stagnating, total_DAC_CDR_Footprint_max_stagnating = \
            make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated_stagnating,
                                                                   flying_nonCO2_abated_stagnating,
                                                                   flying_nonCO2_abated_MIN=flying_nonCO2_abated_min_stagnating,
                                                                   flying_nonCO2_abated_MAX=flying_nonCO2_abated_max_stagnating)

        DAC_CDR_CO2_Gt_histgrowth, DAC_CDR_CO2_MWh_histgrowth, DAC_CDR_CO2_MWhth_histgrowth, DAC_CDR_nonCO2_Gt_histgrowth, DAC_CDR_nonCO2_Gt_min_histgrowth, DAC_CDR_nonCO2_Gt_max_histgrowth, \
            DAC_CDR_nonCO2_MWh_histgrowth, DAC_CDR_nonCO2_MWh_min_histgrowth, DAC_CDR_nonCO2_MWh_max_histgrowth, \
            DAC_CDR_nonCO2_MWhth_histgrowth, DAC_CDR_nonCO2_MWhth_min_histgrowth, DAC_CDR_nonCO2_MWhth_max_histgrowth, \
            DAC_CDR_CO2_MaterialFootprint_histgrowth, DAC_CDR_CO2_ElectricityFootprint_histgrowth, \
            DAC_CDR_nonCO2_MaterialFootprint_histgrowth, DAC_CDR_nonCO2_MaterialFootprint_min_histgrowth, DAC_CDR_nonCO2_MaterialFootprint_max_histgrowth, \
            DAC_CDR_nonCO2_ElectricityFootprint_histgrowth, DAC_CDR_nonCO2_ElectricityFootprint_min_histgrowth, DAC_CDR_nonCO2_ElectricityFootprint_max_histgrowth, \
            total_DAC_CDR_Footprint_histgrowth, total_DAC_CDR_Footprint_min_histgrowth, total_DAC_CDR_Footprint_max_histgrowth = \
            make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated_histgrowth,
                                                                   flying_nonCO2_abated_histgrowth,
                                                                   flying_nonCO2_abated_MIN=flying_nonCO2_abated_min_histgrowth,
                                                                   flying_nonCO2_abated_MAX=flying_nonCO2_abated_max_histgrowth)


        # calculate all indirect emissions together (well-to-tank + indirect electricity and material footprint)
        totalIndirectEmissions_decreasing, totalIndirectEmissions_min_decreasing, totalIndirectEmissions_max_decreasing, \
            Delta_totalIndirectEmissions_decreasing, Delta_totalIndirectEmissions_min_decreasing, Delta_totalIndirectEmissions_max_decreasing, \
            BAU_EmissionsGt_decreasing, BAU_EmissionsGt_min_decreasing, BAU_EmissionsGt_max_decreasing, \
            totalNetEmissions_decreasing, totalNetEmissions_min_decreasing, totalNetEmissions_max_decreasing = \
            make_indirect_delta_net_emissions_II(yearlyWTT_decreasing, total_DACCU_Footprint_decreasing,
                                                 total_DAC_CDR_Footprint_decreasing,
                                                 flying_CO2_emissions_decreasing, flying_nonCO2_emissions_decreasing,
                                                 flying_CO2_abated_decreasing, flying_nonCO2_abated_decreasing,
                                                 total_DAC_CDR_Footprint_min_decreasing,
                                                 total_DAC_CDR_Footprint_max_decreasing,
                                                 flying_nonCO2_emissions_min_decreasing,
                                                 flying_nonCO2_emissions_max_decreasing,
                                                 flying_nonCO2_abated_min_decreasing,
                                                 flying_nonCO2_abated_max_decreasing
                                                 )
        # calculate all indirect emissions together (well-to-tank + indirect electricity and material footprint)
        totalIndirectEmissions_stagnating, totalIndirectEmissions_min_stagnating, totalIndirectEmissions_max_stagnating, \
            Delta_totalIndirectEmissions_stagnating, Delta_totalIndirectEmissions_min_stagnating, Delta_totalIndirectEmissions_max_stagnating, \
            BAU_EmissionsGt_stagnating, BAU_EmissionsGt_min_stagnating, BAU_EmissionsGt_max_stagnating, \
            totalNetEmissions_stagnating, totalNetEmissions_min_stagnating, totalNetEmissions_max_stagnating = \
            make_indirect_delta_net_emissions_II(yearlyWTT_stagnating, total_DACCU_Footprint_stagnating,
                                                 total_DAC_CDR_Footprint_stagnating,
                                                 flying_CO2_emissions_stagnating, flying_nonCO2_emissions_stagnating,
                                                 flying_CO2_abated_stagnating, flying_nonCO2_abated_stagnating,
                                                 total_DAC_CDR_Footprint_min_stagnating,
                                                 total_DAC_CDR_Footprint_max_stagnating,
                                                 flying_nonCO2_emissions_min_stagnating,
                                                 flying_nonCO2_emissions_max_stagnating,
                                                 flying_nonCO2_abated_min_stagnating,
                                                 flying_nonCO2_abated_max_stagnating
                                                 )

        # calculate all indirect emissions together (well-to-tank + indirect electricity and material footprint)
        totalIndirectEmissions_histgrowth, totalIndirectEmissions_min_histgrowth, totalIndirectEmissions_max_histgrowth, \
            Delta_totalIndirectEmissions_histgrowth, Delta_totalIndirectEmissions_min_histgrowth, Delta_totalIndirectEmissions_max_histgrowth, \
            BAU_EmissionsGt_histgrowth, BAU_EmissionsGt_min_histgrowth, BAU_EmissionsGt_max_histgrowth, \
            totalNetEmissions_histgrowth, totalNetEmissions_min_histgrowth, totalNetEmissions_max_histgrowth = \
            make_indirect_delta_net_emissions_II(yearlyWTT_histgrowth, total_DACCU_Footprint_histgrowth,
                                                 total_DAC_CDR_Footprint_histgrowth,
                                                 flying_CO2_emissions_histgrowth, flying_nonCO2_emissions_histgrowth,
                                                 flying_CO2_abated_histgrowth, flying_nonCO2_abated_histgrowth,
                                                 total_DAC_CDR_Footprint_min_histgrowth,
                                                 total_DAC_CDR_Footprint_max_histgrowth,
                                                 flying_nonCO2_emissions_min_histgrowth,
                                                 flying_nonCO2_emissions_max_histgrowth,
                                                 flying_nonCO2_abated_min_histgrowth,
                                                 flying_nonCO2_abated_max_histgrowth
                                                 )

        # total emissions - decreasing growth
        total_abated_emissions_decreasing = BAU_EmissionsGt_decreasing - totalNetEmissions_decreasing
        total_abated_emissions_min_decreasing = BAU_EmissionsGt_min_decreasing - totalNetEmissions_min_decreasing
        total_abated_emissions_max_decreasing = BAU_EmissionsGt_max_decreasing - totalNetEmissions_max_decreasing
        total_absolute_emissions_decreasing = totalIndirectEmissions_decreasing + flying_CO2_emissions_decreasing + flying_nonCO2_emissions_decreasing
        total_absolute_emissions_min_decreasing = totalIndirectEmissions_min_decreasing + flying_CO2_emissions_decreasing + flying_nonCO2_emissions_min_decreasing
        total_absolute_emissions_max_decreasing = totalIndirectEmissions_max_decreasing + flying_CO2_emissions_decreasing + flying_nonCO2_emissions_max_decreasing
        # stagnating growth
        total_abated_emissions_stagnating = BAU_EmissionsGt_stagnating - totalNetEmissions_stagnating
        total_abated_emissions_min_stagnating = BAU_EmissionsGt_min_stagnating - totalNetEmissions_min_stagnating
        total_abated_emissions_max_stagnating = BAU_EmissionsGt_max_stagnating - totalNetEmissions_max_stagnating
        total_absolute_emissions_stagnating = totalIndirectEmissions_stagnating + flying_CO2_emissions_stagnating + flying_nonCO2_emissions_stagnating
        total_absolute_emissions_min_stagnating = totalIndirectEmissions_min_stagnating + flying_CO2_emissions_stagnating + flying_nonCO2_emissions_min_stagnating
        total_absolute_emissions_max_stagnating = totalIndirectEmissions_max_stagnating + flying_CO2_emissions_stagnating + flying_nonCO2_emissions_max_stagnating
        # historic growth rate
        total_abated_emissions_histgrowth = BAU_EmissionsGt_histgrowth - totalNetEmissions_histgrowth
        total_abated_emissions_min_histgrowth = BAU_EmissionsGt_min_histgrowth - totalNetEmissions_min_histgrowth
        total_abated_emissions_max_histgrowth = BAU_EmissionsGt_max_histgrowth - totalNetEmissions_max_histgrowth
        total_absolute_emissions_histgrowth = totalIndirectEmissions_histgrowth + flying_CO2_emissions_histgrowth + flying_nonCO2_emissions_histgrowth
        total_absolute_emissions_min_histgrowth = totalIndirectEmissions_min_histgrowth + flying_CO2_emissions_histgrowth + flying_nonCO2_emissions_min_histgrowth
        total_absolute_emissions_max_histgrowth = totalIndirectEmissions_max_histgrowth + flying_CO2_emissions_histgrowth + flying_nonCO2_emissions_max_histgrowth

        # ============================= COST CALCULATIONS - DECREASING DEMAND ==========================
        # =============== COST Calculations =================================
        # make yearly DAC costs based on learning
        totalYearlyDAC_need_decreasing, totalYearlyDAC_need_min_decreasing, totalYearlyDAC_need_max_decreasing, \
            total_DAC_InstalledCapacity_decreasing, total_DAC_InstalledCapacity_min_decreasing, total_DAC_InstalledCapacity_max_decreasing, \
            yearlyAddedCapacityDAC_decreasing, yearlyAddedCapacityDAC_min_decreasing, yearlyAddedCapacityDAC_max_decreasing, \
            cost_DAC_ct_decreasing, cost_DAC_ct_min_decreasing, cost_DAC_ct_max_decreasing = \
            make_learning_curve_DAC_II(LEARNING_RATE_DAC, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt_decreasing,
                                       DAC_CDR_CO2_Gt_decreasing, DAC_CDR_nonCO2_Gt_decreasing,
                                       Delta_totalIndirectEmissions_decreasing, DAC_CDR_nonCO2_Gt_min_decreasing,
                                       DAC_CDR_nonCO2_Gt_max_decreasing, Delta_totalIndirectEmissions_min_decreasing,
                                       Delta_totalIndirectEmissions_max_decreasing)

        # make yearly H2 and CO costs based on learning
        totalYearlyH2_need_decreasing, total_H2_InstalledCapacity_decreasing, yearlyAddedCapacityH2_decreasing, yearlyH2_need_increase_decreasing, cost_H2_ct_decreasing, \
            totalYearlyCO_need_decreasing, total_CO_InstalledCapacity_decreasing, yearlyAddedCapacityCO_decreasing, yearlyCO_need_increase_decreasing, cost_CO_ct_decreasing = \
            make_learning_curve_H2_CO_II(LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, H2_q0_Mt_2020,
                                         H2_c0_2020, H2_DACCU_Mt_decreasing, CO_q0_Mt_2020, CO_c0_2020,
                                         CO_DACCU_Mt_decreasing)

        # calculate cost of DAC
        total_yearly_DAC_Cost_decreasing, total_yearly_DAC_Cost_min_decreasing, total_yearly_DAC_Cost_max_decreasing, yearly_DAC_DACCU_Cost_decreasing, yearly_DAC_CDR_CO2_Cost_decreasing, \
            yearly_DAC_CDR_nonCO2_Cost_decreasing, yearly_DAC_CDR_nonCO2_Cost_min_decreasing, yearly_DAC_CDR_nonCO2_Cost_max_decreasing, \
            yearly_DAC_DeltaEmissions_Cost_decreasing, yearly_DAC_DeltaEmissions_Cost_min_decreasing, yearly_DAC_DeltaEmissions_Cost_max_decreasing = calculate_DAC_costs_II(
            cost_DAC_ct_decreasing, yearlyAddedCapacityDAC_decreasing, totalYearlyDAC_need_decreasing,
            DAC_DACCU_Gt_decreasing, DAC_CDR_CO2_Gt_decreasing, DAC_CDR_nonCO2_Gt_decreasing,
            Delta_totalIndirectEmissions_decreasing, cost_DAC_ct_MIN=cost_DAC_ct_min_decreasing,
            cost_DAC_ct_MAX=cost_DAC_ct_max_decreasing,
            yearlyAddedCapacityDAC_MIN=yearlyAddedCapacityDAC_min_decreasing,
            yearlyAddedCapacityDAC_MAX=yearlyAddedCapacityDAC_max_decreasing,
            totalYearlyDAC_need_MIN=totalYearlyDAC_need_min_decreasing,
            totalYearlyDAC_need_MAX=totalYearlyDAC_need_max_decreasing,
            DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_decreasing,
            DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_decreasing,
            Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_decreasing,
            Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_decreasing)

        # calculate cost of H2 and CO
        total_yearly_H2_Cost_decreasing, total_yearly_CO_Cost_decreasing = calculate_H2_CO_costs_II(
            cost_H2_ct_decreasing, yearlyH2_need_increase_decreasing,
            cost_CO_ct_decreasing, yearlyCO_need_increase_decreasing)

        # calculate final yearly costs including electricity, fossil fuel, CO2 transport and storage, etc.
        finalcost_decreasing, finalcost_min_decreasing, finalcost_max_decreasing, finalcost_BAU_decreasing, \
            finalcost_electricity_decreasing, finalcost_electricity_min_decreasing, finalcost_electricity_max_decreasing, \
            finalcost_transport_storageCO2_decreasing, finalcost_transport_storageCO2_min_decreasing, finalcost_transport_storageCO2_max_decreasing, \
            finalcost_fossilfuel_decreasing, finalcost_heat_decreasing, finalcost_heat_min_decreasing, finalcost_heat_max_decreasing, total_DACCU_electricity_cost_decreasing, \
            total_DACCS_electricity_cost_decreasing, total_DACCS_electricity_cost_min_decreasing, total_DACCS_electricity_cost_max_decreasing, \
            total_DACCU_production_cost_decreasing, total_DACCS_cost_decreasing, total_DACCS_cost_min_decreasing, total_DACCS_cost_max_decreasing, \
            total_DACCU_heat_cost_decreasing, total_DACCS_heat_cost_decreasing, total_DACCS_heat_cost_min_decreasing, total_DACCS_heat_cost_max_decreasing = \
            calculate_final_cost_II(ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                    DAC_DACCU_MWh_decreasing, H2_DACCU_MWh_decreasing,
                                    CO_DACCU_MWh_decreasing, FT_DACCU_MWh_decreasing, DAC_CDR_CO2_MWh_decreasing,
                                    DAC_CDR_nonCO2_MWh_decreasing, Delta_totalIndirectEmissions_decreasing,
                                    DAC_CDR_CO2_Gt_decreasing, DAC_CDR_nonCO2_Gt_decreasing, FF_volumes_EJ_decreasing,
                                    yearly_DAC_DACCU_Cost_decreasing, total_yearly_H2_Cost_decreasing,
                                    total_yearly_CO_Cost_decreasing, total_yearly_DAC_Cost_decreasing, df_input,
                                    CO_DACCU_MWh_heat_decreasing, DAC_DACCU_MWh_heat_decreasing,
                                    DAC_CDR_CO2_MWhth_decreasing,
                                    DAC_CDR_nonCO2_MWhth_decreasing,
                                    DAC_CDR_nonCO2_MWh_MIN=DAC_CDR_nonCO2_MWh_min_decreasing,
                                    DAC_CDR_nonCO2_MWh_MAX=DAC_CDR_nonCO2_MWh_max_decreasing,
                                    Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_decreasing,
                                    Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_decreasing,
                                    DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_decreasing,
                                    DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_decreasing,
                                    total_yearly_DAC_COST_MIN=total_yearly_DAC_Cost_min_decreasing,
                                    total_yearly_DAC_COST_MAX=total_yearly_DAC_Cost_max_decreasing,
                                    DAC_CDR_nonCO2_MWhth_MIN=DAC_CDR_nonCO2_MWh_min_decreasing,
                                    DAC_CDR_nonCO2_MWhth_MAX=DAC_CDR_nonCO2_MWh_max_decreasing)
        # STAGNATING DEMAND
        # =============== COST Calculations =================================
        # make yearly DAC costs based on learning
        # =============== COST Calculations =================================
        # make yearly DAC costs based on learning
        totalYearlyDAC_need_stagnating, totalYearlyDAC_need_min_stagnating, totalYearlyDAC_need_max_stagnating, \
            total_DAC_InstalledCapacity_stagnating, total_DAC_InstalledCapacity_min_stagnating, total_DAC_InstalledCapacity_max_stagnating, \
            yearlyAddedCapacityDAC_stagnating, yearlyAddedCapacityDAC_min_stagnating, yearlyAddedCapacityDAC_max_stagnating, \
            cost_DAC_ct_stagnating, cost_DAC_ct_min_stagnating, cost_DAC_ct_max_stagnating = \
            make_learning_curve_DAC_II(LEARNING_RATE_DAC, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt_stagnating,
                                       DAC_CDR_CO2_Gt_stagnating, DAC_CDR_nonCO2_Gt_stagnating,
                                       Delta_totalIndirectEmissions_stagnating, DAC_CDR_nonCO2_Gt_min_stagnating,
                                       DAC_CDR_nonCO2_Gt_max_stagnating, Delta_totalIndirectEmissions_min_stagnating,
                                       Delta_totalIndirectEmissions_max_stagnating)

        # make yearly H2 and CO costs based on learning
        totalYearlyH2_need_stagnating, total_H2_InstalledCapacity_stagnating, yearlyAddedCapacityH2_stagnating, yearlyH2_need_increase_stagnating, cost_H2_ct_stagnating, \
            totalYearlyCO_need_stagnating, total_CO_InstalledCapacity_stagnating, yearlyAddedCapacityCO_stagnating, yearlyCO_need_increase_stagnating, cost_CO_ct_stagnating = \
            make_learning_curve_H2_CO_II(LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, H2_q0_Mt_2020,
                                         H2_c0_2020, H2_DACCU_Mt_stagnating, CO_q0_Mt_2020, CO_c0_2020,
                                         CO_DACCU_Mt_stagnating)

        # calculate cost of DAC
        total_yearly_DAC_Cost_stagnating, total_yearly_DAC_Cost_min_stagnating, total_yearly_DAC_Cost_max_stagnating, yearly_DAC_DACCU_Cost_stagnating, yearly_DAC_CDR_CO2_Cost_stagnating, \
            yearly_DAC_CDR_nonCO2_Cost_stagnating, yearly_DAC_CDR_nonCO2_Cost_min_stagnating, yearly_DAC_CDR_nonCO2_Cost_max_stagnating, \
            yearly_DAC_DeltaEmissions_Cost_stagnating, yearly_DAC_DeltaEmissions_Cost_min_stagnating, yearly_DAC_DeltaEmissions_Cost_max_stagnating = calculate_DAC_costs_II(
            cost_DAC_ct_stagnating, yearlyAddedCapacityDAC_stagnating, totalYearlyDAC_need_stagnating,
            DAC_DACCU_Gt_stagnating, DAC_CDR_CO2_Gt_stagnating, DAC_CDR_nonCO2_Gt_stagnating,
            Delta_totalIndirectEmissions_stagnating, cost_DAC_ct_MIN=cost_DAC_ct_min_stagnating,
            cost_DAC_ct_MAX=cost_DAC_ct_max_stagnating,
            yearlyAddedCapacityDAC_MIN=yearlyAddedCapacityDAC_min_stagnating,
            yearlyAddedCapacityDAC_MAX=yearlyAddedCapacityDAC_max_stagnating,
            totalYearlyDAC_need_MIN=totalYearlyDAC_need_min_stagnating,
            totalYearlyDAC_need_MAX=totalYearlyDAC_need_max_stagnating,
            DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_stagnating,
            DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_stagnating,
            Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_stagnating,
            Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_stagnating)

        # calculate cost of H2 and CO
        total_yearly_H2_Cost_stagnating, total_yearly_CO_Cost_stagnating = calculate_H2_CO_costs_II(
            cost_H2_ct_stagnating, yearlyH2_need_increase_stagnating,
            cost_CO_ct_stagnating, yearlyCO_need_increase_stagnating)

        # calculate final yearly costs including electricity, fossil fuel, CO2 transport and storage, etc.
        finalcost_stagnating, finalcost_min_stagnating, finalcost_max_stagnating, finalcost_BAU_stagnating, \
            finalcost_electricity_stagnating, finalcost_electricity_min_stagnating, finalcost_electricity_max_stagnating, \
            finalcost_transport_storageCO2_stagnating, finalcost_transport_storageCO2_min_stagnating, finalcost_transport_storageCO2_max_stagnating, \
            finalcost_fossilfuel_stagnating, finalcost_heat_stagnating, finalcost_heat_min_stagnating, finalcost_heat_max_stagnating, total_DACCU_electricity_cost_stagnating, \
            total_DACCS_electricity_cost_stagnating, total_DACCS_electricity_cost_min_stagnating, total_DACCS_electricity_cost_max_stagnating, \
            total_DACCU_production_cost_stagnating, total_DACCS_cost_stagnating, total_DACCS_cost_min_stagnating, total_DACCS_cost_max_stagnating, \
            total_DACCU_heat_cost_stagnating, total_DACCS_heat_cost_stagnating, total_DACCS_heat_cost_min_stagnating, total_DACCS_heat_cost_max_stagnating = \
            calculate_final_cost_II(ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                    DAC_DACCU_MWh_stagnating, H2_DACCU_MWh_stagnating,
                                    CO_DACCU_MWh_stagnating, FT_DACCU_MWh_stagnating, DAC_CDR_CO2_MWh_stagnating,
                                    DAC_CDR_nonCO2_MWh_stagnating, Delta_totalIndirectEmissions_stagnating,
                                    DAC_CDR_CO2_Gt_stagnating, DAC_CDR_nonCO2_Gt_stagnating, FF_volumes_EJ_stagnating,
                                    yearly_DAC_DACCU_Cost_stagnating, total_yearly_H2_Cost_stagnating,
                                    total_yearly_CO_Cost_stagnating, total_yearly_DAC_Cost_stagnating, df_input,
                                    CO_DACCU_MWh_heat_stagnating, DAC_DACCU_MWh_heat_stagnating,
                                    DAC_CDR_CO2_MWhth_stagnating,
                                    DAC_CDR_nonCO2_MWhth_stagnating,
                                    DAC_CDR_nonCO2_MWh_MIN=DAC_CDR_nonCO2_MWh_min_stagnating,
                                    DAC_CDR_nonCO2_MWh_MAX=DAC_CDR_nonCO2_MWh_max_stagnating,
                                    Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_stagnating,
                                    Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_stagnating,
                                    DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_stagnating,
                                    DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_stagnating,
                                    total_yearly_DAC_COST_MIN=total_yearly_DAC_Cost_min_stagnating,
                                    total_yearly_DAC_COST_MAX=total_yearly_DAC_Cost_max_stagnating,
                                    DAC_CDR_nonCO2_MWhth_MIN=DAC_CDR_nonCO2_MWh_min_stagnating,
                                    DAC_CDR_nonCO2_MWhth_MAX=DAC_CDR_nonCO2_MWh_max_stagnating)


        # HISTORIC GROWTH RATES
        # =============== COST Calculations =================================
        # make yearly DAC costs based on learning
        totalYearlyDAC_need_histgrowth, totalYearlyDAC_need_min_histgrowth, totalYearlyDAC_need_max_histgrowth, \
            total_DAC_InstalledCapacity_histgrowth, total_DAC_InstalledCapacity_min_histgrowth, total_DAC_InstalledCapacity_max_histgrowth, \
            yearlyAddedCapacityDAC_histgrowth, yearlyAddedCapacityDAC_min_histgrowth, yearlyAddedCapacityDAC_max_histgrowth, \
            cost_DAC_ct_histgrowth, cost_DAC_ct_min_histgrowth, cost_DAC_ct_max_histgrowth = \
            make_learning_curve_DAC_II(LEARNING_RATE_DAC, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt_histgrowth,
                                       DAC_CDR_CO2_Gt_histgrowth, DAC_CDR_nonCO2_Gt_histgrowth,
                                       Delta_totalIndirectEmissions_histgrowth, DAC_CDR_nonCO2_Gt_min_histgrowth,
                                       DAC_CDR_nonCO2_Gt_max_histgrowth, Delta_totalIndirectEmissions_min_histgrowth,
                                       Delta_totalIndirectEmissions_max_histgrowth)

        # make yearly H2 and CO costs based on learning
        totalYearlyH2_need_histgrowth, total_H2_InstalledCapacity_histgrowth, yearlyAddedCapacityH2_histgrowth, yearlyH2_need_increase_histgrowth, cost_H2_ct_histgrowth, \
            totalYearlyCO_need_histgrowth, total_CO_InstalledCapacity_histgrowth, yearlyAddedCapacityCO_histgrowth, yearlyCO_need_increase_histgrowth, cost_CO_ct_histgrowth = \
            make_learning_curve_H2_CO_II(LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, H2_q0_Mt_2020,
                                         H2_c0_2020, H2_DACCU_Mt_histgrowth, CO_q0_Mt_2020, CO_c0_2020,
                                         CO_DACCU_Mt_histgrowth)

        # calculate cost of DAC
        total_yearly_DAC_Cost_histgrowth, total_yearly_DAC_Cost_min_histgrowth, total_yearly_DAC_Cost_max_histgrowth, yearly_DAC_DACCU_Cost_histgrowth, yearly_DAC_CDR_CO2_Cost_histgrowth, \
            yearly_DAC_CDR_nonCO2_Cost_histgrowth, yearly_DAC_CDR_nonCO2_Cost_min_histgrowth, yearly_DAC_CDR_nonCO2_Cost_max_histgrowth, \
            yearly_DAC_DeltaEmissions_Cost_histgrowth, yearly_DAC_DeltaEmissions_Cost_min_histgrowth, yearly_DAC_DeltaEmissions_Cost_max_histgrowth = calculate_DAC_costs_II(
            cost_DAC_ct_histgrowth, yearlyAddedCapacityDAC_histgrowth, totalYearlyDAC_need_histgrowth,
            DAC_DACCU_Gt_histgrowth, DAC_CDR_CO2_Gt_histgrowth, DAC_CDR_nonCO2_Gt_histgrowth,
            Delta_totalIndirectEmissions_histgrowth, cost_DAC_ct_MIN=cost_DAC_ct_min_histgrowth,
            cost_DAC_ct_MAX=cost_DAC_ct_max_histgrowth,
            yearlyAddedCapacityDAC_MIN=yearlyAddedCapacityDAC_min_histgrowth,
            yearlyAddedCapacityDAC_MAX=yearlyAddedCapacityDAC_max_histgrowth,
            totalYearlyDAC_need_MIN=totalYearlyDAC_need_min_histgrowth,
            totalYearlyDAC_need_MAX=totalYearlyDAC_need_max_histgrowth,
            DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_histgrowth,
            DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_histgrowth,
            Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_histgrowth,
            Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_histgrowth)

        # calculate cost of H2 and CO
        total_yearly_H2_Cost_histgrowth, total_yearly_CO_Cost_histgrowth = calculate_H2_CO_costs_II(
            cost_H2_ct_histgrowth, yearlyH2_need_increase_histgrowth,
            cost_CO_ct_histgrowth, yearlyCO_need_increase_histgrowth)

        # calculate final yearly costs including electricity, fossil fuel, CO2 transport and storage, etc.
        finalcost_histgrowth, finalcost_min_histgrowth, finalcost_max_histgrowth, finalcost_BAU_histgrowth, \
            finalcost_electricity_histgrowth, finalcost_electricity_min_histgrowth, finalcost_electricity_max_histgrowth, \
            finalcost_transport_storageCO2_histgrowth, finalcost_transport_storageCO2_min_histgrowth, finalcost_transport_storageCO2_max_histgrowth, \
            finalcost_fossilfuel_histgrowth, finalcost_heat_histgrowth, finalcost_heat_min_histgrowth, finalcost_heat_max_histgrowth, total_DACCU_electricity_cost_histgrowth, \
            total_DACCS_electricity_cost_histgrowth, total_DACCS_electricity_cost_min_histgrowth, total_DACCS_electricity_cost_max_histgrowth, \
            total_DACCU_production_cost_histgrowth, total_DACCS_cost_histgrowth, total_DACCS_cost_min_histgrowth, total_DACCS_cost_max_histgrowth, \
            total_DACCU_heat_cost_histgrowth, total_DACCS_heat_cost_histgrowth, total_DACCS_heat_cost_min_histgrowth, total_DACCS_heat_cost_max_histgrowth = \
            calculate_final_cost_II(ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                    DAC_DACCU_MWh_histgrowth, H2_DACCU_MWh_histgrowth,
                                    CO_DACCU_MWh_histgrowth, FT_DACCU_MWh_histgrowth, DAC_CDR_CO2_MWh_histgrowth,
                                    DAC_CDR_nonCO2_MWh_histgrowth, Delta_totalIndirectEmissions_histgrowth,
                                    DAC_CDR_CO2_Gt_histgrowth, DAC_CDR_nonCO2_Gt_histgrowth, FF_volumes_EJ_histgrowth,
                                    yearly_DAC_DACCU_Cost_histgrowth, total_yearly_H2_Cost_histgrowth,
                                    total_yearly_CO_Cost_histgrowth, total_yearly_DAC_Cost_histgrowth, df_input,
                                    CO_DACCU_MWh_heat_histgrowth, DAC_DACCU_MWh_heat_histgrowth,
                                    DAC_CDR_CO2_MWhth_histgrowth,
                                    DAC_CDR_nonCO2_MWhth_histgrowth,
                                    DAC_CDR_nonCO2_MWh_MIN=DAC_CDR_nonCO2_MWh_min_histgrowth,
                                    DAC_CDR_nonCO2_MWh_MAX=DAC_CDR_nonCO2_MWh_max_histgrowth,
                                    Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_histgrowth,
                                    Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_histgrowth,
                                    DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_histgrowth,
                                    DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_histgrowth,
                                    total_yearly_DAC_COST_MIN=total_yearly_DAC_Cost_min_histgrowth,
                                    total_yearly_DAC_COST_MAX=total_yearly_DAC_Cost_max_histgrowth,
                                    DAC_CDR_nonCO2_MWhth_MIN=DAC_CDR_nonCO2_MWh_min_histgrowth,
                                    DAC_CDR_nonCO2_MWhth_MAX=DAC_CDR_nonCO2_MWh_max_histgrowth)

        finalcost_per_liter_decreasing, finalcost_per_liter_stagnating, finalcost_per_liter, finalcost_per_liter_histgrowth, \
            finalcost_per_liter_BAU = calc_array_cost_per_liter(finalcost_decreasing, finalcost_stagnating,
                                                                finalcost, finalcost_histgrowth,
                                                                FF_volumes_Tg_decreasing, FF_volumes_Tg_stagnating,
                                                                FF_volumes_Tg, FF_volumes_Tg_histgrowth,
                                                                DACCU_volumes_Tg_decreasing,
                                                                DACCU_volumes_Tg_stagnating, DACCU_volumes_Tg,
                                                                DACCU_volumes_Tg_histgrowth,
                                                                cost5=finalcost_BAU, ff5=FF_volumes_Tg[0, 0],
                                                                daccu5=DACCU_volumes_Tg[0, 0])

    if running_contrails is True:

        # RUN IGNORING CONTRAILS #####
        # calculate flight emissions with separated CC
        _, _, flying_othernonCO2_emissions, flying_othernonCO2_emissions_min, \
            flying_othernonCO2_emissions_max, flying_othernonCO2_abated, flying_othernonCO2_abated_min, flying_othernonCO2_abated_max, \
            flying_CC_emissions, flying_CC_emissions_min, flying_CC_emissions_max, flying_CC_abated, flying_CC_abated_min, flying_CC_abated_max, \
            emissions_GWPstar_NOx, emissions_GWPstar_BC, \
            emissions_GWPstar_SO4, emissions_GWPstar_H2O, emissions_GWPstar_CC = \
            make_emissions_CO2equivalent_star(df_input, ERF_total, emissions_Tg_total, ERF_total_min, ERF_total_max,
                                              separate_contrails=True, output_all_emissiong_GWPstar=True)
        # calculate material and electricity indirect emissions in DACCS utilization - without CC
        _, _, _, DAC_CDR_othernonCO2_Gt, DAC_CDR_othernonCO2_Gt_min, DAC_CDR_othernonCO2_Gt_max, \
            DAC_CDR_othernonCO2_MWh, DAC_CDR_othernonCO2_MWh_min, DAC_CDR_othernonCO2_MWh_max, \
            DAC_CDR_othernonCO2_MWhth, DAC_CDR_othernonCO2_MWhth_min, DAC_CDR_othernonCO2_MWhth_max, \
            _, _, \
            DAC_CDR_othernonCO2_MaterialFootprint, DAC_CDR_othernonCO2_MaterialFootprint_min, DAC_CDR_othernonCO2_MaterialFootprint_max, \
            DAC_CDR_othernonCO2_ElectricityFootprint, DAC_CDR_othernonCO2_ElectricityFootprint_min, DAC_CDR_othernonCO2_ElectricityFootprint_max, \
            total_DAC_CDR_Footprint_noCC, total_DAC_CDR_Footprint_min_noCC, total_DAC_CDR_Footprint_max_noCC = \
            make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated,
                                                                   flying_othernonCO2_abated,
                                                                   flying_nonCO2_abated_MIN=flying_othernonCO2_abated_min,
                                                                   flying_nonCO2_abated_MAX=flying_othernonCO2_abated_max)
        # calculate all indirect emissions together (well-to-tank + indirect electricity and material footprint)
        totalIndirectEmissions_noCC, totalIndirectEmissions_min_noCC, totalIndirectEmissions_max_noCC, \
            Delta_totalIndirectEmissions_noCC, Delta_totalIndirectEmissions_min_noCC, Delta_totalIndirectEmissions_max_noCC, \
            BAU_EmissionsGt_noCC, BAU_EmissionsGt_min_noCC, BAU_EmissionsGt_max_noCC, \
            totalNetEmissions_noCC, totalNetEmissions_min_noCC, totalNetEmissions_max_noCC = \
            make_indirect_delta_net_emissions_II(yearlyWTT, total_DACCU_Footprint, total_DAC_CDR_Footprint_noCC,
                                                 flying_CO2_emissions,
                                                 flying_othernonCO2_emissions, flying_CO2_abated,
                                                 flying_othernonCO2_abated,
                                                 total_DAC_CDR_Footprint_min_noCC, total_DAC_CDR_Footprint_max_noCC,
                                                 flying_othernonCO2_emissions_min, flying_othernonCO2_emissions_max,
                                                 flying_othernonCO2_abated_min, flying_othernonCO2_abated_max
                                                 )
        total_abated_emissions_noCC = BAU_EmissionsGt_noCC - totalNetEmissions_noCC
        total_abated_emissions_min_noCC = BAU_EmissionsGt_min_noCC - totalNetEmissions_min_noCC
        total_abated_emissions_max_noCC = BAU_EmissionsGt_max_noCC - totalNetEmissions_max_noCC
        total_absolute_emissions_noCC = totalIndirectEmissions_noCC + flying_CO2_emissions + flying_othernonCO2_emissions
        total_absolute_emissions_min_noCC = totalIndirectEmissions_min_noCC + flying_CO2_emissions + flying_othernonCO2_emissions_min
        total_absolute_emissions_max_noCC = totalIndirectEmissions_max_noCC + flying_CO2_emissions + flying_othernonCO2_emissions_max

        # make yearly DAC costs based on learning - no CC
        totalYearlyDAC_need_noCC, totalYearlyDAC_need_min_noCC, totalYearlyDAC_need_max_noCC, \
            total_DAC_InstalledCapacity_noCC, total_DAC_InstalledCapacity_min_noCC, total_DAC_InstalledCapacity_max_noCC, \
            yearlyAddedCapacityDAC_noCC, yearlyAddedCapacityDAC_min_noCC, yearlyAddedCapacityDAC_max_noCC, \
            cost_DAC_ct_noCC, cost_DAC_ct_min_noCC, cost_DAC_ct_max_noCC = \
            make_learning_curve_DAC_II(LEARNING_RATE_DAC, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt, DAC_CDR_CO2_Gt,
                                       DAC_CDR_othernonCO2_Gt,
                                       Delta_totalIndirectEmissions_noCC, DAC_CDR_othernonCO2_Gt_min,
                                       DAC_CDR_othernonCO2_Gt_max, Delta_totalIndirectEmissions_min_noCC,
                                       Delta_totalIndirectEmissions_max_noCC)

        # calculate cost of DAC - with CC
        total_yearly_DAC_Cost_noCC, total_yearly_DAC_Cost_min_noCC, total_yearly_DAC_Cost_max_noCC, yearly_DAC_DACCU_Cost_noCC, yearly_DAC_CDR_CO2_Cost_noCC, \
            yearly_DAC_CDR_othernonCO2_Cost_noCC, yearly_DAC_CDR_othernonCO2_Cost_min_noCC, yearly_DAC_CDR_othernonCO2_Cost_max_noCC, \
            yearly_DAC_DeltaEmissions_Cost_noCC, yearly_DAC_DeltaEmissions_Cost_min_noCC, yearly_DAC_DeltaEmissions_Cost_max_noCC = calculate_DAC_costs_II(
            cost_DAC_ct_noCC, yearlyAddedCapacityDAC_noCC, totalYearlyDAC_need_noCC,
            DAC_DACCU_Gt, DAC_CDR_CO2_Gt, DAC_CDR_othernonCO2_Gt,
            Delta_totalIndirectEmissions_noCC, cost_DAC_ct_MIN=cost_DAC_ct_min_noCC,
            cost_DAC_ct_MAX=cost_DAC_ct_max_noCC,
            yearlyAddedCapacityDAC_MIN=yearlyAddedCapacityDAC_min_noCC,
            yearlyAddedCapacityDAC_MAX=yearlyAddedCapacityDAC_max_noCC,
            totalYearlyDAC_need_MIN=totalYearlyDAC_need_min_noCC,
            totalYearlyDAC_need_MAX=totalYearlyDAC_need_max_noCC, DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_othernonCO2_Gt_min,
            DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_othernonCO2_Gt_max,
            Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_noCC,
            Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_noCC)

        # calculate final yearly costs including electricity, fossil fuel, CO2 transport and storage, etc.
        finalcost_noCC, finalcost_min_noCC, finalcost_max_noCC, finalcost_BAU_noCC, \
        finalcost_electricity_noCC, finalcost_electricity_min_noCC, finalcost_electricity_max_noCC, \
        finalcost_transport_storageCO2_noCC, finalcost_transport_storageCO2_min_noCC, finalcost_transport_storageCO2_max_noCC, \
        finalcost_fossilfuel_noCC, finalcost_heat_noCC, finalcost_heat_min_noCC, finalcost_heat_max_noCC, total_DACCU_electricity_cost_noCC, \
        total_DACCS_electricity_cost_noCC, total_DACCS_electricity_cost_min_noCC, total_DACCS_electricity_cost_max_noCC, \
        total_DACCU_production_cost_noCC, total_DACCS_cost_noCC, total_DACCS_cost_min_noCC, total_DACCS_cost_max_noCC, \
        total_DACCU_heat_cost_noCC, total_DACCS_heat_cost_noCC, total_DACCS_heat_cost_min_noCC, total_DACCS_heat_cost_max_noCC = \
        calculate_final_cost_II(ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_DACCU_MWh,
                                H2_DACCU_MWh,
                                CO_DACCU_MWh, FT_DACCU_MWh, DAC_CDR_CO2_MWh, DAC_CDR_nonCO2_MWh,
                                Delta_totalIndirectEmissions_noCC,
                                DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt, FF_volumes_EJ, yearly_DAC_DACCU_Cost_noCC,
                                total_yearly_H2_Cost,
                                total_yearly_CO_Cost, total_yearly_DAC_Cost_noCC, df_input, CO_DACCU_MWh_heat,
                                DAC_DACCU_MWh_heat, DAC_CDR_CO2_MWhth,
                                DAC_CDR_nonCO2_MWhth,
                                DAC_CDR_nonCO2_MWh_MIN=DAC_CDR_nonCO2_MWh_min,
                                DAC_CDR_nonCO2_MWh_MAX=DAC_CDR_nonCO2_MWh_max,
                                Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_noCC,
                                Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_noCC,
                                DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min,
                                DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max,
                                total_yearly_DAC_COST_MIN=total_yearly_DAC_Cost_min_noCC,
                                total_yearly_DAC_COST_MAX=total_yearly_DAC_Cost_max_noCC,
                                DAC_CDR_nonCO2_MWhth_MIN=DAC_CDR_nonCO2_MWh_min,
                                DAC_CDR_nonCO2_MWhth_MAX=DAC_CDR_nonCO2_MWh_max)

        total_DACCU_CAPEX_cost_noCC = (
                    total_DACCU_production_cost_noCC - total_DACCU_electricity_cost_noCC - total_DACCU_heat_cost_noCC)
        total_DACCS_CAPEX_cost_noCC = (
                    yearly_DAC_CDR_CO2_Cost_noCC + yearly_DAC_CDR_othernonCO2_Cost_noCC + yearly_DAC_DeltaEmissions_Cost_noCC)
        total_DACCS_CAPEX_cost_noCC_min = yearly_DAC_CDR_CO2_Cost_noCC + yearly_DAC_CDR_othernonCO2_Cost_min_noCC + yearly_DAC_DeltaEmissions_Cost_min_noCC
        total_DACCS_CAPEX_cost_noCC_max = yearly_DAC_CDR_CO2_Cost_noCC + yearly_DAC_CDR_othernonCO2_Cost_max_noCC + yearly_DAC_DeltaEmissions_Cost_max_noCC

        total_cost_per_CO2eq_noCC = finalcost_noCC / (total_abated_emissions_noCC * 10 ** 12)
        DAC_total_Gt_noCC = DAC_DACCU_Gt + DAC_CDR_CO2_Gt + DAC_CDR_othernonCO2_Gt
        DAC_yearly_cost_per_ton_noCC = total_yearly_DAC_Cost_noCC / (DAC_total_Gt_noCC * 10 ** 9)
        cost_by_demand_noCC = finalcost_noCC / (baseDemand_km * 10 ** 6)

        #========================== CONTRAILS AVOIDANCE VIA REROUTING ======================
        # make aviation demand (fuels = EJ) and in distance flown
        baseDemand_EJ_rerouting, _ = make_base_demand_EJ_KM_II(df_input, GROWTH_RATE_AVIATION_FUEL_DEMAND,
                                                               EFFICIENCY_INCREASE_YEARLY, rerouting=True)

        # calculate volumes of DACCUs and FF both in EJ and Tg
        DACCU_volumes_EJ_rerouting, FF_volumes_EJ_rerouting, DACCU_volumes_Tg_rerouting, FF_volumes_Tg_rerouting = \
            make_DACCU_FF_EJ_Tg_II(df_input, baseDemand_EJ_rerouting)

        # caclulate distance flown by DACCUs and FF and in Tg fuel used from 1990
        FF_km_1990_rerouting, DACCU_km_1990_rerouting, \
            FF_Tg_1990_rerouting, DACCU_Tg_1990_rerouting = make_historic_demand_II(df_input, df_emissions_input,
                                                                                    baseDemand_EJ_rerouting,
                                                                                    baseDemand_km)

        # calculate DAC (GtCO2 and MWh), H2 (Mt and MWh), CO (Mt), FT (MWh) and electricity needed to produce DACCUs
        DAC_DACCU_Gt_rerouting, DAC_DACCU_MWh_rerouting, H2_DACCU_Mt_rerouting, H2_DACCU_MWh_rerouting, CO_DACCU_Mt_rerouting, \
            CO_DACCU_MWh_rerouting, FT_DACCU_MWh_rerouting, \
            DAC_Diesel_Gt_rerouting, DAC_Diesel_MWh_rerouting, H2_Diesel_Mt_rerouting, H2_Diesel_MWh_rerouting, CO_Diesel_Mt_rerouting, CO_Diesel_MWh_rerouting, FT_Diesel_MWh_rerouting, \
            CO_DACCU_MWh_heat_rerouting, DAC_DACCU_MWh_heat_rerouting, CO_Diesel_MWh_heat_rerouting, DAC_Diesel_MWh_heat_rerouting = \
            make_DACCU_need_DAC_H2_CO_FT_electricity_II(df_input, DACCU_volumes_Tg_rerouting, JETFUEL_ALLOCATION_SHARE,
                                                        configuration=CONFIGURATION, efficiency_increase=True)

        # =================== LCA part with Uncertainty =========================================

        emissions_Tg_ff_rerouting, emissions_Tg_DACCU_rerouting, emissions_Tg_total_rerouting, ERF_ff_rerouting, ERF_ff_min_rerouting, ERF_ff_max_rerouting, \
            ERF_DACCU_rerouting, ERF_DACCU_min_rerouting, ERF_DACCU_max_rerouting, \
            ERF_total_rerouting, ERF_total_min_rerouting, ERF_total_max_rerouting = make_emissions_and_ERF(df_input,
                                                                                                           df_emissions_input,
                                                                                                           FF_km_1990_rerouting,
                                                                                                           DACCU_km_1990_rerouting,
                                                                                                           FF_Tg_1990_rerouting,
                                                                                                           DACCU_Tg_1990_rerouting,
                                                                                                           scenario=SCENARIO,
                                                                                                           erf_uncertain=ERF_factors_uncertain,
                                                                                                           uncertainty_daccu=True,
                                                                                                           rerouting=True)

        flying_CO2_emissions_rerouting, flying_CO2_abated_rerouting, flying_nonCO2_emissions_rerouting, flying_nonCO2_emissions_min_rerouting, \
            flying_nonCO2_emissions_max_rerouting, flying_nonCO2_abated_rerouting, flying_nonCO2_abated_min_rerouting, flying_nonCO2_abated_max_rerouting, \
            emissions_GWPstar_NOx_rerouting, emissions_GWPstar_BC_rerouting, \
            emissions_GWPstar_SO4_rerouting, emissions_GWPstar_H2O_rerouting, emissions_GWPstar_CC_rerouting = \
            make_emissions_CO2equivalent_star(df_input, ERF_total_rerouting, emissions_Tg_total_rerouting,
                                              ERF_TOTAL_MIN=ERF_total_min_rerouting,
                                              ERF_TOTAL_MAX=ERF_total_max_rerouting,
                                              separate_contrails=False, output_all_emissiong_GWPstar=True)

        # make well-to-tank emissions (driven by FF)
        yearlyWTT_rerouting = make_WTT_emissions_II(df_input, FF_volumes_EJ_rerouting)

        # calculate material and electricity indirect emissions in DACCU production
        DAC_DACCU_MaterialFootprint_rerouting, DAC_DACCU_ElectricityFootprint_rerouting, H2_DACCU_MaterialFootprint_rerouting, \
            H2_DACCU_ElectricityFootprint_rerouting, CO_DACCU_MaterialFootprint_rerouting, CO_DACCU_ElectricityFootprint_rerouting, \
            FT_DACCU_ElectricityFootprint_rerouting, total_DACCU_MaterialFootprint_rerouting, total_DACCU_ElectricitryFootprint_rerouting, total_DACCU_Footprint_rerouting = \
            make_DACCU_indirect_emissions_II(df_input, DAC_DACCU_Gt_rerouting, DAC_DACCU_MWh_rerouting,
                                             H2_DACCU_Mt_rerouting, H2_DACCU_MWh_rerouting, CO_DACCU_Mt_rerouting,
                                             CO_DACCU_MWh_rerouting, FT_DACCU_MWh_rerouting)

        # calculate material and electricity indirect emissions in DACCS utilization - without CC
        DAC_CDR_CO2_Gt_rerouting, DAC_CDR_CO2_MWh_rerouting, DAC_CDR_CO2_MWhth_rerouting, DAC_CDR_nonCO2_Gt_rerouting, DAC_CDR_nonCO2_Gt_min_rerouting, DAC_CDR_nonCO2_Gt_max_rerouting, \
            DAC_CDR_nonCO2_MWh_rerouting, DAC_CDR_nonCO2_MWh_min_rerouting, DAC_CDR_nonCO2_MWh_max_rerouting, \
            DAC_CDR_nonCO2_MWhth_rerouting, DAC_CDR_nonCO2_MWhth_min_rerouting, DAC_CDR_nonCO2_MWhth_max_rerouting, \
            DAC_CDR_CO2_MaterialFootprint_rerouting, DAC_CDR_CO2_ElectricityFootprint_rerouting, \
            DAC_CDR_nonCO2_MaterialFootprint_rerouting, DAC_CDR_nonCO2_MaterialFootprint_min_rerouting, DAC_CDR_nonCO2_MaterialFootprint_max_rerouting, \
            DAC_CDR_nonCO2_ElectricityFootprint_rerouting, DAC_CDR_nonCO2_ElectricityFootprint_min_rerouting, DAC_CDR_nonCO2_ElectricityFootprint_max_rerouting, \
            total_DAC_CDR_Footprint_rerouting, total_DAC_CDR_Footprint_min_rerouting, total_DAC_CDR_Footprint_max_rerouting = \
            make_DAC_CDR_need_DAC_electricty_indirect_emissions_II(df_input, flying_CO2_abated_rerouting,
                                                                   flying_nonCO2_abated_rerouting,
                                                                   flying_nonCO2_abated_MIN=flying_nonCO2_abated_min_rerouting,
                                                                   flying_nonCO2_abated_MAX=flying_nonCO2_abated_max_rerouting)
        # calculate all indirect emissions together (well-to-tank + indirect electricity and material footprint)
        totalIndirectEmissions_rerouting, totalIndirectEmissions_min_rerouting, totalIndirectEmissions_max_rerouting, \
            Delta_totalIndirectEmissions_rerouting, Delta_totalIndirectEmissions_min_rerouting, Delta_totalIndirectEmissions_max_rerouting, \
            BAU_EmissionsGt_rerouting, BAU_EmissionsGt_min_rerouting, BAU_EmissionsGt_max_rerouting, \
            totalNetEmissions_rerouting, totalNetEmissions_min_rerouting, totalNetEmissions_max_rerouting = \
            make_indirect_delta_net_emissions_II(yearlyWTT_rerouting, total_DACCU_Footprint_rerouting,
                                                 total_DAC_CDR_Footprint_rerouting, flying_CO2_emissions_rerouting,
                                                 flying_nonCO2_emissions_rerouting, flying_CO2_abated_rerouting,
                                                 flying_nonCO2_abated_rerouting,
                                                 total_DAC_CDR_Footprint_min_rerouting,
                                                 total_DAC_CDR_Footprint_max_rerouting,
                                                 flying_nonCO2_emissions_min_rerouting,
                                                 flying_nonCO2_emissions_max_rerouting,
                                                 flying_nonCO2_abated_min_rerouting, flying_nonCO2_abated_max_rerouting
                                                 )

        total_abated_emissions_rerouting = BAU_EmissionsGt - totalNetEmissions_rerouting
        total_abated_emissions_min_rerouting = BAU_EmissionsGt_min - totalNetEmissions_min_rerouting
        total_abated_emissions_max_rerouting = BAU_EmissionsGt_max - totalNetEmissions_max_rerouting

        total_absolute_emissions_rerouting = totalIndirectEmissions_rerouting + flying_CO2_emissions_rerouting + flying_nonCO2_emissions_rerouting
        total_absolute_emissions_min_rerouting = totalIndirectEmissions_min_rerouting + flying_CO2_emissions_rerouting + flying_nonCO2_emissions_min_rerouting
        total_absolute_emissions_max_rerouting = totalIndirectEmissions_max_rerouting + flying_CO2_emissions_rerouting + flying_nonCO2_emissions_max_rerouting

        # =============== COST Calculations =================================
        # make yearly DAC costs based on learning
        totalYearlyDAC_need_rerouting, totalYearlyDAC_need_min_rerouting, totalYearlyDAC_need_max_rerouting, \
            total_DAC_InstalledCapacity_rerouting, total_DAC_InstalledCapacity_min_rerouting, total_DAC_InstalledCapacity_max_rerouting, \
            yearlyAddedCapacityDAC_rerouting, yearlyAddedCapacityDAC_min_rerouting, yearlyAddedCapacityDAC_max_rerouting, \
            cost_DAC_ct_rerouting, cost_DAC_ct_min_rerouting, cost_DAC_ct_max_rerouting = \
            make_learning_curve_DAC_II(LEARNING_RATE_DAC, DAC_q0_Gt_2020, DAC_c0_2020, DAC_DACCU_Gt, DAC_CDR_CO2_Gt,
                                       DAC_CDR_nonCO2_Gt_rerouting,
                                       Delta_totalIndirectEmissions_rerouting, DAC_CDR_nonCO2_Gt_min_rerouting,
                                       DAC_CDR_nonCO2_Gt_max_rerouting, Delta_totalIndirectEmissions_min_rerouting,
                                       Delta_totalIndirectEmissions_max_rerouting)

        # make yearly H2 and CO costs based on learning
        totalYearlyH2_need_rerouting, total_H2_InstalledCapacity_rerouting, yearlyAddedCapacityH2_rerouting, yearlyH2_need_increase_rerouting, cost_H2_ct_rerouting, \
            totalYearlyCO_need_rerouting, total_CO_InstalledCapacity_rerouting, yearlyAddedCapacityCO_rerouting, yearlyCO_need_increase_rerouting, cost_CO_ct_rerouting = \
            make_learning_curve_H2_CO_II(LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, H2_q0_Mt_2020, H2_c0_2020,
                                         H2_DACCU_Mt_rerouting, CO_q0_Mt_2020, CO_c0_2020, CO_DACCU_Mt_rerouting)

        # calculate cost of DAC
        total_yearly_DAC_Cost_rerouting, total_yearly_DAC_Cost_min_rerouting, total_yearly_DAC_Cost_max_rerouting, yearly_DAC_DACCU_Cost_rerouting, yearly_DAC_CDR_CO2_Cost_rerouting, \
            yearly_DAC_CDR_nonCO2_Cost_rerouting, yearly_DAC_CDR_nonCO2_Cost_min_rerouting, yearly_DAC_CDR_nonCO2_Cost_max_rerouting, \
            yearly_DAC_DeltaEmissions_Cost_rerouting, yearly_DAC_DeltaEmissions_Cost_min_rerouting, yearly_DAC_DeltaEmissions_Cost_max_rerouting = calculate_DAC_costs_II(
            cost_DAC_ct_rerouting, yearlyAddedCapacityDAC_rerouting, totalYearlyDAC_need_rerouting,
            DAC_DACCU_Gt_rerouting, DAC_CDR_CO2_Gt_rerouting, DAC_CDR_nonCO2_Gt_rerouting,
            Delta_totalIndirectEmissions_rerouting,
            cost_DAC_ct_MIN=cost_DAC_ct_min_rerouting, cost_DAC_ct_MAX=cost_DAC_ct_max_rerouting,
            yearlyAddedCapacityDAC_MIN=yearlyAddedCapacityDAC_min_rerouting,
            yearlyAddedCapacityDAC_MAX=yearlyAddedCapacityDAC_max_rerouting,
            totalYearlyDAC_need_MIN=totalYearlyDAC_need_min_rerouting,
            totalYearlyDAC_need_MAX=totalYearlyDAC_need_max_rerouting,
            DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_rerouting, DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_rerouting,
            Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_rerouting,
            Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_rerouting)

        # calculate cost of H2 and CO
        total_yearly_H2_Cost_rerouting, total_yearly_CO_Cost_rerouting = calculate_H2_CO_costs_II(
            cost_H2_ct_rerouting, yearlyH2_need_increase_rerouting,
            cost_CO_ct_rerouting, yearlyCO_need_increase_rerouting)

        # calculate final yearly costs including electricity, fossil fuel, CO2 transport and storage, etc.
        finalcost_rerouting, finalcost_min_rerouting, finalcost_max_rerouting, finalcost_BAU_rerouting, \
            finalcost_electricity_rerouting, finalcost_electricity_min_rerouting, finalcost_electricity_max_rerouting, \
            finalcost_transport_storageCO2_rerouting, finalcost_transport_storageCO2_min_rerouting, finalcost_transport_storageCO2_max_rerouting, \
            finalcost_fossilfuel_rerouting, finalcost_heat_rerouting, finalcost_heat_min_rerouting, finalcost_heat_max_rerouting, \
            total_DACCU_electricity_cost_rerouting, total_DACCS_electricity_cost_rerouting, total_DACCS_electricity_cost_min_rerouting, \
            total_DACCS_electricity_cost_max_rerouting, total_DACCU_production_cost_rerouting, total_DACCS_cost_rerouting, total_DACCS_cost_min_rerouting, \
            total_DACCS_cost_max_rerouting, total_DACCU_heat_cost_rerouting, total_DACCS_heat_cost_rerouting, total_DACCS_heat_cost_min_rerouting, \
            total_DACCS_heat_cost_max_rerouting = calculate_final_cost_II(
            ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_DACCU_MWh_rerouting,
            H2_DACCU_MWh_rerouting,
            CO_DACCU_MWh_rerouting, FT_DACCU_MWh_rerouting, DAC_CDR_CO2_MWh_rerouting, DAC_CDR_nonCO2_MWh_rerouting,
            Delta_totalIndirectEmissions_rerouting, DAC_CDR_CO2_Gt_rerouting, DAC_CDR_nonCO2_Gt_rerouting,
            FF_volumes_EJ_rerouting,
            yearly_DAC_DACCU_Cost_rerouting, total_yearly_H2_Cost_rerouting, total_yearly_CO_Cost_rerouting,
            total_yearly_DAC_Cost_rerouting,
            df_input, CO_DACCU_MWh_heat_rerouting, DAC_DACCU_MWh_heat_rerouting, DAC_CDR_CO2_MWhth_rerouting,
            DAC_CDR_nonCO2_MWhth_rerouting,
            DAC_CDR_nonCO2_MWh_MIN=DAC_CDR_nonCO2_MWh_min_rerouting,
            DAC_CDR_nonCO2_MWh_MAX=DAC_CDR_nonCO2_MWh_max_rerouting,
            Delta_totalIndirectEmissions_MIN=Delta_totalIndirectEmissions_min_rerouting,
            Delta_totalIndirectEmissions_MAX=Delta_totalIndirectEmissions_max_rerouting,
            DAC_CDR_nonCO2_Gt_MIN=DAC_CDR_nonCO2_Gt_min_rerouting, DAC_CDR_nonCO2_Gt_MAX=DAC_CDR_nonCO2_Gt_max_rerouting,
            total_yearly_DAC_COST_MIN=total_yearly_DAC_Cost_min_rerouting,
            total_yearly_DAC_COST_MAX=total_yearly_DAC_Cost_max_rerouting,
            DAC_CDR_nonCO2_MWhth_MIN=DAC_CDR_nonCO2_MWhth_min_rerouting,
            DAC_CDR_nonCO2_MWhth_MAX=DAC_CDR_nonCO2_MWhth_max_rerouting
        )

        total_DACCU_CAPEX_cost_rerouting = (
                    total_DACCU_production_cost_rerouting - total_DACCU_electricity_cost_rerouting - total_DACCU_heat_cost_rerouting)
        total_DACCS_CAPEX_cost_rerouting = (
                    yearly_DAC_CDR_CO2_Cost_rerouting + yearly_DAC_CDR_nonCO2_Cost_rerouting + yearly_DAC_DeltaEmissions_Cost_rerouting)
        total_DACCS_CAPEX_cost_min_rerouting = (
                    yearly_DAC_CDR_CO2_Cost_rerouting + yearly_DAC_CDR_nonCO2_Cost_min_rerouting + yearly_DAC_DeltaEmissions_Cost_min_rerouting)
        total_DACCS_CAPEX_cost_max_rerouting = (
                    yearly_DAC_CDR_CO2_Cost_rerouting + yearly_DAC_CDR_nonCO2_Cost_max_rerouting + yearly_DAC_DeltaEmissions_Cost_max_rerouting)

    if plot_main_figures is True:

        # FIGURE 1 - scenarios explanation and ERF of scenarios
        # Call the function with np_bau as a series of zeros
        prepdata_DAC_CDR_CO2 = prep_data_bardata(DAC_CDR_CO2_Gt[2, :, :], DAC_CDR_CO2_Gt[1, :, :],
                                                 np_bau=np.zeros((2, 41)))

        prepdata_DAC_CDR_nonCO2 = prep_data_bardata(DAC_CDR_nonCO2_Gt[2, :, :], DAC_CDR_nonCO2_Gt[1, :, :],
                                                    np_bau=np.zeros((2, 41)))
        prepdata_DAC_CDR_othernonCO2 = prep_data_bardata(DAC_CDR_othernonCO2_Gt[2, :, :], DAC_CDR_othernonCO2_Gt[1, :, :],
                                                    np_bau=np.zeros((2, 41)))
        prepdata_DAC_CDR_CC = prepdata_DAC_CDR_nonCO2 - prepdata_DAC_CDR_othernonCO2

        prepdata_flying_CO2 = prep_data_bardata(flying_CO2_emissions[2, :, :], flying_CO2_emissions[1, :, :],
                                                    np_bau=flying_CO2_emissions[0, :, :])
        prepdata_flying_nonCO2 = prep_data_bardata(flying_nonCO2_emissions[2, :, :], flying_nonCO2_emissions[1, :, :],
                                                np_bau=flying_nonCO2_emissions[0, :, :])
        prepdata_flying_othernonCO2 = prep_data_bardata(flying_othernonCO2_emissions[2, :, :], flying_othernonCO2_emissions[1, :, :],
                                                np_bau=flying_othernonCO2_emissions[0, :, :])
        prepdata_flying_CC = prep_data_bardata(flying_CC_emissions[2, :, :],
                                                        flying_CC_emissions[1, :, :],
                                                        np_bau=flying_CC_emissions[0, :, :])
        prepdata_indirect_emissions = prep_data_bardata(totalIndirectEmissions[2,:,:]-Delta_totalIndirectEmissions[2,:,:],
                                                        totalIndirectEmissions[1,:,:]-Delta_totalIndirectEmissions[1,:,:],
                                                        np_bau=yearlyWTT[0,:,:])

        # prepped min

        prepdata_DAC_CDR_nonCO2_min = prep_data_bardata(DAC_CDR_nonCO2_Gt_min[2, :, :], DAC_CDR_nonCO2_Gt_min[1, :, :],
                                                    np_bau=np.zeros((2, 41)))
        prepdata_DAC_CDR_othernonCO2_min = prep_data_bardata(DAC_CDR_othernonCO2_Gt_min[2, :, :], DAC_CDR_othernonCO2_Gt_min[1, :, :],
                                                    np_bau=np.zeros((2, 41)))
        prepdata_DAC_CDR_CC_min = prepdata_DAC_CDR_nonCO2_min - prepdata_DAC_CDR_othernonCO2_min

        prepdata_flying_nonCO2_min = prep_data_bardata(flying_nonCO2_emissions_min[2, :, :],
                                                       flying_nonCO2_emissions_min[1, :, :],
                                                np_bau=flying_nonCO2_emissions_min[0, :, :])
        prepdata_flying_othernonCO2_min = prep_data_bardata(flying_othernonCO2_emissions_min[2, :, :],
                                                            flying_othernonCO2_emissions_min[1, :, :],
                                                np_bau=flying_othernonCO2_emissions_min[0, :, :])
        prepdata_flying_CC_min = prep_data_bardata(flying_CC_emissions_min[2, :, :],
                                                        flying_CC_emissions_min[1, :, :],
                                                        np_bau=flying_CC_emissions_min[0, :, :])
        prepdata_indirect_emissions_min = prep_data_bardata(totalIndirectEmissions_min[2,:,:]-Delta_totalIndirectEmissions_min[2,:,:],
                                                        totalIndirectEmissions_min[1,:,:]--Delta_totalIndirectEmissions_min[1,:,:],
                                                        np_bau=yearlyWTT[0,:,:])

        # prepped max
        prepdata_DAC_CDR_nonCO2_max = prep_data_bardata(DAC_CDR_nonCO2_Gt_max[2, :, :], DAC_CDR_nonCO2_Gt_max[1, :, :],
                                                    np_bau=np.zeros((2, 41)))
        prepdata_DAC_CDR_othernonCO2_max = prep_data_bardata(DAC_CDR_othernonCO2_Gt_max[2, :, :], DAC_CDR_othernonCO2_Gt_max[1, :, :],
                                                    np_bau=np.zeros((2, 41)))
        prepdata_DAC_CDR_CC_max = prepdata_DAC_CDR_nonCO2_max - prepdata_DAC_CDR_othernonCO2_max

        prepdata_flying_nonCO2_max = prep_data_bardata(flying_nonCO2_emissions_max[2, :, :],
                                                       flying_nonCO2_emissions_max[1, :, :],
                                                np_bau=flying_nonCO2_emissions_max[0, :, :])
        prepdata_flying_othernonCO2_max = prep_data_bardata(flying_othernonCO2_emissions_max[2, :, :],
                                                            flying_othernonCO2_emissions_max[1, :, :],
                                                np_bau=flying_othernonCO2_emissions_max[0, :, :])
        prepdata_flying_CC_max = prep_data_bardata(flying_CC_emissions_max[2, :, :],
                                                        flying_CC_emissions_max[1, :, :],
                                                        np_bau=flying_CC_emissions_max[0, :, :])
        prepdata_indirect_emissions_max = prep_data_bardata(totalIndirectEmissions_max[2,:,:]-Delta_totalIndirectEmissions_max[2,:,:],
                                                        totalIndirectEmissions_max[1,:,:]--Delta_totalIndirectEmissions_max[1,:,:],
                                                        np_bau=yearlyWTT[0,:,:])

        plot_stacked_bars_four_scenarios(-prepdata_DAC_CDR_CC, -prepdata_DAC_CDR_othernonCO2,
                                        -prepdata_DAC_CDR_CO2,
                                        YEAR,
                                        'CDR contrails', 'CDR other non-CO$_2$', 'CDR CO$_2$',
                                        prepdata_flying_CO2, 'Flight CO$_2$',
                                        prepdata_flying_othernonCO2, 'Flight other non-CO$_2$',
                                        prepdata_flying_CC, 'Flight contrails',
                                        prepdata_indirect_emissions, 'Lifecycle',
                                        scenario='Scenarios explanation',
                                        scenario2='BAU \n carbon neutrality', scenario3='DACCU \n carbon neutrality', scenario1='DACCU \n carbon neutrality',
                                         scenario4='DACCS \n carbon neutrality', scenario5='DACCU \n climate neutrality',
                                         scenario6='DACCS \n climate neutrality',
                                        what="Emissions", palette=rgb_colors,
                                         np2_min=-prepdata_DAC_CDR_othernonCO2_min,
                                         np2_max=-prepdata_DAC_CDR_othernonCO2_max,
                                         np3_min=-prepdata_DAC_CDR_CC_min,
                                         np3_max=-prepdata_DAC_CDR_CC_max,
                                         np5_min=prepdata_flying_othernonCO2_min,
                                         np5_max=prepdata_flying_othernonCO2_max,
                                         np6_min=prepdata_flying_CC_min,
                                         np6_max=prepdata_flying_CC_max,
                                         np7_min=prepdata_indirect_emissions_min,
                                         np7_max=prepdata_indirect_emissions_max,
                                            BAU_cost=np.zeros((len(flying_CO2_emissions[0, 0]))),
                                         fmt_choice='o', error_label='Net emissions', BAU_label=None)


        # FIGURE 2 - COST OF DIFFERENT SCENARIOS
        plot_stacked_bars_two_scenarios(
            cost_per_liter_DACCU_CAPEX, cost_per_liter_DACCS_CAPEX, cost_per_liter_electricity,
            YEAR,
            'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
            cost_per_liter_fossilfuel, 'Fossil kerosene',
            cost_per_liter_transport_storageCO2, 'CO$_2$ transport and storage',
            cost_per_liter_heat, 'Heat', scenario='Carbon neutrality',
            what="Cost per liter fuel", palette=PALETTE,
            np2_min=cost_per_liter_DACCS_CAPEX_min,
            np2_max=cost_per_liter_DACCS_CAPEX_max,
            np3_min=cost_per_liter_electricity_min, np3_max=cost_per_liter_electricity_max,
            np5_min=cost_per_liter_transport_storageCO2_min,
            np5_max=cost_per_liter_transport_storageCO2_max,
            np6_min=cost_per_liter_heat_min,
            np6_max=cost_per_liter_heat_max,
            BAU_cost=calculate_cost_per_liter(finalcost_BAU, FF_volumes_Tg[0, 0], DACCU_volumes_Tg[0, 0]))

        # cost per total abated emissions
        plot_stacked_bars_two_scenarios(
            (total_DACCU_production_cost - total_DACCU_electricity_cost - total_DACCU_heat_cost) / (
                        total_abated_emissions * 10 ** 9),
            (yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost) / (total_abated_emissions * 10 ** 9),
            finalcost_electricity / (total_abated_emissions * 10 ** 9), YEAR,
            'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
            finalcost_fossilfuel / (total_abated_emissions * 10 ** 9), 'Fossil kerosene',
            finalcost_transport_storageCO2 / (total_abated_emissions * 10 ** 9), 'CO$_2$ transport and storage',
            finalcost_heat / (total_abated_emissions * 10 ** 9),
            'Heat',
            scenario='Carbon neutrality',
            what="Cost per emissions", palette=PALETTE, cumulative=False,
            np2_min=(yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_min) / (total_abated_emissions * 10 ** 9),
            np2_max=(yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_max) / (total_abated_emissions * 10 ** 9),
            np3_min=finalcost_electricity_min / (total_abated_emissions * 10 ** 9),
            np3_max=finalcost_electricity_max / (total_abated_emissions * 10 ** 9),
            np5_min=(finalcost_transport_storageCO2_min) / (total_abated_emissions * 10 ** 9),
            np5_max=(finalcost_transport_storageCO2_max) / (total_abated_emissions * 10 ** 9),
            np6_min=(finalcost_heat) / (total_abated_emissions * 10 ** 9),
            np6_max=(finalcost_heat) / (total_abated_emissions * 10 ** 9)
            )
        # cost per DAC installed by 2050
        plot_stacked_bars_two_scenarios(
            (total_DACCU_production_cost - total_DACCU_electricity_cost) / (DAC_total_Gt * 10 ** 9),
            (yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost) / (DAC_total_Gt * 10 ** 9),
            finalcost_electricity / (DAC_total_Gt * 10 ** 9), YEAR,
            'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
            finalcost_fossilfuel / (DAC_total_Gt * 10 ** 9), 'Fossil kerosene',
            finalcost_transport_storageCO2 / (DAC_total_Gt * 10 ** 9),
            'CO$_2$ transport and storage',
            finalcost_heat / (total_abated_emissions * 10 ** 9),
            'Heat',
            scenario='Carbon neutrality',
            what="Cost per DAC", palette=PALETTE, cumulative=False,
            np2_min=(yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_min) /
                    (DAC_total_Gt * 10 ** 9),
            np2_max=(yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_max) / (
                    DAC_total_Gt * 10 ** 9),
            np3_min=finalcost_electricity_min / (DAC_total_Gt * 10 ** 9),
            np3_max=finalcost_electricity_max / (DAC_total_Gt * 10 ** 9),
            np5_min=(finalcost_transport_storageCO2_min) / (DAC_total_Gt * 10 ** 9),
            np5_max=(finalcost_transport_storageCO2_max) / (DAC_total_Gt * 10 ** 9),
            np6_min=(finalcost_heat) / (total_abated_emissions * 10 ** 9),
            np6_max=(finalcost_heat) / (total_abated_emissions * 10 ** 9)
            )

        # FIGURE 3 - price per ticket
        # cost per passenger by 2050 - here no uncertainty since it's already quite uncertain the main calculation
        plot_stacked_bars_two_scenarios(cost_neutrality_per_passenger_LN_NY, cost_neutrality_per_passenger_LN_Berlin,
                                        cost_neutrality_per_passenger_LN_Perth, YEAR,
                                        'London-New York', 'London-Berlin', 'London-Perth',
                                        what='Cost neutrality per flight', scenario='Carbon neutrality',
                                        palette=PALETTE, stacked=False)
        # change in cost per passenger  by 2050
        plot_stacked_bars_two_scenarios(increase_neutrality_flight_price_LN_NY * 100,
                                        increase_neutrality_flight_price_LN_Berlin * 100,
                                        increase_neutrality_flight_price_LN_Perth * 100, YEAR,
                                        'London-New York', 'London-Berlin', 'London-Perth',
                                        what='Change flight price', scenario='Carbon neutrality', palette=PALETTE,
                                        stacked=False)

        # FIGURE 4 - variations in demand and contrail mitigation
        prepped_DAC_CDR_CO2_variations = prep_data_bardata(DAC_CDR_CO2_Gt[1,:,:],
                                                   DAC_CDR_CO2_Gt_stagnating[1,:,:],
                                                   DAC_CDR_CO2_Gt_decreasing[1,:,:],
                                                   DAC_CDR_CO2_Gt_rerouting[1,:,:],
                                                   DAC_CDR_CO2_Gt[1,:,:])

        prepped_DAC_CDR_nonCO2_variations = prep_data_bardata(DAC_CDR_nonCO2_Gt[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_stagnating[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_decreasing[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_rerouting[1,:,:],
                                                   DAC_CDR_othernonCO2_Gt[1,:,:])
        prepped_flying_CO2_emissions_variations = prep_data_bardata(flying_CO2_emissions[1, :, :],
                                                 flying_CO2_emissions_stagnating[1, :, :],
                                                 flying_CO2_emissions_decreasing[1, :, :],
                                                 flying_CO2_emissions_rerouting[1, :, :],
                                                 flying_CO2_emissions[1, :, :])
        prepped_flying_nonCO2_emissions_variations = prep_data_bardata(flying_nonCO2_emissions[1, :, :],
                                                 flying_nonCO2_emissions_stagnating[1, :, :],
                                                 flying_nonCO2_emissions_decreasing[1, :, :],
                                                 flying_nonCO2_emissions_rerouting[1, :, :],
                                                 flying_othernonCO2_emissions[1, :, :])
        prepped_indirect_emissions_variation = prep_data_bardata(totalIndirectEmissions[1, :, :] - Delta_totalIndirectEmissions[1, :, :],
                                                 totalIndirectEmissions_stagnating[1, :, :] - Delta_totalIndirectEmissions_stagnating[1, :, :],
                                                 totalIndirectEmissions_decreasing[1, :, :] - Delta_totalIndirectEmissions_decreasing[1, :, :],
                                                 totalIndirectEmissions_rerouting[1, :, :] - Delta_totalIndirectEmissions_rerouting[1, :, :],
                                                 totalIndirectEmissions_noCC[1, :, :] - Delta_totalIndirectEmissions_noCC[1, :, :])

        prepped_flying_CC_avoided = prep_data_bardata(
            np.zeros((2, 41)),
            np.zeros((2, 41)),
            np.zeros((2, 41)),
            flying_CC_emissions[1, :, :] - (
                        flying_nonCO2_emissions_rerouting[1, :, :] - flying_othernonCO2_emissions[1, :, :]),
            np.zeros((2, 41))
        )


        # Min values
        prepped_DAC_CDR_nonCO2_variations_min = prep_data_bardata(DAC_CDR_nonCO2_Gt_min[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_min_stagnating[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_min_decreasing[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_min_rerouting[1,:,:],
                                                   DAC_CDR_othernonCO2_Gt_min[1,:,:])
        prepped_flying_nonCO2_emissions_variations_min = prep_data_bardata(flying_nonCO2_emissions_min[1, :, :],
                                                 flying_nonCO2_emissions_min_stagnating[1, :, :],
                                                 flying_nonCO2_emissions_min_decreasing[1, :, :],
                                                 flying_nonCO2_emissions_min_rerouting[1, :, :],
                                                 flying_othernonCO2_emissions_min[1, :, :])
        prepped_indirect_emissions_variation_min = prep_data_bardata(totalIndirectEmissions_min[1, :, :] - Delta_totalIndirectEmissions_min[1, :, :],
                                                 totalIndirectEmissions_min_stagnating[1, :, :] - Delta_totalIndirectEmissions_min_stagnating[1, :, :],
                                                 totalIndirectEmissions_min_decreasing[1, :, :] - Delta_totalIndirectEmissions_min_decreasing[1, :, :],
                                                 totalIndirectEmissions_min_rerouting[1, :, :] - Delta_totalIndirectEmissions_min_rerouting[1, :, :],
                                                 totalIndirectEmissions_min_noCC[1, :, :] - Delta_totalIndirectEmissions_min_noCC[1, :, :])

        prepped_flying_CC_avoided_min = prep_data_bardata(
            np.zeros((2, 41)),
            np.zeros((2, 41)),
            np.zeros((2, 41)),
            flying_CC_emissions_min[1, :, :] - (
                        flying_nonCO2_emissions_min_rerouting[1, :, :] - flying_othernonCO2_emissions_min[1, :, :]),
            np.zeros((2, 41))
        )

        # Max values
        prepped_DAC_CDR_nonCO2_variations_max = prep_data_bardata(DAC_CDR_nonCO2_Gt_max[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_max_stagnating[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_max_decreasing[1,:,:],
                                                   DAC_CDR_nonCO2_Gt_max_rerouting[1,:,:],
                                                   DAC_CDR_othernonCO2_Gt_max[1,:,:])
        prepped_flying_nonCO2_emissions_variations_max = prep_data_bardata(flying_nonCO2_emissions_max[1, :, :],
                                                 flying_nonCO2_emissions_max_stagnating[1, :, :],
                                                 flying_nonCO2_emissions_max_decreasing[1, :, :],
                                                 flying_nonCO2_emissions_max_rerouting[1, :, :],
                                                 flying_othernonCO2_emissions_max[1, :, :])
        prepped_indirect_emissions_variation_max = prep_data_bardata(totalIndirectEmissions_max[1, :, :] - Delta_totalIndirectEmissions_max[1, :, :],
                                                 totalIndirectEmissions_max_stagnating[1, :, :] - Delta_totalIndirectEmissions_max_stagnating[1, :, :],
                                                 totalIndirectEmissions_max_decreasing[1, :, :] - Delta_totalIndirectEmissions_max_decreasing[1, :, :],
                                                 totalIndirectEmissions_max_rerouting[1, :, :] - Delta_totalIndirectEmissions_max_rerouting[1, :, :],
                                                 totalIndirectEmissions_max_noCC[1, :, :] - Delta_totalIndirectEmissions_max_noCC[1, :, :])

        prepped_flying_CC_avoided_max = prep_data_bardata(
            np.zeros((2, 41)),
            np.zeros((2, 41)),
            np.zeros((2, 41)),
            flying_CC_emissions_max[1, :, :] - (
                        flying_nonCO2_emissions_max_rerouting[1, :, :] - flying_othernonCO2_emissions_max[1, :, :]),
            np.zeros((2, 41))
        )

        plot_stacked_bars_four_scenarios(-prepped_DAC_CDR_nonCO2_variations, -prepped_DAC_CDR_CO2_variations, prepped_flying_nonCO2_emissions_variations,
                                         YEAR,
                                         'CDR non-CO$_2$', 'CDR CO$_2$', 'Flight non-CO$_2$',
                                         prepped_flying_CO2_emissions_variations, 'Flight CO$_2$',
                                         prepped_indirect_emissions_variation, 'Indirect emissions',
                                         np6=None, yaxis6=None,
                                         np7 = prepped_flying_CC_avoided, yaxis7='Avoided contrails',
                                         scenario='Climate neutrality only',
                                         what="Emissions", palette=rgb_colors[1:],
                                         np1_min=-prepped_DAC_CDR_nonCO2_variations_min,
                                         np1_max=-prepped_DAC_CDR_nonCO2_variations_max,
                                         np3_min=prepped_flying_nonCO2_emissions_variations_min,
                                         np3_max=prepped_flying_nonCO2_emissions_variations_max,
                                         np5_min=prepped_indirect_emissions_variation_min,
                                         np5_max=prepped_indirect_emissions_variation_max,
                                         np7_min=prepped_flying_CC_avoided_min,
                                         np7_max= prepped_flying_CC_avoided_max,
                                         scenario1 = 'DACCU \n climate \n neutrality',
                                         scenario2 = 'DACCS \n climate \n neutrality',
                                         scenario3 = 'DACCU \n stagnating',
                                         scenario4 = 'DACCS \n stagnating',
                                         scenario5 = 'DACCU \n decreasing',
                                         scenario6 = 'DACCS \n decreasing',
                                         scenario7='DACCU \n rerouting',
                                         scenario8='DACCS \n rerouting',
                                         scenario9='DACCU \n ignoring CC',
                                         scenario10='DACCS \n ignoring CC',
                                         BAU_cost=np.zeros((len(flying_CO2_emissions[0, 0]))),
                                         fmt_choice='o', error_label='Net emissions', BAU_label=None)

        # BIG cost plot with all climate neutrality options
        prepped_DACCU_CAPEX = prep_data_bardata(
            (total_DACCU_production_cost[1,:,:] - total_DACCU_electricity_cost[1,:,:] - total_DACCU_heat_cost[1,:,:]) / 10 ** 9,
            (total_DACCU_production_cost_stagnating[1, :, :] - total_DACCU_electricity_cost_stagnating[1, :, :] - total_DACCU_heat_cost_stagnating[1, :,
                                                                                            :]) / 10 ** 9,
            (total_DACCU_production_cost_decreasing[1, :, :] - total_DACCU_electricity_cost_decreasing[1, :, :] - total_DACCU_heat_cost_decreasing[1, :,
                                                                                            :]) / 10 ** 9,
            (
                        total_DACCU_production_cost_rerouting[1,:,:] - total_DACCU_electricity_cost_rerouting[1,:,:] - total_DACCU_heat_cost_rerouting[1,:,:]) / 10 ** 9,
            (
                        total_DACCU_production_cost_noCC[1,:,:] - total_DACCU_electricity_cost_noCC[1,:,:] - total_DACCU_heat_cost_noCC[1,:,:]) / 10 ** 9)

        prepped_DACCS_CAPEX = prep_data_bardata(
            (yearly_DAC_CDR_CO2_Cost[1,:,:] + yearly_DAC_CDR_nonCO2_Cost[1,:,:] + yearly_DAC_DeltaEmissions_Cost[1,:,:]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_stagnating[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_stagnating[1, :, :] + yearly_DAC_DeltaEmissions_Cost_stagnating[1,
                                                                                      :, :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_decreasing[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_decreasing[1, :, :] + yearly_DAC_DeltaEmissions_Cost_decreasing[1,
                                                                                      :, :]) / 10 ** 9,
            (
                        yearly_DAC_CDR_CO2_Cost_rerouting[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_rerouting[1, :, :] + yearly_DAC_DeltaEmissions_Cost_rerouting[1, :, :]) / 10 ** 9,
            (
                        yearly_DAC_CDR_CO2_Cost_noCC[1, :, :] + yearly_DAC_CDR_othernonCO2_Cost_noCC[1, :, :] + yearly_DAC_DeltaEmissions_Cost_noCC[1, :, :]) / 10 ** 9)

        prepped_cost_electricity = prep_data_bardata(
            finalcost_electricity[1, :, :] / 10 ** 9,
            finalcost_electricity_stagnating[1, :, :] / 10 ** 9,
            finalcost_electricity_decreasing[1, :, :] / 10 ** 9,
            finalcost_electricity_rerouting[1, :, :] / 10 ** 9,
            finalcost_electricity_noCC[1, :, :] / 10 ** 9
        )

        prepped_cost_fossilfuel = prep_data_bardata(
            finalcost_fossilfuel[1, :, :] / 10 ** 9,
            finalcost_fossilfuel_stagnating[1, :, :] / 10 ** 9,
            finalcost_fossilfuel_decreasing[1, :, :] / 10 ** 9,
            finalcost_fossilfuel_rerouting[1, :, :] / 10 ** 9,
            finalcost_fossilfuel_noCC[1, :, :] / 10 ** 9
        )

        prepped_cost_transport_storageCO2 = prep_data_bardata(
            finalcost_transport_storageCO2[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_stagnating[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_decreasing[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_rerouting[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_noCC[1, :, :] / 10 ** 9
        )

        prepped_cost_heat = prep_data_bardata(
            finalcost_heat[1, :, :] / 10 ** 9,
            finalcost_heat_stagnating[1, :, :] / 10 ** 9,
            finalcost_heat_decreasing[1, :, :] / 10 ** 9,
            finalcost_heat_rerouting[1, :, :] / 10 ** 9,
            finalcost_heat_noCC[1, :, :] / 10 ** 9
        )

        # Variables with _min suffix
        prepped_DACCS_CAPEX_min = prep_data_bardata(
            (yearly_DAC_CDR_CO2_Cost[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min[1, :,
                                                :] + yearly_DAC_DeltaEmissions_Cost_min[1, :, :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_stagnating[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min_stagnating[1, :,
                                                           :] + yearly_DAC_DeltaEmissions_Cost_min_stagnating[1, :,
                                                                :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_decreasing[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min_decreasing[1, :,
                                                           :] + yearly_DAC_DeltaEmissions_Cost_min_decreasing[1, :,
                                                                :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_rerouting[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min_rerouting[1, :,
                                                          :] + yearly_DAC_DeltaEmissions_Cost_min_rerouting[1, :,
                                                               :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_noCC[1, :, :] + yearly_DAC_CDR_othernonCO2_Cost_min_noCC[1, :,
                                                     :] + yearly_DAC_DeltaEmissions_Cost_min_noCC[1, :, :]) / 10 ** 9
        )

        prepped_cost_electricity_min = prep_data_bardata(
            finalcost_electricity_min[1, :, :] / 10 ** 9,
            finalcost_electricity_min_stagnating[1, :, :] / 10 ** 9,
            finalcost_electricity_min_decreasing[1, :, :] / 10 ** 9,
            finalcost_electricity_min_rerouting[1, :, :] / 10 ** 9,
            finalcost_electricity_min_noCC[1, :, :] / 10 ** 9
        )

        prepped_cost_transport_storageCO2_min = prep_data_bardata(
            finalcost_transport_storageCO2_min[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_min_stagnating[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_min_decreasing[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_min_rerouting[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_min_noCC[1, :, :] / 10 ** 9
        )

        prepped_cost_heat_min = prep_data_bardata(
            finalcost_heat_min[1, :, :] / 10 ** 9,
            finalcost_heat_min_stagnating[1, :, :] / 10 ** 9,
            finalcost_heat_min_decreasing[1, :, :] / 10 ** 9,
            finalcost_heat_min_rerouting[1, :, :] / 10 ** 9,
            finalcost_heat_min_noCC[1, :, :] / 10 ** 9
        )

        # Variables with _max suffix
        prepped_DACCS_CAPEX_max = prep_data_bardata(
            (yearly_DAC_CDR_CO2_Cost[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max[1, :,
                                                :] + yearly_DAC_DeltaEmissions_Cost_max[1, :, :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_stagnating[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max_stagnating[1, :,
                                                           :] + yearly_DAC_DeltaEmissions_Cost_max_stagnating[1, :,
                                                                :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_decreasing[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max_decreasing[1, :,
                                                           :] + yearly_DAC_DeltaEmissions_Cost_max_decreasing[1, :,
                                                                :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_rerouting[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max_rerouting[1, :,
                                                          :] + yearly_DAC_DeltaEmissions_Cost_max_rerouting[1, :,
                                                               :]) / 10 ** 9,
            (yearly_DAC_CDR_CO2_Cost_noCC[1, :, :] + yearly_DAC_CDR_othernonCO2_Cost_max_noCC[1, :,
                                                     :] + yearly_DAC_DeltaEmissions_Cost_max_noCC[1, :, :]) / 10 ** 9
        )

        prepped_cost_electricity_max = prep_data_bardata(
            finalcost_electricity_max[1, :, :] / 10 ** 9,
            finalcost_electricity_max_stagnating[1, :, :] / 10 ** 9,
            finalcost_electricity_max_decreasing[1, :, :] / 10 ** 9,
            finalcost_electricity_max_rerouting[1, :, :] / 10 ** 9,
            finalcost_electricity_max_noCC[1, :, :] / 10 ** 9
        )

        prepped_cost_transport_storageCO2_max = prep_data_bardata(
            finalcost_transport_storageCO2_max[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_max_stagnating[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_max_decreasing[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_max_rerouting[1, :, :] / 10 ** 9,
            finalcost_transport_storageCO2_max_noCC[1, :, :] / 10 ** 9
        )

        prepped_cost_heat_max = prep_data_bardata(
            finalcost_heat_max[1, :, :] / 10 ** 9,
            finalcost_heat_max_stagnating[1, :, :] / 10 ** 9,
            finalcost_heat_max_decreasing[1, :, :] / 10 ** 9,
            finalcost_heat_max_rerouting[1, :, :] / 10 ** 9,
            finalcost_heat_max_noCC[1, :, :] / 10 ** 9
        )



        plot_stacked_bars_four_scenarios(prepped_DACCU_CAPEX, prepped_DACCS_CAPEX, prepped_cost_electricity,
                                         YEAR,
                                         'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                         prepped_cost_fossilfuel, 'Fossil kerosene',
                                         prepped_cost_transport_storageCO2, 'CO$_2$ transport and storage',
                                         prepped_cost_heat, 'Heat',
                                         scenario='Climate neutrality only',
                                         what="Total cost", palette=PALETTE,
                                         np2_min=prepped_DACCS_CAPEX_min,
                                         np2_max=prepped_DACCS_CAPEX_max,
                                         np3_min=prepped_cost_electricity_min,
                                         np3_max=prepped_cost_electricity_max,
                                         np5_min=prepped_cost_transport_storageCO2_min,
                                         np5_max=prepped_cost_transport_storageCO2_max,
                                         np6_min=prepped_cost_heat_min,
                                         np6_max=prepped_cost_heat_max,
                                         scenario1='DACCU \n climate \n neutrality',
                                         scenario2='DACCS \n climate \n neutrality',
                                         scenario3='DACCU \n stagnating',
                                         scenario4='DACCS \n stagnating',
                                         scenario5='DACCU \n decreasing',
                                         scenario6='DACCS \n decreasing',
                                         scenario7='DACCU \n rerouting',
                                         scenario8='DACCS \n rerouting',
                                         scenario9='DACCU \n ignoring CC',
                                         scenario10='DACCS \n ignoring CC',
                                         BAU_cost=finalcost_BAU / 10 ** 9,
                                         )

        # ALTERNATIVE FIGURE 7
        prepped_DACCU_CAPEX_per_liter = prep_data_bardata(
            calculate_cost_per_liter(
                total_DACCU_production_cost[1, :, :] - total_DACCU_electricity_cost[1, :, :] - total_DACCU_heat_cost[1,
                                                                                               :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(
                total_DACCU_production_cost_stagnating[1, :, :] - total_DACCU_electricity_cost_stagnating[1, :,
                                                                  :] - total_DACCU_heat_cost_stagnating[1, :, :],
                FF_volumes_Tg_stagnating[1, :, :], DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(
                total_DACCU_production_cost_decreasing[1, :, :] - total_DACCU_electricity_cost_decreasing[1, :,
                                                                  :] - total_DACCU_heat_cost_decreasing[1, :, :],
                FF_volumes_Tg_decreasing[1, :, :], DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(
                total_DACCU_production_cost_rerouting[1, :, :] - total_DACCU_electricity_cost_rerouting[1, :,
                                                                 :] - total_DACCU_heat_cost_rerouting[1, :, :],
                FF_volumes_Tg_rerouting[1, :, :], DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(
                total_DACCU_production_cost_noCC[1, :, :] - total_DACCU_electricity_cost_noCC[1, :,
                                                            :] - total_DACCU_heat_cost_noCC[1, :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :])
        )

        prepped_DACCS_CAPEX_per_liter = prep_data_bardata(
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost[1, :, :] + yearly_DAC_CDR_nonCO2_Cost[1, :, :] + yearly_DAC_DeltaEmissions_Cost[
                                                                                         1, :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_stagnating[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_stagnating[1, :,
                                                              :] + yearly_DAC_DeltaEmissions_Cost_stagnating[1, :, :],
                FF_volumes_Tg_stagnating[1, :, :], DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_decreasing[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_decreasing[1, :,
                                                              :] + yearly_DAC_DeltaEmissions_Cost_decreasing[1, :, :],
                FF_volumes_Tg_decreasing[1, :, :], DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_rerouting[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_rerouting[1, :,
                                                             :] + yearly_DAC_DeltaEmissions_Cost_rerouting[1, :, :],
                FF_volumes_Tg_rerouting[1, :, :], DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_noCC[1, :, :] + yearly_DAC_CDR_othernonCO2_Cost_noCC[1, :,
                                                        :] + yearly_DAC_DeltaEmissions_Cost_noCC[1, :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_electricity_per_liter = prep_data_bardata(
            calculate_cost_per_liter(finalcost_electricity[1, :, :], FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_stagnating[1, :, :], FF_volumes_Tg_stagnating[1, :, :],
                                     DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_decreasing[1, :, :], FF_volumes_Tg_decreasing[1, :, :],
                                     DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_rerouting[1, :, :], FF_volumes_Tg_rerouting[1, :, :],
                                     DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_fossilfuel_per_liter = prep_data_bardata(
            calculate_cost_per_liter(finalcost_fossilfuel[1, :, :], FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_fossilfuel_stagnating[1, :, :], FF_volumes_Tg_stagnating[1, :, :],
                                     DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_fossilfuel_decreasing[1, :, :], FF_volumes_Tg_decreasing[1, :, :],
                                     DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_fossilfuel_rerouting[1, :, :], FF_volumes_Tg_rerouting[1, :, :],
                                     DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_fossilfuel_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_transport_storageCO2_per_liter = prep_data_bardata(
            calculate_cost_per_liter(finalcost_transport_storageCO2[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_stagnating[1, :, :],
                                     FF_volumes_Tg_stagnating[1, :, :], DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_decreasing[1, :, :],
                                     FF_volumes_Tg_decreasing[1, :, :], DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_rerouting[1, :, :],
                                     FF_volumes_Tg_rerouting[1, :, :], DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_heat_per_liter = prep_data_bardata(
            calculate_cost_per_liter(finalcost_heat[1, :, :], FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_stagnating[1, :, :], FF_volumes_Tg_stagnating[1, :, :],
                                     DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_decreasing[1, :, :], FF_volumes_Tg_decreasing[1, :, :],
                                     DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_rerouting[1, :, :], FF_volumes_Tg_rerouting[1, :, :],
                                     DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_noCC[1, :, :], FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :])
        )

        prepped_DACCS_CAPEX_per_liter_min = prep_data_bardata(
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min[1, :,
                                                   :] + yearly_DAC_DeltaEmissions_Cost_min[1, :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_stagnating[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min_stagnating[1, :,
                                                              :] + yearly_DAC_DeltaEmissions_Cost_min_stagnating[1, :,
                                                                   :],
                FF_volumes_Tg_stagnating[1, :, :], DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_decreasing[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min_decreasing[1, :,
                                                              :] + yearly_DAC_DeltaEmissions_Cost_min_decreasing[1, :,
                                                                   :],
                FF_volumes_Tg_decreasing[1, :, :], DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_rerouting[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_min_rerouting[1, :,
                                                             :] + yearly_DAC_DeltaEmissions_Cost_min_rerouting[1, :, :],
                FF_volumes_Tg_rerouting[1, :, :], DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_noCC[1, :, :] + yearly_DAC_CDR_othernonCO2_Cost_min_noCC[1, :,
                                                        :] + yearly_DAC_DeltaEmissions_Cost_min_noCC[1, :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_electricity_per_liter_min = prep_data_bardata(
            calculate_cost_per_liter(finalcost_electricity_min[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_min_stagnating[1, :, :], FF_volumes_Tg_stagnating[1, :, :],
                                     DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_min_decreasing[1, :, :], FF_volumes_Tg_decreasing[1, :, :],
                                     DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_min_rerouting[1, :, :], FF_volumes_Tg_rerouting[1, :, :],
                                     DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_min_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_transport_storageCO2_per_liter_min = prep_data_bardata(
            calculate_cost_per_liter(finalcost_transport_storageCO2_min[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_min_stagnating[1, :, :],
                                     FF_volumes_Tg_stagnating[1, :, :], DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_min_decreasing[1, :, :],
                                     FF_volumes_Tg_decreasing[1, :, :], DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_min_rerouting[1, :, :],
                                     FF_volumes_Tg_rerouting[1, :, :], DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_min_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_heat_per_liter_min = prep_data_bardata(
            calculate_cost_per_liter(finalcost_heat_min[1, :, :], FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_min_stagnating[1, :, :], FF_volumes_Tg_stagnating[1, :, :],
                                     DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_min_decreasing[1, :, :], FF_volumes_Tg_decreasing[1, :, :],
                                     DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_min_rerouting[1, :, :], FF_volumes_Tg_rerouting[1, :, :],
                                     DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_min_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_DACCS_CAPEX_per_liter_max = prep_data_bardata(
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max[1, :,
                                                   :] + yearly_DAC_DeltaEmissions_Cost_max[1, :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_stagnating[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max_stagnating[1, :,
                                                              :] + yearly_DAC_DeltaEmissions_Cost_max_stagnating[1, :,
                                                                   :],
                FF_volumes_Tg_stagnating[1, :, :], DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_decreasing[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max_decreasing[1, :,
                                                              :] + yearly_DAC_DeltaEmissions_Cost_max_decreasing[1, :,
                                                                   :],
                FF_volumes_Tg_decreasing[1, :, :], DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_rerouting[1, :, :] + yearly_DAC_CDR_nonCO2_Cost_max_rerouting[1, :,
                                                             :] + yearly_DAC_DeltaEmissions_Cost_max_rerouting[1, :, :],
                FF_volumes_Tg_rerouting[1, :, :], DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(
                yearly_DAC_CDR_CO2_Cost_noCC[1, :, :] + yearly_DAC_CDR_othernonCO2_Cost_max_noCC[1, :,
                                                        :] + yearly_DAC_DeltaEmissions_Cost_max_noCC[1, :, :],
                FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_electricity_per_liter_max = prep_data_bardata(
            calculate_cost_per_liter(finalcost_electricity_max[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_max_stagnating[1, :, :], FF_volumes_Tg_stagnating[1, :, :],
                                     DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_max_decreasing[1, :, :], FF_volumes_Tg_decreasing[1, :, :],
                                     DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_max_rerouting[1, :, :], FF_volumes_Tg_rerouting[1, :, :],
                                     DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_electricity_max_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_transport_storageCO2_per_liter_max = prep_data_bardata(
            calculate_cost_per_liter(finalcost_transport_storageCO2_max[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_max_stagnating[1, :, :],
                                     FF_volumes_Tg_stagnating[1, :, :], DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_max_decreasing[1, :, :],
                                     FF_volumes_Tg_decreasing[1, :, :], DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_max_rerouting[1, :, :],
                                     FF_volumes_Tg_rerouting[1, :, :], DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_transport_storageCO2_max_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        prepped_cost_heat_per_liter_max = prep_data_bardata(
            calculate_cost_per_liter(finalcost_heat_max[1, :, :], FF_volumes_Tg[1, :, :], DACCU_volumes_Tg[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_max_stagnating[1, :, :], FF_volumes_Tg_stagnating[1, :, :],
                                     DACCU_volumes_Tg_stagnating[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_max_decreasing[1, :, :], FF_volumes_Tg_decreasing[1, :, :],
                                     DACCU_volumes_Tg_decreasing[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_max_rerouting[1, :, :], FF_volumes_Tg_rerouting[1, :, :],
                                     DACCU_volumes_Tg_rerouting[1, :, :]),
            calculate_cost_per_liter(finalcost_heat_max_noCC[1, :, :], FF_volumes_Tg[1, :, :],
                                     DACCU_volumes_Tg[1, :, :])
        )

        plot_stacked_bars_four_scenarios(prepped_DACCU_CAPEX_per_liter, prepped_DACCS_CAPEX_per_liter, prepped_cost_electricity_per_liter,
                                         YEAR,
                                         'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                         prepped_cost_fossilfuel_per_liter, 'Fossil kerosene',
                                         prepped_cost_transport_storageCO2_per_liter, 'CO$_2$ transport and storage',
                                         prepped_cost_heat_per_liter, 'Heat',
                                         scenario='Climate neutrality only',
                                         what="Cost per liter fuel", palette=PALETTE,
                                         np2_min=prepped_DACCS_CAPEX_per_liter_min,
                                         np2_max=prepped_DACCS_CAPEX_per_liter_max,
                                         np3_min=prepped_cost_electricity_per_liter_min,
                                         np3_max=prepped_cost_electricity_per_liter_max,
                                         np5_min=prepped_cost_transport_storageCO2_per_liter_min,
                                         np5_max=prepped_cost_transport_storageCO2_per_liter_max,
                                         np6_min=prepped_cost_heat_per_liter_min,
                                         np6_max=prepped_cost_heat_per_liter_max,
                                         scenario1='DACCU \n climate \n neutrality',
                                         scenario2='DACCS \n climate \n neutrality',
                                         scenario3='DACCU \n stagnating',
                                         scenario4='DACCS \n stagnating',
                                         scenario5='DACCU \n decreasing',
                                         scenario6='DACCS \n decreasing',
                                         scenario7='DACCU \n rerouting',
                                         scenario8='DACCS \n rerouting',
                                         scenario9='DACCU \n ignoring CC',
                                         scenario10='DACCS \n ignoring CC',
                                         BAU_cost=finalcost_BAU/(FF_volumes_Tg[1,1,:]*10**9*1.22),
                                         )


    if plotting_SI_figures is True:
        plot_time_series_two_scenarios(ERF_total.SO4, ERF_total.CC, ERF_total.netNOX,
                                       YEAR,
                                       'SO$_4$', 'Contrails', 'NO$_x$',
                                       ERF_total.H2O, 'H$_2$O',
                                       ERF_total.BC, 'Soot',
                                       ERF_total.CO2, 'CO$_2$',
                                       low_lim1=-30, low_lim2=-30,
                                       up_lim1=350, up_lim2=350,
                                       what="ERF", ylabel="Effective Radiative Forcing (Wm$^2$)",
                                       scenario1='DACCU \n climate \n neutrality',
                                       scenario2='DACCS \n climate \n neutrality',
                                       palette='icefire')

        # PLOT the DAC rates
        plot_time_series_two_scenarios(DAC_DACCU_Gt, DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt, YEAR,
                                       'DAC for CO$_2$ feedstock', 'DAC CDR CO$_2$', 'DAC CDR non-CO$_2$',
                                       DAC_Diesel_Gt, "DAC for Diesel", low_lim1=-0.2, up_lim1=10.8, low_lim2=-0.2,
                                       up_lim2=10.8,
                                       scenario1='DACCU \n carbon \n neutrality',
                                       scenario2='DACCS \n carbon \n neutrality',
                                       scenario3='DACCU \n climate \n neutrality',
                                       scenario4='DACCS \n climate \n neutrality',
                                       what="DAC_Gt_withDiesel", ylabel="DAC rates [GtCO$_2$/year]", palette=PALETTE,
                                       np3_min=DAC_CDR_nonCO2_Gt_min, np3_max=DAC_CDR_nonCO2_Gt_max)
        plot_time_series_two_scenarios(DAC_DACCU_Gt, DAC_CDR_CO2_Gt, DAC_CDR_nonCO2_Gt, YEAR,
                                       'DAC for CO$_2$ feedstock', 'DAC CDR CO$_2$', 'DAC CDR non-CO$_2$',
                                       low_lim1=-0.2, up_lim1=9, low_lim2=-0.2, up_lim2=9,
                                       scenario1='DACCU \n carbon \n neutrality',
                                       scenario2='DACCS \n carbon \n neutrality',
                                       scenario3='DACCU \n climate \n neutrality',
                                       scenario4='DACCS \n climate \n neutrality',
                                       what="DAC_Gt_withoutDiesel", ylabel="DAC rates [GtCO$_2$/year]", palette=PALETTE,
                                       np3_min=DAC_CDR_nonCO2_Gt_min, np3_max=DAC_CDR_nonCO2_Gt_max)

        # PLOT emissions by sources
        plot_time_series_two_scenarios(flying_CO2_emissions, flying_nonCO2_emissions, yearlyWTT, YEAR,
                                       'Flying CO$_2$', 'Flying non-CO$_2$', 'Well-to-tank',
                                       total_DACCU_MaterialFootprint, 'Material footprint',
                                       total_DACCU_ElectricitryFootprint, 'Electricity footprint', low_lim1=-0.2,
                                       up_lim1=11.7,
                                       scenario1='DACCU \n climate \n neutrality',
                                       scenario2='DACCS \n climate \n neutrality',
                                       what="Emissions", ylabel="Emissions [GtCO$_2$eq/year]", palette=PALETTE,
                                       np2_min=flying_nonCO2_emissions_min, np2_max=flying_nonCO2_emissions_max)

        # Plot electricity
        plot_time_series_two_scenarios(DAC_DACCU_MWh, H2_DACCU_MWh, CO_DACCU_MWh + FT_DACCU_MWh, YEAR,
                                       'DAC for CO$_2$ feedstock', 'H$_2$ production', 'CO reduction & FT',
                                       DAC_CDR_CO2_MWh, 'DACCS (CO$_2$)', DAC_CDR_nonCO2_MWh,
                                       'DACCS (non-CO$_2$)',
                                       (DAC_Diesel_MWh + CO_Diesel_MWh + H2_Diesel_MWh + FT_Diesel_MWh),
                                       "Diesel production",
                                       (
                                                   CO_DACCU_MWh_heat + DAC_DACCU_MWh_heat + DAC_CDR_CO2_MWhth + DAC_CDR_nonCO2_MWhth),
                                       "Heat",
                                       what="Electricity_withDiesel", ylabel="Energy [TWh/year]", low_lim1=-0.1,
                                       up_lim1=3.3 * 10 ** 10,
                                       low_lim2=-0.1, up_lim2=3.5 * 10 ** 10,
                                       scenario1='DACCU \n carbon \n neutrality',
                                       scenario2='DACCS \n carbon \n neutrality',
                                       scenario3='DACCU \n climate \n neutrality',
                                       scenario4='DACCS \n climate \n neutrality',
                                       palette=PALETTE,
                                       np6_min=DAC_CDR_nonCO2_MWh_min + DAC_CDR_nonCO2_MWhth_min,
                                       np6_max=+DAC_CDR_nonCO2_MWhth_max + DAC_CDR_nonCO2_MWhth_max)

        # Plot electricity without Diesel
        plot_time_series_two_scenarios(DAC_DACCU_MWh / 10 ** 6, H2_DACCU_MWh / 10 ** 6,
                                       (CO_DACCU_MWh + FT_DACCU_MWh) / 10 ** 6, YEAR,
                                       'DAC for CO$_2$ feedstock', 'H$_2$ production', 'CO reduction & FT',
                                       (DAC_CDR_CO2_MWh + DAC_CDR_nonCO2_MWh) / 10 ** 6, 'DACCS',
                                       CO_DACCU_MWh_heat / 10 ** 6, 'CO reduction heat',
                                       (DAC_DACCU_MWh_heat + DAC_CDR_CO2_MWhth + DAC_CDR_nonCO2_MWhth) / 10 ** 6,
                                       "DAC heat",
                                       what="Electricity_withoutDiesel", ylabel="Energy [TWh/year]", low_lim1=-0.1,
                                       up_lim1=2.1 * 10 ** 4,
                                       low_lim2=-0.1, up_lim2=2.1 * 10 ** 4,
                                       scenario1='DACCU \n carbon \n neutrality',
                                       scenario2='DACCS \n carbon \n neutrality',
                                       scenario3='DACCU \n climate \n neutrality',
                                       scenario4='DACCS \n climate \n neutrality',
                                       palette=PALETTE,
                                       np4_min=DAC_CDR_nonCO2_MWh_min / 10 ** 6,
                                       np4_max=DAC_CDR_nonCO2_MWh_max / 10 ** 6,
                                       np6_min=(DAC_CDR_nonCO2_MWhth_min) / 10 ** 6,
                                       np6_max=(DAC_CDR_nonCO2_MWhth_max) / 10 ** 6
                                       )

        # Plot costs
        plot_time_series_two_scenarios(total_DACCU_production_cost - total_DACCU_electricity_cost,
                                       yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost + yearly_DAC_DeltaEmissions_Cost
                                       , finalcost_electricity,
                                       YEAR,
                                       'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                       finalcost_fossilfuel, 'Fossil kerosene',
                                       finalcost_transport_storageCO2, 'CO$_2$ transport and storage',
                                       finalcost_heat, 'Heat',
                                       what="Cost", ylabel="Cost [€/year]", low_lim1=-0.1, up_lim1=1.6 * 10 ** 12,
                                       low_lim2=-0.1, up_lim2=1.6 * 10 ** 12,
                                       scenario1='DACCU \n carbon \n neutrality',
                                       scenario2='DACCS \n carbon \n neutrality',
                                       scenario3='DACCU \n climate \n neutrality',
                                       scenario4='DACCS \n climate \n neutrality',
                                       palette=PALETTE,
                                       np2_min=yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_min,
                                       np2_max=yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_max,
                                       np3_min=finalcost_electricity_min, np3_max=finalcost_electricity_max,
                                       np5_min=finalcost_transport_storageCO2_min,
                                       np5_max=finalcost_transport_storageCO2_max,
                                       np6_min=finalcost_heat_min, np6_max=finalcost_heat_max)

        # Plot costs
        # Cumulative
        plot_stacked_bars_two_scenarios(total_DACCU_CAPEX_cost / 10 ** 12, total_DACCS_CAPEX_cost / 10 ** 12,
                                        finalcost_electricity / 10 ** 12,
                                        YEAR,
                                        'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                        finalcost_fossilfuel / 10 ** 12, 'Fossil kerosene',
                                        finalcost_transport_storageCO2 / 10 ** 12, 'CO$_2$ transport and storage',
                                        finalcost_heat / 10 ** 12, 'Heat', scenario='Carbon neutrality',
                                        what="Total cost", palette=PALETTE, cumulative=True,
                                        np2_min=total_DACCS_CAPEX_cost_min / 10 ** 12,
                                        np2_max=total_DACCS_CAPEX_cost_max / 10 ** 12,
                                        np3_min=finalcost_electricity_min / 10 ** 12,
                                        np3_max=finalcost_electricity_max / 10 ** 12,
                                        np5_min=finalcost_transport_storageCO2_min / 10 ** 12,
                                        np5_max=finalcost_transport_storageCO2_max / 10 ** 12,
                                        np6_min=finalcost_heat_min / 10 ** 12,
                                        np6_max=finalcost_heat_max / 10 ** 12,
                                        BAU_cost=finalcost_BAU / 10 ** 12)

        # Cumulative difference
        plot_stacked_bars_two_scenarios((finalcost - finalcost_BAU)/10**12,
                                        (finalcost_stagnating - finalcost_BAU_stagnating)/10**12,
                                        (finalcost_decreasing - finalcost_BAU_decreasing) / 10 ** 12,
                                        YEAR,
                                        'Default', '+0% demand', '-2% demand',
                                        (finalcost_rerouting - finalcost_BAU_rerouting) / 10 ** 12,
                                        'Rerouting',
                                        (finalcost_noCC - finalcost_BAU_noCC) / 10 ** 12, 'Ignoring contrails',
                                        scenario='Varying demand',
                                        what="Cost difference", palette=PALETTE, cumulative=True, stacked = False
                                        )

        # FIGURE 3 - price per ticket
        # cost per passenger by 2050 - here no uncertainty since it's already quite uncertain the main calculation
        plot_stacked_bars_two_scenarios(cost_neutrality_per_passenger_LN_NY, cost_neutrality_per_passenger_LN_Berlin,
                                        cost_neutrality_per_passenger_LN_Perth, 2030,
                                        'London-New York', 'London-Berlin', 'London-Perth',
                                        what='Cost neutrality per flight', scenario='Carbon neutrality',
                                        palette=PALETTE, stacked=False)
        # change in cost per passenger  by YEAR
        plot_stacked_bars_two_scenarios(increase_neutrality_flight_price_LN_NY * 100,
                                        increase_neutrality_flight_price_LN_Berlin * 100,
                                        increase_neutrality_flight_price_LN_Perth * 100, 2030,
                                        'London-New York', 'London-Berlin', 'London-Perth',
                                        what='Change flight price', scenario='Carbon neutrality', palette=PALETTE,
                                        stacked=False)

        finalcost_conf1, _, finalcost_electricity_conf1, _, _, finalcost_heat_conf1, \
            total_DACCU_electricity_cost_conf1, total_DACCU_production_cost_conf1, total_yearly_H2_COST_conf1, total_yearly_CO_COST_conf1, \
            _, _, _, _, \
            _, totalIndirectEmissions_conf1, _, _, \
            totalNetEmissions_conf1, _, _, _, _, \
            DAC_CDR_CO2_MaterialFootprint_conf1, DAC_CDR_nonCO2_ElectricityFootprint_conf1, total_DAC_CDR_Footprint_conf1, \
            _, _, _, _, _, _, \
            _, _, _, _, H2_DACCU_Mt_conf1, H2_DACCU_MWh_conf1, \
            CO_DACCU_Mt_conf1, CO_DACCU_MWh_conf1, FT_DACCU_MWh_conf1 \
            = run_model_II(df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                           LEARNING_RATE_electrolysis,
                           LEARNING_RATE_CO, LEARNING_RATE_CO,
                           ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_q0_Gt_2020,
                           DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                           JETFUEL_ALLOCATION_SHARE, CC_EFFICACY, scenario='Both', configuration_PtL='AEC+RWGS+FT',
                           efficiency_increase=True)

        finalcost_conf2, _, finalcost_electricity_conf2, _, _, finalcost_heat_conf2, \
            total_DACCU_electricity_cost_conf2, total_DACCU_production_cost_conf2, total_yearly_H2_COST_conf2, total_yearly_CO_COST_conf2, \
            _, _, _, _, \
            _, totalIndirectEmissions_conf2, _, _, \
            totalNetEmissions_conf2, _, _, _, _, \
            DAC_CDR_CO2_MaterialFootprint_conf2, DAC_CDR_nonCO2_ElectricityFootprint_conf2, total_DAC_CDR_Footprint_conf2, \
            _, _, _, _, _, _, \
            _, _, _, _, H2_DACCU_Mt_conf2, H2_DACCU_MWh_conf2, \
            CO_DACCU_Mt_conf2, CO_DACCU_MWh_conf2, FT_DACCU_MWh_conf2 \
            = run_model_II(df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                           LEARNING_RATE_electrolysis,
                           LEARNING_RATE_CO, LEARNING_RATE_CO,
                           ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_q0_Gt_2020,
                           DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                           JETFUEL_ALLOCATION_SHARE, CC_EFFICACY, scenario='Both', configuration_PtL='PEM+RWGS+FT',
                           efficiency_increase=True)

        finalcost_conf3, _, finalcost_electricity_conf3, _, _, finalcost_heat_conf3, \
            total_DACCU_electricity_cost_conf3, total_DACCU_production_cost_conf3, total_yearly_H2_COST_conf3, total_yearly_CO_COST_conf3, \
            _, _, _, _, \
            _, totalIndirectEmissions_conf3, _, _, \
            totalNetEmissions_conf3, _, _, _, _, \
            DAC_CDR_CO2_MaterialFootprint_conf3, DAC_CDR_nonCO2_ElectricityFootprint_conf3, total_DAC_CDR_Footprint_conf3, \
            _, _, _, _, _, _, \
            _, _, _, _, H2_DACCU_Mt_conf3, H2_DACCU_MWh_conf3, \
            CO_DACCU_Mt_conf3, CO_DACCU_MWh_conf3, FT_DACCU_MWh_conf3 \
            = run_model_II(df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                           LEARNING_RATE_electrolysis,
                           LEARNING_RATE_CO, LEARNING_RATE_CO,
                           ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_q0_Gt_2020,
                           DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                           JETFUEL_ALLOCATION_SHARE, CC_EFFICACY, scenario='Both', configuration_PtL='AEC+El.CO2+FT',
                           efficiency_increase=True)

        finalcost_conf4, _, finalcost_electricity_conf4, _, _, finalcost_heat_conf4, \
            total_DACCU_electricity_cost_conf4, total_DACCU_production_cost_conf4, total_yearly_H2_COST_conf4, total_yearly_CO_COST_conf4, \
            _, _, _, _, \
            _, totalIndirectEmissions_conf4, _, _, \
            totalNetEmissions_conf4, _, _, _, _, \
            DAC_CDR_CO2_MaterialFootprint_conf4, DAC_CDR_nonCO2_ElectricityFootprint_conf4, total_DAC_CDR_Footprint_conf4, \
            _, _, _, _, _, _, \
            _, _, _, _, H2_DACCU_Mt_conf4, H2_DACCU_MWh_conf4, \
            CO_DACCU_Mt_conf4, CO_DACCU_MWh_conf4, FT_DACCU_MWh_conf4 \
            = run_model_II(df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                           LEARNING_RATE_electrolysis,
                           LEARNING_RATE_CO, LEARNING_RATE_CO,
                           ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_q0_Gt_2020,
                           DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                           JETFUEL_ALLOCATION_SHARE, CC_EFFICACY, scenario='Both', configuration_PtL='PEM+El.CO2+FT',
                           efficiency_increase=True)

        # change in cost per passenger  by 2050
        plot_stacked_bars_two_scenarios(finalcost_conf1, finalcost_conf2,
                                        finalcost_conf3, YEAR,
                                        'AEC+RWGS+FT', 'PEM+RWGS+FT', 'AEC+El.CO2+FT',
                                        finalcost_conf4, 'PEM+El.CO2+FT',
                                        what='Cost with different configurations', scenario='Carbon neutrality',
                                        palette=PALETTE, stacked=False)

        mean_finalcost = np.mean([finalcost_conf1, finalcost_conf2, finalcost_conf3, finalcost_conf4], axis=0)
        plot_single_variables_time_series(mean_finalcost, 2060, what='Cost',
                                          ylabel="Total costs (€/year)",
                                          scenario='Carbon neutrality',
                                          data_array_min=finalcost_conf1,
                                          data_array_max=finalcost_conf4)

        plot_stacked_bars_two_scenarios(finalcost_decreasing / 10 ** 9, finalcost_stagnating / 10 ** 9,
                                        finalcost / 10 ** 9, YEAR,
                                        '-2% demand', '+0% demand', '+2% demand',
                                        finalcost_histgrowth / 10 ** 9, '+4% demand',
                                        what='Total cost', scenario='varying demand',
                                        palette=PALETTE, stacked=False, BAU_cost=finalcost_BAU / 10 ** 9)

        # 2050
        # Decreasing Scenario
        plot_stacked_bars_two_scenarios((
                                                    total_DACCU_production_cost_decreasing - total_DACCU_electricity_cost_decreasing - total_DACCU_heat_cost_decreasing) / 10 ** 9,
                                        (
                                                    yearly_DAC_CDR_CO2_Cost_decreasing + yearly_DAC_CDR_nonCO2_Cost_decreasing + yearly_DAC_DeltaEmissions_Cost_decreasing) / 10 ** 9,
                                        finalcost_electricity_decreasing / 10 ** 9,
                                        YEAR,
                                        'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                        finalcost_fossilfuel_decreasing / 10 ** 9, 'Fossil kerosene',
                                        finalcost_transport_storageCO2_decreasing / 10 ** 9,
                                        'CO$_2$ transport and storage',
                                        finalcost_heat_decreasing / 10 ** 9, 'Heat', scenario='decreasing demand',
                                        what="Total cost", palette=PALETTE,
                                        np2_min=(
                                                            yearly_DAC_CDR_CO2_Cost_decreasing + yearly_DAC_CDR_nonCO2_Cost_min_decreasing + yearly_DAC_DeltaEmissions_Cost_min_decreasing) / 10 ** 9,
                                        np2_max=(
                                                            yearly_DAC_CDR_CO2_Cost_decreasing + yearly_DAC_CDR_nonCO2_Cost_max_decreasing + yearly_DAC_DeltaEmissions_Cost_max_decreasing) / 10 ** 9,
                                        np3_min=finalcost_electricity_min_decreasing / 10 ** 9,
                                        np3_max=finalcost_electricity_max_decreasing / 10 ** 9,
                                        np5_min=finalcost_transport_storageCO2_min_decreasing / 10 ** 9,
                                        np5_max=finalcost_transport_storageCO2_max_decreasing / 10 ** 9,
                                        np6_min=finalcost_heat_min_decreasing / 10 ** 9,
                                        np6_max=finalcost_heat_max_decreasing / 10 ** 9,
                                        BAU_cost=finalcost_BAU_decreasing / 10 ** 9)

        # Stagnating Scenario
        plot_stacked_bars_two_scenarios((
                                                    total_DACCU_production_cost_stagnating - total_DACCU_electricity_cost_stagnating - total_DACCU_heat_cost_stagnating) / 10 ** 9,
                                        (
                                                    yearly_DAC_CDR_CO2_Cost_stagnating + yearly_DAC_CDR_nonCO2_Cost_stagnating + yearly_DAC_DeltaEmissions_Cost_stagnating) / 10 ** 9,
                                        finalcost_electricity_stagnating / 10 ** 9,
                                        YEAR,
                                        'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                        finalcost_fossilfuel_stagnating / 10 ** 9, 'Fossil kerosene',
                                        finalcost_transport_storageCO2_stagnating / 10 ** 9,
                                        'CO$_2$ transport and storage',
                                        finalcost_heat_stagnating / 10 ** 9, 'Heat', scenario='stagnating demand',
                                        what="Total cost", palette=PALETTE,
                                        np2_min=(
                                                            yearly_DAC_CDR_CO2_Cost_stagnating + yearly_DAC_CDR_nonCO2_Cost_min_stagnating + yearly_DAC_DeltaEmissions_Cost_min_stagnating) / 10 ** 9,
                                        np2_max=(
                                                            yearly_DAC_CDR_CO2_Cost_stagnating + yearly_DAC_CDR_nonCO2_Cost_max_stagnating + yearly_DAC_DeltaEmissions_Cost_max_stagnating) / 10 ** 9,
                                        np3_min=finalcost_electricity_min_stagnating / 10 ** 9,
                                        np3_max=finalcost_electricity_max_stagnating / 10 ** 9,
                                        np5_min=finalcost_transport_storageCO2_min_stagnating / 10 ** 9,
                                        np5_max=finalcost_transport_storageCO2_max_stagnating / 10 ** 9,
                                        np6_min=finalcost_heat_min_stagnating / 10 ** 9,
                                        np6_max=finalcost_heat_max_stagnating / 10 ** 9,
                                        BAU_cost=finalcost_BAU_stagnating / 10 ** 9)

        # Base Scenario
        plot_stacked_bars_two_scenarios((
                                                    total_DACCU_production_cost - total_DACCU_electricity_cost - total_DACCU_heat_cost) / 10 ** 9,
                                        (
                                                    yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost + yearly_DAC_DeltaEmissions_Cost) / 10 ** 9,
                                        finalcost_electricity / 10 ** 9,
                                        YEAR,
                                        'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                        finalcost_fossilfuel / 10 ** 9, 'Fossil kerosene',
                                        finalcost_transport_storageCO2 / 10 ** 9, 'CO$_2$ transport and storage',
                                        finalcost_heat / 10 ** 9, 'Heat', scenario='base demand growth',
                                        what="Total cost", palette=PALETTE,
                                        np2_min=(
                                                            yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_min + yearly_DAC_DeltaEmissions_Cost_min) / 10 ** 9,
                                        np2_max=(
                                                            yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost_max + yearly_DAC_DeltaEmissions_Cost_max) / 10 ** 9,
                                        np3_min=finalcost_electricity_min / 10 ** 9,
                                        np3_max=finalcost_electricity_max / 10 ** 9,
                                        np5_min=finalcost_transport_storageCO2_min / 10 ** 9,
                                        np5_max=finalcost_transport_storageCO2_max / 10 ** 9,
                                        np6_min=finalcost_heat_min / 10 ** 9,
                                        np6_max=finalcost_heat_max / 10 ** 9,
                                        BAU_cost=finalcost_BAU / 10 ** 9)

        # Histgrowth Scenario
        plot_stacked_bars_two_scenarios((
                                                    total_DACCU_production_cost_histgrowth - total_DACCU_electricity_cost_histgrowth - total_DACCU_heat_cost_histgrowth) / 10 ** 9,
                                        (
                                                    yearly_DAC_CDR_CO2_Cost_histgrowth + yearly_DAC_CDR_nonCO2_Cost_histgrowth + yearly_DAC_DeltaEmissions_Cost_histgrowth) / 10 ** 9,
                                        finalcost_electricity_histgrowth / 10 ** 9,
                                        YEAR,
                                        'DACCU CAPEX', 'DACCS CAPEX', 'Electricity',
                                        finalcost_fossilfuel_histgrowth / 10 ** 9, 'Fossil kerosene',
                                        finalcost_transport_storageCO2_histgrowth / 10 ** 9,
                                        'CO$_2$ transport and storage',
                                        finalcost_heat_histgrowth / 10 ** 9, 'Heat',
                                        scenario='historically observed demand',
                                        what="Total cost", palette=PALETTE,
                                        np2_min=(
                                                            yearly_DAC_CDR_CO2_Cost_histgrowth + yearly_DAC_CDR_nonCO2_Cost_min_histgrowth + yearly_DAC_DeltaEmissions_Cost_min_histgrowth) / 10 ** 9,
                                        np2_max=(
                                                            yearly_DAC_CDR_CO2_Cost_histgrowth + yearly_DAC_CDR_nonCO2_Cost_max_histgrowth + yearly_DAC_DeltaEmissions_Cost_max_histgrowth) / 10 ** 9,
                                        np3_min=finalcost_electricity_min_histgrowth / 10 ** 9,
                                        np3_max=finalcost_electricity_max_histgrowth / 10 ** 9,
                                        np5_min=finalcost_transport_storageCO2_min_histgrowth / 10 ** 9,
                                        np5_max=finalcost_transport_storageCO2_max_histgrowth / 10 ** 9,
                                        np6_min=finalcost_heat_min_histgrowth / 10 ** 9,
                                        np6_max=finalcost_heat_max_histgrowth / 10 ** 9,
                                        BAU_cost=finalcost_BAU_histgrowth / 10 ** 9)

    if running_singlefactor_SA is True:
        # ================================================================================================================
        # SENSITIVITY ANALYSIS
        # varying from -90% to +90%
        # range_percent = np.arange(1-0.9, 1+1.9, 0.3)
        range_percent = np.array((0, 0.3, 0.5, 0.7, 1, 1.5, 1.7, 2, 3, 5))
        range_percent_label = ["-100%", "-70%", "-50%", "-30%", "0%", "+50%", "+70%", "+100%",  "+200%", "+400%"]
        range_percent_exploratory = np.array((-3, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 5))
        range_percent_exp_label = ["-400%", "-300%", "-200%", "-100%", "-50%", "0%", "+50%", "+100%", "+200%", "+400%"]
        if SA_type == "Exploratory":
            range = range_percent_exploratory
            r_label = range_percent_exp_label
        else:
            range = range_percent
            r_label = range_percent_label
        # vary electricity cost between 0.003 and 0.07 €/kWh
        VARIATION_SA_electricity_cost_KWh = ELECTRICITY_COST_KWH * range  # np.arange(0.1*ELECTRICITY_COST_KWH, 2.5*ELECTRICITY_COST_KWH, (2.5*ELECTRICITY_COST_KWH - 0.1*ELECTRICITY_COST_KWH)/10)
        # vary learning rate between 1 and 22%
        VARIATION_SA_LEARNING_H2 = LEARNING_RATE_electrolysis_endogenous * range  # np.arange(0.1*LEARNING_RATE, 2.5*LEARNING_RATE, (2.5*LEARNING_RATE - 0.1*LEARNING_RATE)/10)
        # vary learning rate between 1 and 22%
        VARIATION_SA_LEARNING_DAC = LEARNING_RATE_DAC * range  # np.arange(0.1*LEARNING_RATE, 2.5*LEARNING_RATE, (2.5*LEARNING_RATE - 0.1*LEARNING_RATE)/10)
        # vary learning rate between 1 and 22%
        VARIATION_SA_LEARNING_RATE = LEARNING_RATE * range  # np.arange(0.1*LEARNING_RATE, 2.5*LEARNING_RATE, (2.5*LEARNING_RATE - 0.1*LEARNING_RATE)/10)
        # vary efficiency increase
        VARIATION_SA_EFFICIENCY = EFFICIENCY_INCREASE_YEARLY * range
        # vary growth rate aviation fuel demand between 0.2% to 4.5%
        VARIATION_SA_GROWTH_RATE_AVIATION_FUEL_DEMAND = GROWTH_RATE_AVIATION_FUEL_DEMAND * range  # np.arange(0.1*GROWTH_RATE_AVIATION_FUEL_DEMAND, 2.5*GROWTH_RATE_AVIATION_FUEL_DEMAND, (2.5*GROWTH_RATE_AVIATION_FUEL_DEMAND - 0.1*GROWTH_RATE_AVIATION_FUEL_DEMAND)/10)
        # vary BAU electricity cost between 0 and 0.0025 €/kWH
        VARIATION_SA_BAU_electricity_cost_KWh = np.arange(0, 0.026, 0.0026)  ## Vary between 0 and 2.5 cents per KWh
        # vary learning rate between 10 and 55%
        VARIATION_SA_LEARNING_RATE_high = np.arange(0.1, 0.6, 0.05)  # Vary LEARNING RATE between 15 and 60%
        # vary fossil jet fuel cost between 6 cent and 1.36 €/L
        VARIATION_SA_FOSSIL_FUEL_COST = FF_MARKET_COST * range  # np.arange(0.1*FF_MARKET_COST, 2.5*FF_MARKET_COST, (2.5*FF_MARKET_COST - 0.1*FF_MARKET_COST)/10)
        # vary DAC initial cost between 85 and 1921 €/tCO2
        VARIATION_SA_DAC_c0_2020 = DAC_c0_2020 * range  # np.arange(0.1*DAC_c0_2020, 2.5*DAC_c0_2020, (2.5*DAC_c0_2020 - 0.1*DAC_c0_2020)/10)
        # vary allocation to e-diesel between 5% and
        VARIATION_SA_JETFUEL_ALLOCATION = JETFUEL_ALLOCATION_SHARE * range  # np.arange(0.1, 1.1, 0.1)
        VARIATION_SA_JETFUEL_ALLOCATION[9] = 1.
        # 0.1*JETFUEL_ALLOCATION_SHARE, 2.5*JETFUEL_ALLOCATION_SHARE, (2.5*JETFUEL_ALLOCATION_SHARE - 0.1*JETFUEL_ALLOCATION_SHARE)/10)
        # vary contrail cirrus efficacy
        VARIATION_CC_EFFICACY = range * CC_EFFICACY

        # ======================= WHEN DO CONDITIONS FLIP? =======================================
        # Run single-factor SA
        # Analysis of single-factor influence
        # DACCU - DACCS

        df_SA_baseline_learning, df_SA_neutrality_learning, df_SA_zerocarbon_learning = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "learning rate", VARIATION_SA_LEARNING_RATE,
                                                    "DACCU vs. DACCS", SCENARIO, CONFIGURATION)

        df_SA_baseline_learning_H2, df_SA_neutrality_learning_H2, df_SA_zerocarbon_learning_H2 = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "learning rate H2", VARIATION_SA_LEARNING_H2,
                                                    "DACCU vs. DACCS", SCENARIO)

        df_SA_baseline_learning_DAC, df_SA_neutrality_learning_DAC, df_SA_zerocarbon_learning_DAC = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "learning rate DAC", VARIATION_SA_LEARNING_DAC,
                                                    "DACCU vs. DACCS", SCENARIO)

        df_SA_baseline_electricity, df_SA_neutrality_electricity, df_SA_zerocarbon_electricity = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "electricity cost", VARIATION_SA_electricity_cost_KWh,
                                                    "DACCU vs. DACCS", SCENARIO)

        df_SA_baseline_electricity_low, df_SA_neutrality_electricity_low, df_SA_zerocarbon_electricity_low = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "electricity cost", VARIATION_SA_BAU_electricity_cost_KWh,
                                                    "DACCU vs. DACCS", SCENARIO)

        df_SA_baseline_growth, df_SA_neutrality_growth, df_SA_zerocarbon_growth = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "growth rate", VARIATION_SA_GROWTH_RATE_AVIATION_FUEL_DEMAND,
                                                    "DACCU vs. DACCS", SCENARIO)

        df_SA_baseline_ffprice, df_SA_neutrality_ffprice, df_SA_zerocarbon_ffprice = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "fossil fuel cost", VARIATION_SA_FOSSIL_FUEL_COST,
                                                    "DACCUs vs. DACCS", SCENARIO)

        df_SA_baseline_DAC_c0, df_SA_neutrality_DAC_c0, df_SA_zerocarbon_DAC_c0 = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "DAC initial cost", VARIATION_SA_DAC_c0_2020,
                                                    "DACCU vs. DACCS", SCENARIO)
        df_SA_baseline_cc_efficacy, df_SA_neutrality_cc_efficacy, df_SA_zerocarbon_cc_efficacy = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "cc efficacy", VARIATION_CC_EFFICACY,
                                                    "DACCU vs. DACCS", SCENARIO)
        df_SA_baseline_fuel_efficiency, df_SA_neutrality_fuel_efficiency, df_SA_zerocarbon_fuel_efficiency = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "fuel efficiency", VARIATION_SA_EFFICIENCY,
                                                    "DACCU vs. DACCS", SCENARIO)


        # SENSITIVITY ANALYSIS
        ### BAU vs. DACCU

        df_SA_BAU_baseline_learning, df_SA_BAU_neutrality_learning, df_SA_BAU_zerocarbon_learning = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "learning rate", VARIATION_SA_LEARNING_RATE,
                                                    "BAU", SCENARIO)

        df_SA_BAU_baseline_learning_H2, df_SA_BAU_neutrality_learning_H2, df_SA_BAU_zerocarbon_learning_H2 = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "learning rate H2", VARIATION_SA_LEARNING_H2,
                                                    "BAU", SCENARIO)
        df_SA_BAU_baseline_learning_DAC, df_SA_BAU_neutrality_learning_DAC, df_SA_BAU_zerocarbon_learning_DAC = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "learning rate DAC", VARIATION_SA_LEARNING_DAC,
                                                    "BAU", SCENARIO)

        df_SA_BAU_baseline_growth, df_SA_BAU_neutrality_growth, df_SA_BAU_zerocarbon_growth = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "growth rate", VARIATION_SA_GROWTH_RATE_AVIATION_FUEL_DEMAND,
                                                    "BAU", SCENARIO)

        df_SA_BAU_baseline_electricity, df_SA_BAU_neutrality_electricity, df_SA_BAU_zerocarbon_electricity = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "electricity cost", VARIATION_SA_electricity_cost_KWh,
                                                    "BAU", SCENARIO)

        df_SA_BAU_baseline_ffprice, df_SA_BAU_neutrality_ffprice, df_SA_BAU_zerocarbon_ffprice = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "fossil fuel cost", VARIATION_SA_FOSSIL_FUEL_COST,
                                                    "BAU", SCENARIO)

        df_SA_BAU_baseline_DAC_c0, df_SA_BAU_neutrality_DAC_c0, df_SA_BAU_zerocarbon_DAC_c0 = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "DAC initial cost", VARIATION_SA_DAC_c0_2020,
                                                    "BAU", SCENARIO)

        df_SA_BAU_baseline_fuel_efficiency, df_SA_BAU_neutrality_fuel_efficiency, df_SA_BAU_zerocarbon_fuel_efficiency = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "fuel efficiency", VARIATION_SA_EFFICIENCY,
                                                    "BAU", SCENARIO)

        df_SA_BAU_baseline_cc_efficacy, df_SA_BAU_neutrality_cc_efficacy, df_SA_BAU_zerocarbon_cc_efficacy = \
            run_sensitivity_analysis_single_factors(df_input, df_emissions_input,
                                                    GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                                    LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                                    LEARNING_RATE_DAC,
                                                    ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                                    DAC_q0_Gt_2020,
                                                    DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                                    JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                                    "cc efficacy", VARIATION_CC_EFFICACY,
                                                    "BAU", SCENARIO)

        standard_values_tornado = {
            "levelized cost\nof electricity": ELECTRICITY_COST_KWH,
            "all learning\nrates": LEARNING_RATE,
            "learning\nrate H$_2$": LEARNING_RATE_electrolysis_endogenous,
            "learning\nrate DAC": LEARNING_RATE_DAC,
            "demand\ngrowth": GROWTH_RATE_AVIATION_FUEL_DEMAND,
            "price of\nfossil kerosene": FF_MARKET_COST,
            "DAC initial\nCAPEX": DAC_c0_2020,
            "fuel\nefficiency": EFFICIENCY_INCREASE_YEARLY,
            "contrails\nefficacy": CC_EFFICACY
        }

        DACCS_DACCU_finalcost_difference_SA_zerocarbon = pd.DataFrame(
            {"levelized cost\nof electricity": df_SA_zerocarbon_electricity.iloc[31, ::-1].values,
             "all learning\nrates": df_SA_zerocarbon_learning.iloc[31, ::-1].values,
             "learning\nrate H$_2$": df_SA_zerocarbon_learning_H2.iloc[31, ::-1].values,
             "learning\nrate DAC": df_SA_zerocarbon_learning_DAC.iloc[31, ::-1].values,
             "demand\ngrowth": df_SA_zerocarbon_growth.iloc[31, ::-1].values,
             "price of\nfossil kerosene": df_SA_zerocarbon_ffprice.iloc[31, ::-1].values,
             "DAC initial\nCAPEX": df_SA_zerocarbon_DAC_c0.iloc[31, ::-1].values,
             "fuel\nefficiency": df_SA_zerocarbon_fuel_efficiency.iloc[31, ::-1].values,
             "contrails\nefficacy": df_SA_zerocarbon_cc_efficacy.iloc[31, ::-1].values
             },
            index=r_label[::-1]
        )

        DACCS_DACCU_finalcost_difference_SA_neutrality = pd.DataFrame(
            {"levelized cost\nof electricity": df_SA_neutrality_electricity.iloc[31, ::-1].values,
             "all learning\nrates": df_SA_neutrality_learning.iloc[31, ::-1].values,
             "learning\nrate H$_2$": df_SA_neutrality_learning_H2.iloc[31, ::-1].values,
             "learning\nrate DAC": df_SA_neutrality_learning_DAC.iloc[31, ::-1].values,
             "demand\ngrowth": df_SA_neutrality_growth.iloc[31, ::-1].values,
             "price of\nfossil kerosene": df_SA_neutrality_ffprice.iloc[31, ::-1].values,
             "DAC initial\nCAPEX": df_SA_neutrality_DAC_c0.iloc[31, ::-1].values,
             "fuel\nefficiency": df_SA_neutrality_fuel_efficiency.iloc[31, ::-1].values,
             "contrails\nefficacy": df_SA_neutrality_cc_efficacy.iloc[31, ::-1].values
             },
            index=r_label[::-1]
        )

        FF_DACCU_finalcost_difference_SA_zerocarbon = pd.DataFrame(
            {"levelized cost\nof electricity": df_SA_BAU_zerocarbon_electricity.iloc[31, ::-1].values,
             "all learning\nrates": df_SA_BAU_zerocarbon_learning.iloc[31, ::-1].values,
             "learning\nrate H$_2$": df_SA_BAU_zerocarbon_learning_H2.iloc[31, ::-1].values,
             "learning\nrate DAC": df_SA_BAU_zerocarbon_learning_DAC.iloc[31, ::-1].values,
             "demand\ngrowth": df_SA_BAU_zerocarbon_growth.iloc[31, ::-1].values,
             "price of\nfossil kerosene": df_SA_BAU_zerocarbon_ffprice.iloc[31, ::-1].values,
             "DAC initial\nCAPEX": df_SA_BAU_zerocarbon_DAC_c0.iloc[31, ::-1].values,
             "fuel\nefficiency": df_SA_BAU_zerocarbon_fuel_efficiency.iloc[31, ::-1].values,
             "contrails\nefficacy": df_SA_BAU_zerocarbon_cc_efficacy.iloc[31, ::-1].values
             },
            index=r_label[::-1]
        )

        if plotting_SA_figures is True:
            # plot DACCS vs. DACCU
            plot_heatmap_single_SA(DACCS_DACCU_finalcost_difference_SA_zerocarbon, "Carbon neutrality", "DACCS")
            plot_heatmap_single_SA(DACCS_DACCU_finalcost_difference_SA_zerocarbon, "Carbon neutrality", "DACCS")
            plot_heatmap_single_SA(DACCS_DACCU_finalcost_difference_SA_neutrality, "Climate neutrality", "DACCS")
            # plot FF vs. DACCU
            plot_heatmap_single_SA(FF_DACCU_finalcost_difference_SA_zerocarbon, "Carbon neutrality", "fossil jet fuel")


    if running_policy_SA is True:
        df_SA_baseline_CO2price, df_SA_neutrality_CO2price, df_SA_zerocarbon_CO2price = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "CO2 prices", range_percent,
                                              "DACCU vs. DACCS", emissions_price=PRICE_CO2,
                                              configuration_PtL=CONFIGURATION)

        df_SA_baseline_cc_price, df_SA_neutrality_cc_price, df_SA_zerocarbon_cc_price = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "prices on climate impacts", range_percent,
                                              "DACCU vs. DACCS", emissions_price=PRICE_CC_IMPACTS)

        #SUBSIDY_DACCS = 100  # €/tCO2
        #SUBSIDY_DACCU = 0.033  # €/kg
        df_SA_baseline_DACCS_subsidy, df_SA_neutrality_DACCS_subsidy, df_SA_zerocarbon_DACCS_subsidy = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "subsidies", range_percent,
                                              "DACCU vs. DACCS", subsidy=SUBSIDY_DACCS, subsidy_type="DACCS")
        df_SA_baseline_DACCU_subsidy, df_SA_neutrality_DACCU_subsidy, df_SA_zerocarbon_DACCU_subsidy = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "subsidies", range_percent,
                                              "DACCU vs. DACCS", subsidy=SUBSIDY_DACCU, subsidy_type="DACCU")

        #EXCESS_ELECTRICITY_COST = 0.003  # €/kWh
        df_SA_baseline_excessel, df_SA_neutrality_excessel, df_SA_zerocarbon_excessel = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "excess electricity", range_percent,
                                              "DACCU vs. DACCS", excess_electricity_price=EXCESS_ELECTRICITY_COST)

        #EXCESS_ELECTRICITY_COST_NEGATIVE = -0.001  # €/kWh
        df_SA_baseline_excessel_neg, df_SA_neutrality_excessel_neg, df_SA_zerocarbon_excessel_neg = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "excess electricity", range_percent,
                                              "DACCU vs. DACCS",
                                              excess_electricity_price=EXCESS_ELECTRICITY_COST_NEGATIVE)

        #DEMAND_RATE_CAPPED = -0.001  # %
        df_SA_baseline_demandcap, df_SA_neutrality_demandcap, df_SA_zerocarbon_demandcap = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "demand reduction", range_percent,
                                              "DACCU vs. DACCS", growth_rate_capped=DEMAND_RATE_CAPPED)

        DACCS_DACCU_finalcost_difference_SA_policies_zerocarbon = pd.DataFrame(
            {"CO$_2$ \n price \n (100€/tCO$_2$)": df_SA_zerocarbon_CO2price.iloc[31].values,
             "CC impacts \n price \n (100€/tCO$_{2eq}$)": df_SA_zerocarbon_cc_price.iloc[31].values,
             "Subsidy \n DACCS \n (100€/tCO$_2$)": df_SA_zerocarbon_DACCS_subsidy.iloc[31].values,
             "Subsidy \n DACCU \n (33€/t)": df_SA_zerocarbon_DACCU_subsidy.iloc[31].values,
             "Excess \n electricity \n(0.003€/kWh)": df_SA_zerocarbon_excessel.iloc[31].values,
             # "Demand \n cap \n (-0.1%/year)": df_SA_zerocarbon_demandcap.iloc[31].values,
             },
            index=r_label
        )

        DACCS_DACCU_finalcost_difference_SA_policies_neutrality = pd.DataFrame(
            {"CO$_2$ \n price \n (100€/tCO$_2$)": df_SA_neutrality_CO2price.iloc[31].values,
             "CC impacts \n price \n (100€/tCO$_{2eq}$)": df_SA_neutrality_cc_price.iloc[31].values,
             "Subsidy \n DACCS \n (100€/tCO$_2$)": df_SA_neutrality_DACCS_subsidy.iloc[31].values,
             "Subsidy \n DACCU \n (33€/t)": df_SA_neutrality_DACCU_subsidy.iloc[31].values,
             "Excess \n electricity \n(0.003€/kWh)": df_SA_neutrality_excessel.iloc[31].values,
             # "Demand \n cap \n (-0.1%/year)": df_SA_neutrality_demandcap.iloc[31].values,
             },
            index=r_label
        )

        df_SA_BAU_baseline_CO2price, df_SA_BAU_neutrality_CO2price, df_SA_BAU_zerocarbon_CO2price = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "CO2 prices", range_percent,
                                              "BAU", emissions_price=PRICE_CO2)

        df_SA_BAU_baseline_cc_price, df_SA_BAU_neutrality_cc_price, df_SA_BAU_zerocarbon_cc_price = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "prices on climate impacts", range_percent,
                                              "BAU", emissions_price=PRICE_CC_IMPACTS)
        df_SA_BAU_baseline_DACCU_subsidy, df_SA_BAU_neutrality_DACCU_subsidy, df_SA_BAU_zerocarbon_DACCU_subsidy = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "subsidies", range_percent,
                                              "BAU", subsidy=SUBSIDY_DACCU, subsidy_type="DACCU")

        df_SA_BAU_baseline_excessel, df_SA_BAU_neutrality_excessel, df_SA_BAU_zerocarbon_excessel = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "excess electricity", range_percent,
                                              "BAU", excess_electricity_price=EXCESS_ELECTRICITY_COST)


        df_SA_BAU_baseline_excessel_neg, df_SA_BAU_neutrality_excessel_neg, df_SA_BAU_zerocarbon_excessel_neg = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "excess electricity", range_percent,
                                              "BAU", excess_electricity_price=EXCESS_ELECTRICITY_COST_NEGATIVE)


        df_SA_BAU_baseline_demandcap, df_SA_BAU_neutrality_demandcap, df_SA_BAU_zerocarbon_demandcap = \
            run_sensitivity_analysis_policies(df_input, df_emissions_input,
                                              GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                              LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                              LEARNING_RATE_DAC,
                                              ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                              DAC_q0_Gt_2020,
                                              DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                              JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                              "demand reduction", range_percent,
                                              "BAU", growth_rate_capped=DEMAND_RATE_CAPPED)
        FF_DACCU_finalcost_difference_SA_policies_zerocarbon = pd.DataFrame(
            {"CO$_2$ \n price \n (100€/tCO$_2$)": df_SA_BAU_zerocarbon_CO2price.iloc[31].values,
             "CC impacts \n price \n (100€/tCO$_{2eq}$)": df_SA_BAU_zerocarbon_cc_price.iloc[31].values,
             "Subsidy \n DACCU \n (33€/t)": df_SA_BAU_zerocarbon_DACCU_subsidy.iloc[31].values,
             #"Subsidy \n DACCU \n (100€/t)": df_SA_BAU_zerocarbon_DACCU_highsubsidy.iloc[31].values,
             "Excess \n electricity \n(0.003€/kWh)": df_SA_BAU_zerocarbon_excessel.iloc[31].values,
             "Excess \n electricity \n(-0.001€/kWh)": df_SA_BAU_zerocarbon_excessel_neg.iloc[31].values,
             # "Demand \n cap \n (-0.1%/year)": df_SA_BAU_zerocarbon_demandcap.iloc[40].values,
             },
            index=r_label
        )

        if plotting_SA_figures is True:
            plot_heatmap_single_SA(DACCS_DACCU_finalcost_difference_SA_policies_zerocarbon[::-1], "Carbon neutrality",
                                   "DACCS", policy=True)
            plot_heatmap_single_SA(DACCS_DACCU_finalcost_difference_SA_policies_neutrality[::-1], "Climate neutrality",
                                   "DACCS", policy=True)
            plot_heatmap_single_SA(FF_DACCU_finalcost_difference_SA_policies_zerocarbon[::-1], "Carbon neutrality",
                                   "BAU", policy=True)
    if running_SA is True:
        df_SA_baseline_electricity_learning, df_SA_neutrality_electricity_learning, df_SA_zerocarbon_electricity_learning = \
            run_sensitivity_analysis_III(df_input, df_emissions_input,
                                         GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                         LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, LEARNING_RATE_DAC,
                                         ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                         DAC_q0_Gt_2020,
                                         DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                         JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                         "electricity cost", VARIATION_SA_electricity_cost_KWh,
                                         "learning rate", VARIATION_SA_LEARNING_RATE,
                                         "DACCU vs. DACCS", SCENARIO, configuration_PtL=CONFIGURATION)
        df_SA_baseline_electricity_growth, df_SA_neutrality_electricity_growth, \
            df_SA_zerocarbon_electricity_growth = \
            run_sensitivity_analysis_III(df_input, df_emissions_input,
                                         GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                         LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, LEARNING_RATE_DAC,
                                         ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                         DAC_q0_Gt_2020,
                                         DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                         JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                         "electricity cost", VARIATION_SA_electricity_cost_KWh,
                                         "growth rate", VARIATION_SA_GROWTH_RATE_AVIATION_FUEL_DEMAND,
                                         "DACCU vs. DACCS", SCENARIO, configuration_PtL=CONFIGURATION)
        df_SA_FF_baseline_electricity_learning, df_SA_FF_neutral_electricity_learning, df_SA_FF_zerocarbon_electricity_learning \
            = run_sensitivity_analysis_III(df_input, df_emissions_input,
                                           GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
                                           LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO,
                                           LEARNING_RATE_DAC,
                                           ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
                                           DAC_q0_Gt_2020,
                                           DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
                                           JETFUEL_ALLOCATION_SHARE, CC_EFFICACY,
                                           var_para_1="electricity cost", var1=VARIATION_SA_electricity_cost_KWh,
                                           var_para_2="learning rate", var2=VARIATION_SA_LEARNING_RATE,
                                           what="BAU", scenario=SCENARIO)
        df_SA_FF_baseline_electricity_ffprice, df_SA_FF_neutral_electricity_ffprice, df_SA_FF_zerocarbon_electricity_ffprice = run_sensitivity_analysis_III(
            df_input, df_emissions_input,
            GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
            LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, LEARNING_RATE_DAC,
            ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_q0_Gt_2020,
            DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020, JETFUEL_ALLOCATION_SHARE,
            CC_EFFICACY,
            var_para_1="electricity cost", var1=VARIATION_SA_electricity_cost_KWh,
            var_para_2="fossil fuel cost", var2=VARIATION_SA_FOSSIL_FUEL_COST,
            what="BAU", scenario=SCENARIO)
        df_SA_baseline_electricity_ffprice, df_SA_neutral_electricity_ffprice, df_SA_zerocarbon_electricity_ffprice = run_sensitivity_analysis_III(
            df_input, df_emissions_input,
            GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
            LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, LEARNING_RATE_DAC,
            ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST, DAC_q0_Gt_2020,
            DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020, JETFUEL_ALLOCATION_SHARE,
            CC_EFFICACY,
            var_para_1="electricity cost", var1=VARIATION_SA_electricity_cost_KWh,
            var_para_2="fossil fuel cost", var2=VARIATION_SA_FOSSIL_FUEL_COST,
            what="DACCU vs. DACCS", scenario=SCENARIO)

        if plotting_SA_figures is True:
            # Varying electricity prices and growth rate of aviation sector
            plot_heatmap_SA_II(df_SA_neutrality_electricity_growth, neutrality="Climate neutrality",
                               comparison="DACCS", var_para_1="electricity cost",
                               var_para_2="growth rate")
            plot_heatmap_SA_II(df_SA_zerocarbon_electricity_growth, neutrality="Carbon neutrality",
                               comparison="DACCS", var_para_1="electricity cost",
                               var_para_2="growth rate")

            plot_heatmap_SA_II(df_SA_neutrality_electricity_learning, neutrality="Climate neutrality",
                               comparison="DACCS", var_para_1="electricity cost",
                               var_para_2="learning rate")
            plot_heatmap_SA_II(df_SA_zerocarbon_electricity_learning, neutrality="Carbon neutrality",
                               comparison="DACCS", var_para_1="electricity cost",
                               var_para_2="learning rate")

            # Varying electricity prices and learning rate of aviation sector
            plot_heatmap_SA_II(df_SA_zerocarbon_electricity_ffprice, neutrality="Carbon neutrality",
                               comparison="DACCS",
                               var_para_1="electricity cost",
                               var_para_2="fossil fuel cost")
            plot_heatmap_SA_II(df_SA_neutral_electricity_ffprice, neutrality="Climate neutrality",
                               comparison="DACCS",
                               var_para_1="electricity cost",
                               var_para_2="fossil fuel cost")

            # Varying electricity prices and learning rate of aviation sector
            plot_heatmap_SA_II(df_SA_FF_zerocarbon_electricity_learning, neutrality="Carbon neutrality",
                               comparison="fossil kerosene", var_para_1="electricity cost",
                               var_para_2="learning rate")
            plot_heatmap_SA_II(df_SA_FF_neutral_electricity_learning, neutrality="Climate neutrality",
                               comparison="fossil kerosene", var_para_1="electricity cost",
                               var_para_2="learning rate")

            # Varying electricity prices and learning rate of aviation sector
            plot_heatmap_SA_II(df_SA_FF_zerocarbon_electricity_ffprice, neutrality="Carbon neutrality",
                               comparison="fossil kerosene", var_para_1="electricity cost",
                               var_para_2="fossil fuel cost")
            plot_heatmap_SA_II(df_SA_FF_neutral_electricity_ffprice, neutrality="Climate neutrality",
                               comparison="fossil kerosene", var_para_1="electricity cost",
                               var_para_2="fossil fuel cost")

    if running_optimization is True:
        range_percent_exploratory = np.array((-3, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 5))
        range_percent_exp_label = ["-400%", "-300%", "-200%", "-100%", "-50%", "0%", "+50%", "+100%", "+200%",
                                   "+400%"]
        range = range_percent_exploratory
        r_label = range_percent_exp_label
        # vary electricity cost between 0.003 and 0.07 €/kWh
        VARIATION_SA_electricity_cost_KWh = ELECTRICITY_COST_KWH * range  # np.arange(0.1*ELECTRICITY_COST_KWH, 2.5*ELECTRICITY_COST_KWH, (2.5*ELECTRICITY_COST_KWH - 0.1*ELECTRICITY_COST_KWH)/10)
        # vary learning rate between 1 and 22%
        VARIATION_SA_LEARNING_H2 = LEARNING_RATE_electrolysis_endogenous * range  # np.arange(0.1*LEARNING_RATE, 2.5*LEARNING_RATE, (2.5*LEARNING_RATE - 0.1*LEARNING_RATE)/10)
        # vary learning rate between 1 and 22%
        VARIATION_SA_LEARNING_DAC = LEARNING_RATE_DAC * range  # np.arange(0.1*LEARNING_RATE, 2.5*LEARNING_RATE, (2.5*LEARNING_RATE - 0.1*LEARNING_RATE)/10)
        # vary learning rate between 1 and 22%
        VARIATION_SA_LEARNING_RATE = LEARNING_RATE * range  # np.arange(0.1*LEARNING_RATE, 2.5*LEARNING_RATE, (2.5*LEARNING_RATE - 0.1*LEARNING_RATE)/10)
        # vary efficiency increase
        VARIATION_SA_EFFICIENCY = EFFICIENCY_INCREASE_YEARLY * range
        # vary growth rate aviation fuel demand between 0.2% to 4.5%
        VARIATION_SA_GROWTH_RATE_AVIATION_FUEL_DEMAND = GROWTH_RATE_AVIATION_FUEL_DEMAND * range  # np.arange(0.1*GROWTH_RATE_AVIATION_FUEL_DEMAND, 2.5*GROWTH_RATE_AVIATION_FUEL_DEMAND, (2.5*GROWTH_RATE_AVIATION_FUEL_DEMAND - 0.1*GROWTH_RATE_AVIATION_FUEL_DEMAND)/10)
        # vary BAU electricity cost between 0 and 0.0025 €/kWH
        VARIATION_SA_BAU_electricity_cost_KWh = np.arange(0, 0.026, 0.0026)  ## Vary between 0 and 2.5 cents per KWh
        # vary learning rate between 10 and 55%
        VARIATION_SA_LEARNING_RATE_high = np.arange(0.1, 0.6, 0.05)  # Vary LEARNING RATE between 15 and 60%
        # vary fossil jet fuel cost between 6 cent and 1.36 €/L
        VARIATION_SA_FOSSIL_FUEL_COST = FF_MARKET_COST * range  # np.arange(0.1*FF_MARKET_COST, 2.5*FF_MARKET_COST, (2.5*FF_MARKET_COST - 0.1*FF_MARKET_COST)/10)
        # vary DAC initial cost between 85 and 1921 €/tCO2
        VARIATION_SA_DAC_c0_2020 = DAC_c0_2020 * range  # np.arange(0.1*DAC_c0_2020, 2.5*DAC_c0_2020, (2.5*DAC_c0_2020 - 0.1*DAC_c0_2020)/10)
        # vary allocation to e-diesel between 5% and
        VARIATION_SA_JETFUEL_ALLOCATION = JETFUEL_ALLOCATION_SHARE * range  # np.arange(0.1, 1.1, 0.1)
        VARIATION_SA_JETFUEL_ALLOCATION[9] = 1.
        # 0.1*JETFUEL_ALLOCATION_SHARE, 2.5*JETFUEL_ALLOCATION_SHARE, (2.5*JETFUEL_ALLOCATION_SHARE - 0.1*JETFUEL_ALLOCATION_SHARE)/10)
        # vary contrail cirrus efficacy
        CC_EFFICACY = 1
        VARIATION_CC_EFFICACY = range * CC_EFFICACY

        # Example usage
        vars_to_analyze = {
            "learning rate": VARIATION_SA_LEARNING_RATE,
            "learning rate H2": VARIATION_SA_LEARNING_H2,
            "learning rate DAC": VARIATION_SA_LEARNING_DAC,
            "electricity cost": VARIATION_SA_BAU_electricity_cost_KWh,
            "fossil fuel cost": VARIATION_SA_FOSSIL_FUEL_COST,
            "growth rate": VARIATION_SA_GROWTH_RATE_AVIATION_FUEL_DEMAND,
            "DAC initial cost": VARIATION_SA_DAC_c0_2020,
            "cc efficacy": VARIATION_CC_EFFICACY,
            "fuel efficiency": VARIATION_SA_EFFICIENCY,
        }

        standard_values = {
            "learning rate": LEARNING_RATE,
            "learning rate H2": LEARNING_RATE_electrolysis_endogenous,
            "learning rate DAC": LEARNING_RATE_DAC,
            "electricity cost": ELECTRICITY_COST_KWH,
            "fossil fuel cost": FF_MARKET_COST,
            "growth rate": GROWTH_RATE_AVIATION_FUEL_DEMAND,
            "DAC initial cost": DAC_c0_2020,
            "cc efficacy": CC_EFFICACY,
            "fuel efficiency": EFFICIENCY_INCREASE_YEARLY,
            # Add other variables and their standard values here
        }

        optimization_DACCUvsDACCS_zerocarbon = run_sensitivity_analysis_all_variables(
            df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
            LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, LEARNING_RATE_DAC,
            ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
            DAC_q0_Gt_2020, DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
            JETFUEL_ALLOCATION_SHARE, CC_EFFICACY, vars_to_analyze, "DACCS", reference="DACCU",
            scenario="carbon neutrality", configuration_PtL=CONFIGURATION)

        optimization_DACCUvsDACCS_climneutrality = run_sensitivity_analysis_all_variables(
            df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
            LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, LEARNING_RATE_DAC,
            ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
            DAC_q0_Gt_2020, DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
            JETFUEL_ALLOCATION_SHARE, CC_EFFICACY, vars_to_analyze, "DACCS", reference="DACCU",
            scenario="climate neutrality", configuration_PtL=CONFIGURATION)

        optimization_DACCUvsBAU_zerocarbon = run_sensitivity_analysis_all_variables(
            df_input, df_emissions_input, GROWTH_RATE_AVIATION_FUEL_DEMAND, EFFICIENCY_INCREASE_YEARLY,
            LEARNING_RATE_electrolysis_endogenous, LEARNING_RATE_CO, LEARNING_RATE_DAC,
            ELECTRICITY_COST_KWH, FF_MARKET_COST, CO2_TRANSPORT_STORAGE_COST,
            DAC_q0_Gt_2020, DAC_c0_2020, H2_q0_Mt_2020, H2_c0_2020, CO_q0_Mt_2020, CO_c0_2020,
            JETFUEL_ALLOCATION_SHARE, CC_EFFICACY, vars_to_analyze, "BAU", reference="DACCU",
            scenario="carbon neutrality", configuration_PtL=CONFIGURATION)


        if plot_main_figures is True:
            # FIGURE 5
            combined_plot(DACCS_DACCU_finalcost_difference_SA_zerocarbon, optimization_DACCUvsDACCS_zerocarbon,
                          '+70%', '-70%', standard_values,  # original_diff_zerocarbon / 10 ** 12,
                          hue_scatter='Change in DACCU cost penalty (%)', percentage_penalty_change=True,
                          neutrality='Carbon neutrality', comparison='DACCS', reference='DACCU')
            combined_plot(FF_DACCU_finalcost_difference_SA_zerocarbon, optimization_DACCUvsBAU_zerocarbon,
                          '+70%', '-70%', standard_values,  # original_diff_zerocarbon / 10 ** 12,
                          hue_scatter='Change in DACCU cost penalty (%)', percentage_penalty_change=True,
                          neutrality='Carbon neutrality', comparison='BAU', reference='DACCU')


        if plotting_SA_figures is True:
            # Call the function
            combined_plot(DACCS_DACCU_finalcost_difference_SA_zerocarbon, optimization_DACCUvsDACCS_zerocarbon,
                          '+100%', '-100%', standard_values,  # original_diff_zerocarbon / 10 ** 12,
                          hue_scatter='Change in DACCU cost penalty (%)',
                          neutrality='Carbon neutrality', comparison='DACCS')
            combined_plot(DACCS_DACCU_finalcost_difference_SA_neutrality, optimization_DACCUvsDACCS_climneutrality,
                          '+100%', '-100%', standard_values,  # original_diff_climneutrality / 10 ** 12,
                          hue_scatter='Change in DACCS cost penalty (%)',
                          neutrality='Climate neutrality', comparison='DACCU', reference='DACCS')
            combined_plot(FF_DACCU_finalcost_difference_SA_zerocarbon, optimization_DACCUvsBAU_zerocarbon,
                          '+100%', '-100%', standard_values,  # original_diff_zerocarbon / 10 ** 12,
                          hue_scatter='Change in DACCU cost penalty (%)',
                          neutrality='Carbon neutrality', comparison='BAU', reference='DACCU')
            combined_plot(DACCS_DACCU_finalcost_difference_SA_neutrality, optimization_DACCUvsDACCS_climneutrality,
                          '+100%', '-100%', standard_values,  # original_diff_zerocarbon / 10 ** 12,
                          hue_scatter='Change in DACCS cost penalty (%)', percentage_penalty_change=True,
                          neutrality='Climate neutrality', comparison='DACCU', reference='DACCS')


    if output_csv is True:

        # DATA FIGURE 1
        # Example usage
        scenario_names_BAU = [
            'Business as Usual',
            'CO2neutrality',
            'Climate neutrality'
        ]

        # Assuming you have the prepped data in the following variables
        prepped_emissions_main_scenario_dict = {
            'Indirect Emissions': prepdata_indirect_emissions,
            'DAC CDR CO2': prepdata_DAC_CDR_CO2,
            'DAC CDR nonCO2': prepdata_DAC_CDR_nonCO2,
            'DAC CDR other nonCO2': prepdata_DAC_CDR_othernonCO2,
            'DAC CDR CC': prepdata_DAC_CDR_CC,
            'Flying CO2': prepdata_flying_CO2,
            'Flying nonCO2': prepdata_flying_nonCO2,
            'Flying other nonCO2': prepdata_flying_othernonCO2,
            'Flying CC': prepdata_flying_CC
        }

        export_prepped_data_to_excel(prepped_emissions_main_scenario_dict, 'Exports/prepped_emissions_main_scenarios.xlsx', scenario_names_BAU, BAU=True, unit = 'GtCO2eq*/year')


        # DATA FIGURE 2
        # FIGURE 2 - COST OF DIFFERENT SCENARIOS
        scenario_names_main = [
            'Baseline',
            'Climate neutrality',
            'CO2neutrality'
        ]

        # Assuming you have the prepped data in the following variables
        prepped_cost_per_liter_main_dict = {
            'DACCU CAPEX': cost_per_liter_DACCU_CAPEX,
            'DACCS CAPEX': cost_per_liter_DACCS_CAPEX,
            'Electricity': cost_per_liter_electricity,
            'Fossil kerosene': cost_per_liter_fossilfuel,
            'CO2 Transport and storage':cost_per_liter_transport_storageCO2,
            'Heat': cost_per_liter_heat,
        }

        export_prepped_data_to_excel(prepped_cost_per_liter_main_dict, 'Exports/costs_per_liter_main_scenarios.xlsx', scenario_names_main, unit = '€/L')

        # Assuming you have the prepped data in the following variables
        emissions_main_dict = {
            'Indirect Emissions': totalIndirectEmissions,
            'Delta Indirect Emissions': Delta_totalIndirectEmissions,
            'DAC CDR CO2': DAC_CDR_CO2_Gt,
            'DAC CDR nonCO2': DAC_CDR_nonCO2_Gt,
            'Flying CO2': flying_CO2_emissions,
            'Flying nonCO2': flying_nonCO2_emissions
        }

        export_prepped_data_to_excel(emissions_main_dict, 'Exports/emissions_main_scenarios.xlsx', scenario_names_main, unit = '€/L')


        prepped_cost_per_emissions_main_dict = {
            'DACCU CAPEX': (total_DACCU_production_cost - total_DACCU_electricity_cost - total_DACCU_heat_cost) / (
                        total_abated_emissions * 10 ** 9),
            'DACCS CAPEX': (yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost) / (total_abated_emissions * 10 ** 9),
            'Electricity': finalcost_electricity / (total_abated_emissions * 10 ** 9),
            'Fossil kerosene': finalcost_fossilfuel / (total_abated_emissions * 10 ** 9),
            'CO2 Transport and storage': finalcost_transport_storageCO2 / (total_abated_emissions * 10 ** 9),
            'Heat': finalcost_heat / (total_abated_emissions * 10 ** 9),
        }

        export_prepped_data_to_excel(prepped_cost_per_emissions_main_dict, 'Exports/costs_per_emissions_main_scenarios.xlsx', scenario_names_main, unit = '€/tCO2eq*')

        prepped_cost_per_DAC_main_dict = {
            'DACCU CAPEX': (total_DACCU_production_cost - total_DACCU_electricity_cost) / (DAC_total_Gt * 10 ** 9),
            'DACCS CAPEX': (yearly_DAC_CDR_CO2_Cost + yearly_DAC_CDR_nonCO2_Cost) / (DAC_total_Gt* 10 ** 9),
            'Electricity': finalcost_electricity / (DAC_total_Gt * 10 ** 9),
            'Fossil kerosene': finalcost_fossilfuel / (DAC_total_Gt * 10 ** 9),
            'CO2 Transport and storage': finalcost_transport_storageCO2 / (DAC_total_Gt * 10 ** 9),
            'Heat': finalcost_heat / (DAC_total_Gt * 10 ** 9),
        }

        export_prepped_data_to_excel(prepped_cost_per_DAC_main_dict, 'Exports/costs_per_DAC_main_scenarios.xlsx', scenario_names_main, unit = '€/tCO2 removed')

        # DATA FIGURE 3
        prepped_increase_cost_passenger_dict = {
            'Absolute London-New York': cost_neutrality_per_passenger_LN_NY,
            'Absolute London-Berlin': cost_neutrality_per_passenger_LN_Berlin,
            'Absolute London-Perth': cost_neutrality_per_passenger_LN_Perth,
            'Percentage London-New York': increase_neutrality_flight_price_LN_NY * 100,
            'Percentage London-Berlin': increase_neutrality_flight_price_LN_Berlin * 100,
            'Percentage London-Perth': increase_neutrality_flight_price_LN_Perth * 100
        }

        export_prepped_data_to_excel(prepped_increase_cost_passenger_dict, 'Exports/costs_per_passenger_main_scenarios.xlsx',
                                     scenario_names_main, unit='€/flight')

        # FIGURE 4 - alternative mitigation options
        scenario_names_variations = [
            'Default',
            'Capped demand (0%)',
            'Decreasing demand (+2%/year)',
            'Rerouting',
            'Ignoring contrails'
        ]

        # Alternative scenarios (climate neutrality only) - Emissions
        prepped_emissions_variations_scenario_dict = {
            'Indirect Emissions': prepped_indirect_emissions_variation,
            'DAC CDR CO2': prepped_DAC_CDR_CO2_variations,
            'DAC CDR nonCO2': prepped_DAC_CDR_nonCO2_variations,
            'Flying CO2': prepped_flying_CO2_emissions_variations,
            'Flying nonCO2': prepped_flying_nonCO2_emissions_variations,
            'Flying CC avoided': prepped_flying_CC_avoided
        }

        export_prepped_data_to_excel(prepped_emissions_variations_scenario_dict, 'Exports/emissions_alternative_scenarios.xlsx', scenario_names_variations, unit = 'GtCO2eq*/year')

        # Alternative scenarios (climate neutrality only) - Costs
        prepped_costs_variations_scenario_dict = {
            'DACCU CAPEX': prepped_DACCU_CAPEX,
            'DACCS CAPEX': prepped_DACCS_CAPEX,
            'Electricity': prepped_cost_electricity,
            'Fossil kerosene': prepped_cost_fossilfuel,
            'CO2 transport and storage': prepped_cost_transport_storageCO2,
            'Heat': prepped_cost_heat
        }

        export_prepped_data_to_excel(prepped_costs_variations_scenario_dict, 'Exports/costs_alternative_scenarios.xlsx', scenario_names_variations, unit = 'Billion €/year')

        # EXTRA DATA
        Demand_fuel_dict = {
            'Fossil kerosene Tg': FF_volumes_Tg,
            'DACCU fuels Tg': DACCU_volumes_Tg,
            'Yearly DACCU share %': DACCU_volumes_Tg/(FF_volumes_Tg+DACCU_volumes_Tg)*100,
        }

        export_prepped_data_to_excel(Demand_fuel_dict, 'Exports/demand_main_scenarios.xlsx', scenario_names_main, unit = '')

















