import numpy as np
import pandas as pd
import hvplot.pandas
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='white')
from matplotlib.lines import Line2D


COLORS = 'magma'

def hex_to_rgb(hex_code):
    """Convert HEX code to RGB tuple in 0-1 range."""
    hex_code = hex_code.lstrip('#')
    h_len = len(hex_code)
    return tuple(int(hex_code[i:i + h_len // 3], 16) / 255.0 for i in range(0, h_len, h_len // 3))

def remap_standard_values(standard_values, name_mapping):
    return {name_mapping.get(k, k): v for k, v in standard_values.items()}


def plot_stacked_bars_two_scenarios(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4=None, yaxis4=None,
                          np5=None, yaxis5=None, np6=None, yaxis6=None, np7=None, yaxis7=None,
                          scenario1="DACCU \n baseline", scenario2="DACCU \n climate \n neutrality",
                          scenario3="DACCS \n baseline", scenario4="DACCS \n climate \n neutrality",
                          scenario5 = "DACCU \n carbon \n neutrality", scenario6 = "DACCS \n carbon \n neutrality",
                          what="DAC_Gt", palette='magma', scenario = "Baseline", cumulative = False, stacked = True,
                         BAU_cost = None, np1_min=None, np1_max=None, np2_min=None, np2_max=None, np3_min=None, np3_max=None,
                         np4_min=None, np4_max=None, np5_min=None, np5_max=None, np6_min=None, np6_max=None,
                         np7_min=None, np7_max=None, hlines_divide = True, fmt_choice = 'none', error_label = None):

    n_colors = 3
    if np4 is not None:
        n_colors += 1
    if np5 is not None:
        n_colors += 1
    if np6 is not None:
        n_colors += 1
    if np7 is not None:
        n_colors += 1
    colors = sns.color_palette(palette, n_colors)

    if cumulative is True:
        tmp1 = np.zeros((3,2))
        tmp2 = np.zeros((3,2))
        tmp3 = np.zeros((3,2))
        if np4 is not None:
            tmp4 = np.zeros((3, 2))
        if np5 is not None:
            tmp5 = np.zeros((3, 2))
        if np6 is not None:
            tmp6 = np.zeros((3, 2))
        if np7 is not None:
            tmp7 = np.zeros((3, 2))
        for i in range(0, 3):
            for j in range(0, 2):
                tmp1[i, j] = np.sum(np1[i, j])
                tmp2[i, j] = np.sum(np2[i, j])
                tmp3[i, j] = np.sum(np3[i, j])
                if np4 is not None:
                    tmp4[i, j] = np.sum(np4[i, j])
                if np5 is not None:
                    tmp5[i, j] = np.sum(np5[i, j])
                if np6 is not None:
                    tmp6[i, j] = np.sum(np6[i, j])
                if np7 is not None:
                    tmp7[i, j] = np.sum(np7[i, j])
        data = {
            yaxis1: tmp1[0, 0],
            yaxis2: tmp2[0, 0],
            yaxis3: tmp3[0, 0]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[0, 0]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[0, 0]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[0, 0]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[0, 0]
        df1 = pd.DataFrame(data, index=[scenario1])

        data = {
            yaxis1: tmp1[1, 0],
            yaxis2: tmp2[1, 0],
            yaxis3: tmp3[1, 0]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[1, 0]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[1, 0]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[1, 0]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[1, 0]
        df2 = pd.DataFrame(data, index=[scenario2])

        data = {
            yaxis1: tmp1[0, 1],
            yaxis2: tmp2[0, 1],
            yaxis3: tmp3[0, 1]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[0, 1]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[0, 1]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[0, 1]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[0, 1]
        df3 = pd.DataFrame(data, index=[scenario3])

        data = {
            yaxis1: tmp1[1, 1],
            yaxis2: tmp2[1, 1],
            yaxis3: tmp3[1, 1]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[1, 1]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[1, 1]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[1, 1]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[1, 1]
        df4 = pd.DataFrame(data, index=[scenario4])

        data = {
            yaxis1: tmp1[2, 0],
            yaxis2: tmp2[2, 0],
            yaxis3: tmp3[2, 0]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[2, 0]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[2, 0]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[2, 0]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[2, 0]
        df5 = pd.DataFrame(data, index=[scenario5])

        data = {
            yaxis1: tmp1[2, 1],
            yaxis2: tmp2[2, 1],
            yaxis3: tmp3[2, 1]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[2, 1]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[2, 1]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[2, 1]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[2, 1]
        df6 = pd.DataFrame(data, index=[scenario6])

    else:
        index_year = year - 2020
        data = {
            yaxis1: np1[0, 0, index_year],
            yaxis2: np2[0, 0, index_year],
            yaxis3: np3[0, 0, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[0, 0, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[0, 0, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[0, 0, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[0, 0, index_year]
        df1 = pd.DataFrame(data, index=[scenario1])

        data = {
            yaxis1: np1[1, 0, index_year],
            yaxis2: np2[1, 0, index_year],
            yaxis3: np3[1, 0, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[1, 0, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[1, 0, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[1, 0, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[1, 0, index_year]
        df2 = pd.DataFrame(data, index=[scenario2])

        data = {
            yaxis1: np1[0, 1, index_year],
            yaxis2: np2[0, 1, index_year],
            yaxis3: np3[0, 1, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[0, 1, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[0, 1, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[0, 1, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[0, 1, index_year]
        df3 = pd.DataFrame(data, index=[scenario3])

        data = {
            yaxis1: np1[1, 1, index_year],
            yaxis2: np2[1, 1, index_year],
            yaxis3: np3[1, 1, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[1, 1, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[1,1, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[1, 1, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[1, 1, index_year]
        df4 = pd.DataFrame(data, index=[scenario4])

        data = {
            yaxis1: np1[2, 0, index_year],
            yaxis2: np2[2, 0, index_year],
            yaxis3: np3[2, 0, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[2, 0, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[2, 0, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[2, 0, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[2, 0, index_year]
        df5 = pd.DataFrame(data, index=[scenario5])

        data = {
            yaxis1: np1[2, 1, index_year],
            yaxis2: np2[2, 1, index_year],
            yaxis3: np3[2, 1, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[2, 1, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[2, 1, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[2, 1, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[2, 1, index_year]
        df6 = pd.DataFrame(data, index=[scenario6])

    if scenario == "Both":
        merged_df = pd.concat([df1, df2, df3, df4, df5, df6])
        index_order = [scenario5, scenario6, scenario1, scenario3, scenario2, scenario4]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#2f609c", "#e26e02", "#2f609c", "#e26e02", "#2f609c", "#e26e02"]
    elif scenario == 'Baseline':
        merged_df = pd.concat([df2, df4, df1, df3])
        index_order = [scenario1, scenario3, scenario2,
                       scenario4]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#2f609c", "#e26e02", "#2f609c", "#e26e02"]
    else:
        merged_df = pd.concat([df2, df4, df5, df6])
        index_order = [scenario5, scenario6, scenario2,
                       scenario4]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#2f609c", "#e26e02", "#2f609c", "#e26e02"]

    # Plot the stacked barplot

    ax = plot_df.plot.bar(stacked=stacked, color=colors)

    if BAU_cost is not None:
        if cumulative is False:
            BAU_cost_date = BAU_cost[index_year]
        else:
            BAU_cost_date = np.sum(BAU_cost)
        ax.axhline(y=BAU_cost_date, color='grey', linestyle='--', label = 'Business-As-Usual')  # You can customize the color and linestyle

    if cumulative is True:
        yerr_min = 0
        yerr_max = 0
        if np1_min is not None:
            tmp1_min = np.zeros((3,2))
            tmp1_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp1_min[i, j] = np.sum(np1_min[i, j])
                    tmp1_max[i, j] = np.sum(np1_max[i, j])
            yerr_min += np.abs(tmp1 - np.abs(tmp1_min))
            yerr_max += np.abs(np.abs(tmp1_max) - tmp1)

        if np2_min is not None:
            tmp2_min = np.zeros((3,2))
            tmp2_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp2_min[i, j] = np.sum(np2_min[i, j])
                    tmp2_max[i, j] = np.sum(np2_max[i, j])
            yerr_min += np.abs(tmp2 - np.abs(tmp2_min))
            yerr_max += np.abs(np.abs(tmp2_max) - tmp2)

        if np3_min is not None:
            tmp3_min = np.zeros((3,2))
            tmp3_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp3_min[i, j] = np.sum(np3_min[i, j])
                    tmp3_max[i, j] = np.sum(np3_max[i, j])
            yerr_min += np.abs(tmp3 - np.abs(tmp3_min))
            yerr_max += np.abs(np.abs(tmp3_max) - tmp3)

        if np4_min is not None:
            tmp4_min = np.zeros((3,2))
            tmp4_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp4_min[i, j] = np.sum(np4_min[i, j])
                    tmp4_max[i, j] = np.sum(np4_max[i, j])
            yerr_min += np.abs(tmp4 - np.abs(tmp4_min))
            yerr_max += np.abs(np.abs(tmp4_max) - tmp4)

        if np5_min is not None:
            tmp5_min = np.zeros((3,2))
            tmp5_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp5_min[i, j] = np.sum(np5_min[i, j])
                    tmp5_max[i, j] = np.sum(np5_max[i, j])
            yerr_min += np.abs(tmp5 - np.abs(tmp5_min))
            yerr_max += np.abs(np.abs(tmp5_max) - tmp5)

        if np6_min is not None:
            tmp6_min = np.zeros((3,2))
            tmp6_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp6_min[i, j] = np.sum(np6_min[i, j])
                    tmp6_max[i, j] = np.sum(np6_max[i, j])
            yerr_min += np.abs(tmp6 - np.abs(tmp6_min))
            yerr_max += np.abs(np.abs(tmp6_max) - tmp6)

        if np7_min is not None:
            tmp7_min = np.zeros((3,2))
            tmp7_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp7_min[i, j] = np.sum(np7_min[i, j])
                    tmp7_max[i, j] = np.sum(np7_max[i, j])
            yerr_min += np.abs(tmp7 - np.abs(tmp7_min))
            yerr_max += np.abs(np.abs(tmp7_max) - tmp7)

    else:
        yerr_min = np.zeros((3,2,41))
        yerr_max = np.zeros((3,2,41))
        highest_np_err = 0
        if np1_min is not None and np1_max is not None:
            yerr_min += np.abs(np1 - np.abs(np1_min))
            yerr_max += np.abs(np.abs(np1_max)-np1)
            highest_np_err += 1
        if np2_min is not None and np2_max is not None:
            yerr_min += np.abs(np2 - np.abs(np2_min))
            yerr_max += np.abs(np.abs(np2_max)-np2)
            highest_np_err = 2
        if np3_min is not None and np3_max is not None:
            yerr_min += np.abs(np3 - np.abs(np3_min))
            yerr_max += np.abs(np.abs(np3_max)-np3)
            highest_np_err = 3
        if np4_min is not None and np4_max is not None:
            yerr_min += np.abs(np4 - np.abs(np4_min))
            yerr_max += np.abs(np.abs(np4_max)-np4)
            highest_np_err = 4
        if np5_min is not None and np5_max is not None:
            yerr_min += np.abs(np5 - np.abs(np5_min))
            yerr_max += np.abs(np.abs(np5_max)-np5)
            highest_np_err = 5
        if np6_min is not None and np6_max is not None:
            yerr_min += np.abs(np6 - np.abs(np6_min))
            yerr_max += np.abs(np.abs(np6_max)-np6)
            highest_np_err = 6
        if np7_min is not None and np7_max is not None:
            yerr_min += np.abs(np7 - np.abs(np7_min))
            yerr_max += np.abs(np.abs(np7_max)-np7)
            highest_np_err = 7
    if np1_min is not None or np2_min is not None or np3_min is not None or np3_min is not None or np4_min is not None or np5_min is not None or np6_min is not None or np7_min is not None:
        final_yerr_min = np.zeros((3, 2))
        final_yerr_max = np.zeros((3, 2))
        if cumulative is True:
            for i in range(0, 3):
                for j in range(0, 2):
                    final_yerr_min[i, j] = np.sum(yerr_min[i, j])
                    final_yerr_max[i, j] = np.sum(yerr_max[i, j])
        else:
            final_yerr_min = yerr_min[:,:,index_year]
            final_yerr_max = yerr_max[:,:,index_year]
        f_yerr_min = {
            scenario1: final_yerr_min[0, 0],
            scenario2: final_yerr_min[1, 0],
            scenario3: final_yerr_min[0, 1],
            scenario4: final_yerr_min[1, 1],
            scenario5: final_yerr_min[2,0],
            scenario6: final_yerr_min[2,1]
        }
        df_yerr_min = pd.DataFrame({'yerr_min': f_yerr_min})
        f_yerr_max = {
            scenario1: final_yerr_max[0, 0],
            scenario2: final_yerr_max[1, 0],
            scenario3: final_yerr_max[0, 1],
            scenario4: final_yerr_max[1, 1],
            scenario5: final_yerr_max[2, 0],
            scenario6: final_yerr_max[2, 1]
        }
        df_yerr_max = pd.DataFrame({'yerr_max': f_yerr_max})

        plot_yerr_df = pd.concat([df_yerr_min, df_yerr_max], axis = 1).reindex(index_order)

        # Define the x positions for the error bars (assuming the same order as the scenarios)
        x_positions = np.arange(len(index_order))
        # Extract error bar values from the combined DataFrame
        error_bars = plot_yerr_df.to_numpy()
        # Plot the error bars
        ax.errorbar(x_positions, plot_df.sum(axis=1).values, yerr=error_bars.T, fmt=fmt_choice, color  = 'black', capsize=5, capthick=2, ecolor='black', label = error_label)

    # Remove the gray background
    ax.set_facecolor('white')

    # Set the title and labels
    if what == 'DAC_Gt' and cumulative is False:
        ylabel = "DAC rates (GtCO$_2$/year in "+str(year)+")"
        leg_loc = 'upper left'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'DAC_Gt' and cumulative is True:
        ylabel = "Cumulative DAC rates (GtCO$_2$)"
        leg_loc = 'upper left'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Total cost" and cumulative is False:
        ylabel = "Cost (billion €/year in "+str(year)+")"
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Total cost" and cumulative is True:
        ylabel = "Cost (trillion €)"
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost" and cumulative is False:
        ylabel = "Cost (€/year in "+str(year)+")"
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Cost' and cumulative is True:
        ylabel = "Cumulative cost (€)"
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Cost difference' and cumulative is True:
        ylabel = "Cumulative cost difference (trillion €) by " + str(year)
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost per emissions" and cumulative is False:
        ylabel = "Cost per emissions (€/tCO$_2$e*/year in "+str(year)+")"
        leg_loc = 'upper center'
        loc_text = [0.1, 0.75]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Cost per emissions' and cumulative is True:
        ylabel = "Cost per emissions (€/tCO$_2$e*)"
        leg_loc = 'upper center'
        loc_text = [0.1, 0.75]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost per DAC" and cumulative is False:
        ylabel = "Cost per installed DAC (€/tCO$_2$ in "+str(year)+")"
        leg_loc = 'upper center'
        loc_text = [0.25, 0.7]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Cost per DAC' and cumulative is True:
        ylabel = "Cost per installed DAC (€/tCO$_2$)"
        leg_loc = 'upper center'
        loc_text = [0.25, 0.7]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost per ton":
        ylabel = "Cost per ton (€/t) in "+ str(year)
        leg_loc = 'upper center'
        loc_text = [0.1, 0.75]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Emissions" and cumulative is False:
        ylabel = "Emissions (GtCO$_2$e*/year in "+str(year)+")"
        leg_loc = 'upper center'
        loc_text = [0.1, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Emissions' and cumulative is True:
        ylabel = "Emissions (GtCO$_2$e*)"
        leg_loc = 'upper left'
    elif what == "Electricity" or what == "Electricity_withDiesel" and cumulative is False:
        ylabel = "Electricity (MWh/year in "+str(year)+")"
        leg_loc = 'upper left'
    elif what == 'Electricity' or what == "Electricity_withDiesel"  and cumulative is True:
        ylabel = "Electricity (MWh)"
        leg_loc = 'upper left'
    elif what == "Change flight price" and cumulative is False:
        ylabel = "\u0394 flight price relative to BAU (%) in "+str(year)
        leg_loc = 'upper center'
        loc_text = [0.1, 0.65]
        num_columns = 3
        leg_box = (0.5, -0.075)
    elif what == "Cost neutrality per flight" and cumulative is False:
        ylabel = "Cost neutrality per flight (€/passenger) in "+str(year)
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 3
        leg_box = (0.5, -0.075)
    elif what == "Cost per liter fuel" and cumulative is False:
        ylabel = "Cost per liter fuel (€/L) in "+str(year)
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 3
        leg_box = (0.5, -0.075)
    else:
        ylabel = what
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)

    if scenario == 'decreasing demand':
        ax.text(0.45, 1.05, "-2% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'stagnating demand':
        ax.text(0.45, 1.05, "+0% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'historically observed demand':
        ax.text(0.45, 1.05, "+4% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'base demand growth':
        ax.text(0.45, 1.05, "+2% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'allnonCO2':
        ax.text(0.35, 1.05, "Offsetting contrails", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'noCC':
        ax.text(0.35, 1.05, "Not offsetting contrails", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'rerouting':
        ax.text(0.35, 1.05, "With rerouting", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')



    plt.ylabel(ylabel)

    # Rotate x-axis labels horizontally
    ax.tick_params(axis='x', rotation=0)
    tick_labels = [label.split()[0] if label.split()[0] in ["DACCS", "DACCU"] else "" for label in index_order]
    ax.set_xticklabels(tick_labels, fontweight='bold')

    # Iterate through tick labels and set custom colors
    for i, label in enumerate(tick_labels):
        if label == 'DACCU':
            ax.get_xticklabels()[i].set_color('#2f609c')  # Custom color for DACCS
        elif label == 'DACCS':
            ax.get_xticklabels()[i].set_color('#e26e02')  # Custom color for DACCU

    if hlines_divide is True:
        ax.axvline(1.5, ymin=0.05, ymax=0.95, color='grey')
        # Add text to the upper left and upper right
        ax.text(loc_text[0], 0.85, "CO$_2$\nneutrality", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
        ax.text(loc_text[1], 0.85, "Climate\nneutrality", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')

        # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc=leg_loc, ncol=num_columns, bbox_to_anchor=leg_box, frameon=False)
    if cumulative is False:
        ax.figure.savefig('Figures/Figures_final/'+what.replace(' ','_') + '_' + scenario.replace(' ','_') + "_" + str(year) +'.png', bbox_inches='tight', dpi=850, transparent = True)
    else:
        ax.figure.savefig('Figures/Figures_final/'+what.replace(' ','_') + '_' + scenario.replace(' ','_') + "_" + 'cumulative' +'.png', bbox_inches='tight', dpi=850, transparent = True)





def plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4 = None, yaxis4 = None,
                            np5 = None, yaxis5 = None, np6 = None, yaxis6 = None, np7 = None, yaxis7 = None,
                            start_year = 2020, palette = 'icefire', fuel = 'DACCU', scenario = "Baseline",
                         low_lim = None, upper_lim = None, ylabel = None, ax = None,
                         np1_min=None, np1_max=None, np2_min=None, np2_max=None, np3_min=None, np3_max=None,
                         np4_min=None, np4_max=None, np5_min=None, np5_max=None, np6_min=None, np6_max=None,
                         np7_min=None, np7_max=None
                         ):
    n_colors = 3
    if np4 is not None:
        n_colors += 1
    if np5 is not None:
        n_colors += 1
    if np6 is not None:
        n_colors += 1
    if np7 is not None:
        n_colors += 1
    colors = sns.color_palette(palette, n_colors)

    if fuel == 'DACCU':
        y = 0
    elif fuel == 'FF':
        y = 1

    if scenario == "Baseline":
        x = 0
    elif scenario == "Climate neutrality":
        x = 1
    elif scenario == "Carbon neutrality":
        x = 2
    else:
        x = 0

    data = {
    yaxis1: np1[x, y],
    yaxis2: np2[x, y],
    yaxis3: np3[x, y]
    }


    if np4 is not None and yaxis4 is not None:
        data[yaxis4] = np4[x, y]
    if np5 is not None and yaxis5 is not None:
        data[yaxis5] = np5[x, y]
    if np6 is not None and yaxis6 is not None:
        data[yaxis6] = np6[x, y]
    if np7 is not None and yaxis7 is not None:
        data[yaxis7] = np7[x, y]
    df = pd.DataFrame(data, index=np.arange(2061 - len(data[yaxis1]),2061))

    ax = ax or plt.gca()
    dates = np.arange(start=2061 -len(data[yaxis1]), stop=2061)
    ax.fill_between(dates, df[yaxis1], label=yaxis1, color=colors[0])
    #ax.hlines(0,year+1-len(data[yaxis1]), year + 1, linestyles="dashed", color='k')
    # Create a baseline that ensures stacking starts from zero
    baseline = df[yaxis1].clip(lower=0)
    if np.sum(df[yaxis2]) != 0:
        ax.fill_between(dates, baseline + df[yaxis2], baseline, label=yaxis2, color=colors[1])
        baseline += df[yaxis2].clip(lower=0)
    if np.sum(df[yaxis3]) != 0:
        ax.fill_between(dates, baseline + df[yaxis3], baseline, label=yaxis3, color=colors[2])
        baseline += df[yaxis3].clip(lower=0)
    if np4 is not None and np.sum(df[yaxis4]) != 0:
        ax.fill_between(dates, baseline + df[yaxis4], baseline, label=yaxis4, color=colors[3])
        baseline += df[yaxis4].clip(lower=0)
    if np5 is not None and np.sum(df[yaxis5]) != 0:
        ax.fill_between(dates, baseline + df[yaxis5], baseline, label=yaxis5, color=colors[4])
        baseline += df[yaxis5].clip(lower=0)
    if np6 is not None and np.sum(df[yaxis6]) != 0:
        ax.fill_between(dates, baseline + df[yaxis6], baseline, label=yaxis6, color=colors[5])
        baseline += df[yaxis6].clip(lower=0)
    if np7 is not None and np.sum(df[yaxis7]) != 0:
        ax.fill_between(dates, baseline + df[yaxis7], baseline, label=yaxis7, color=colors[6])
        baseline += df[yaxis7].clip(lower=0)

    yerr_min = np.zeros(len(dates))
    yerr_max = np.zeros(len(dates))
    highest_np_err = 0
    if np1_min is not None and np1_max is not None:
        yerr_min += np.abs(df[yaxis1] - np.abs(np1_min[x,y]))
        yerr_max += np.abs(np.abs(np1_max[x,y])-df[yaxis1])
        highest_np_err += 1
    if np2_min is not None and np2_max is not None:
        yerr_min += np.abs(df[yaxis2] - np.abs(np2_min[x,y]))
        yerr_max += np.abs(np.abs(np2_max[x,y])-df[yaxis2])
        highest_np_err = 2
    if np3_min is not None and np3_max is not None:
        yerr_min += np.abs(df[yaxis3] - np.abs(np3_min[x,y]))
        yerr_max += np.abs(np.abs(np3_max[x,y])-df[yaxis3])
        highest_np_err = 3
    if np4_min is not None and np4_max is not None:
        yerr_min += np.abs(df[yaxis4] - np.abs(np4_min[x,y]))
        yerr_max += np.abs(np.abs(np4_max[x,y])-df[yaxis4])
        highest_np_err = 4
    if np5_min is not None and np5_max is not None:
        yerr_min += np.abs(df[yaxis5] - np.abs(np5_min[x,y]))
        yerr_max += np.abs(np.abs(np5_max[x,y])-df[yaxis5])
        highest_np_err = 5
    if np6_min is not None and np6_max is not None:
        yerr_min += np.abs(df[yaxis6] - np.abs(np6_min[x,y]))
        yerr_max += np.abs(np.abs(np6_max[x,y])-df[yaxis6])
        highest_np_err = 6
    if np7_min is not None and np7_max is not None:
        yerr_min += np.abs(df[yaxis7] - np.abs(np7_min[x,y]))
        yerr_max += np.abs(np.abs(np7_max[x,y])-df[yaxis7])
        highest_np_err = 7

    # Add errorbars
    if n_colors <= 3:
        ax.errorbar(dates, df[yaxis1] + df[yaxis2] + df[yaxis3], yerr=[yerr_min, yerr_max], linestyle='None', color='grey')
    elif n_colors == 4:
        ax.errorbar(dates, df[yaxis1] + df[yaxis2] + df[yaxis3] + df[yaxis4], yerr=[yerr_min, yerr_max], linestyle='None', color='grey')
    elif n_colors == 5:
        ax.errorbar(dates, df[yaxis1] + df[yaxis2] + df[yaxis3] + df[yaxis4] + df[yaxis5], yerr=[yerr_min, yerr_max], linestyle='None', color='grey')
    elif n_colors == 6:
        ax.errorbar(dates, df[yaxis1] + df[yaxis2] + df[yaxis3] + df[yaxis4] + df[yaxis5] + df[yaxis6], yerr=[yerr_min, yerr_max], linestyle='None', color='grey')
    elif n_colors == 7:
        ax.errorbar(dates, df[yaxis1] + df[yaxis2] + df[yaxis3] + df[yaxis4] + df[yaxis5] + df[yaxis6] + df[yaxis7], yerr=[yerr_min, yerr_max], linestyle='None', color='grey')

    ax.set_ylim(low_lim, upper_lim)
    ax.set_xlim(2061 - len(data[yaxis1]), year)
    ax.set_ylabel(ylabel)
    return


def plot_time_series_two_scenarios(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4 = None, yaxis4 = None,
                            np5 = None, yaxis5 = None, np6 = None, yaxis6 = None, np7 = None, yaxis7 = None,
                         ylabel = None, scenario1 = None, scenario2 = None, scenario3 = None, scenario4 = None,
                   scenario5 = None, scenario6 = None, low_lim1 = None, up_lim1 = None, low_lim2 = None, up_lim2 = None,
                                low_lim3 = None, up_lim3 = None, type ='plot', palette= 'icefire', what = 'new', flip = False,
                                   np1_min = None, np1_max = None, np2_min = None, np2_max = None, np3_min = None, np3_max = None,
                                   np4_min = None, np4_max = None, np5_min = None, np5_max = None, np6_min = None, np6_max = None,
                                   np7_min = None, np7_max = None):
    n_plots = 2
    n_cols = 2
    n_rows = 1
    if scenario3 is not None:
        n_plots += 1
        n_cols += 1
    if scenario4 is not None:
        n_plots += 1
        n_cols -= 1
        n_rows += 1
    if scenario5 is not None:
        n_plots += 1
        n_rows += 1
    if scenario6 is not None:
        n_plots += 1

    # 2 scenarios --> ncols = 2, nrows = 1
    # 3 scenarios --> ncols = 3, nrows = 1
    # 4 scenarios --> ncols = 2, nrows = 2
    # 6 scenarios --> ncols = 2, nrows = 3

    if flip is True:
        tmp = n_rows
        n_rows = n_cols #n_rows becomes n_cols
        n_cols = tmp #n_cols becomes n_rows
    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(3*n_cols, 2.7*n_rows))

    #Access teh subplot axes dynamically based on n_cols and n_rows
    ax1 = ax.flatten()[0]
    ax2 = ax.flatten()[1]

    if n_cols == 2:
        col1 = "DACCU"
        col2 = "FF"
        text2 = "DACCS"
        if n_rows >= 1:
            if "baseline" in scenario1:
                rows1 = "Baseline"
            elif "carbon" in scenario1:
                rows1 = "Carbon neutrality"
            elif "baseline" and "carbon" not in scenario1:
                rows1 = "Climate neutrality"
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6, np7,
                                 yaxis7, low_lim = low_lim1, upper_lim = up_lim1, palette=palette,
                                 ylabel=ylabel, fuel=col1, scenario=rows1, ax=ax1,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6, np7,
                                 yaxis7, ylabel=ylabel, low_lim = low_lim1, upper_lim = up_lim1, palette=palette,
                                 fuel=col2, scenario=rows1, ax=ax2,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            ax1.text(0.5, 1.1, col1, transform=ax1.transAxes, va="top", ha="center")
            ax2.text(0.5, 1.1, text2, transform=ax2.transAxes, va="top", ha="center")
            ax1.text(0.05, 0.95, rows1, transform=ax1.transAxes, va="top", ha="left")
            ax2.text(0.05, 0.95, rows1, transform=ax2.transAxes, va="top", ha="left")
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            all_handles = handles1 + handles2
            all_labels = labels1 + labels2

        if n_rows >=2:
            ax3 = ax.flatten()[2]
            ax4 = ax.flatten()[3]
            if "baseline" in scenario3:
                rows2 = "Baseline"
            elif "carbon" in scenario3:
                rows2 = "Carbon neutrality"
            elif "baseline" and "carbon" not in scenario3:
                rows2 = "Climate neutrality"
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6, np7,
                                 yaxis7, low_lim = low_lim2, upper_lim = up_lim2, palette=palette,
                                 ylabel=ylabel, fuel=col1, scenario=rows2, ax=ax3,
                                 np1_min = np1_min, np1_max = np1_max, np2_min = np2_min, np2_max = np2_max, np3_min = np3_min, np3_max = np3_max,
                                 np4_min = np4_min, np4_max = np4_max, np5_min = np5_min, np5_max = np5_max, np6_min = np6_min, np6_max = np6_max,
                                 np7_min = np7_min, np7_max = np7_max)
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6, np7,
                                 yaxis7, ylabel=ylabel, low_lim = low_lim2, upper_lim = up_lim2, palette=palette,
                                 fuel=col2, scenario=rows2, ax=ax4,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            ax3.text(0.05, 0.95, rows2, transform=ax3.transAxes, va="top", ha="left")
            ax4.text(0.05, 0.95, rows2, transform=ax4.transAxes, va="top", ha="left")
            handles3, labels3 = ax3.get_legend_handles_labels()
            handles4, labels4 = ax4.get_legend_handles_labels()
            all_handles += handles3 + handles4
            all_labels += labels3 + labels4

        if n_rows >=3:
            ax5 = ax.flatten()[4]
            ax6 = ax.flatten()[5]
            if "baseline" in scenario5:
                rows3 = "Baseline"
            elif "carbon" in scenario5:
                rows3 = "Carbon neutrality"
            elif "baseline" and "carbon" not in scenario5:
                rows3 = "Climate neutrality"
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6, np7,
                                 yaxis7, low_lim = low_lim3, upper_lim = up_lim3, palette=palette,
                                 ylabel=ylabel, fuel=col1, scenario=rows3, ax=ax5,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6, np7,
                                 yaxis7, ylabel=ylabel, low_lim = low_lim3, upper_lim = up_lim3, palette=palette,
                                 fuel=col2, scenario=rows3, ax=ax6,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            ax5.text(0.05, 0.95, rows3, transform=ax5.transAxes, va="top", ha="left")
            ax6.text(0.05, 0.95, rows3, transform=ax6.transAxes, va="top", ha="left")
            handles5, labels5 = ax5.get_legend_handles_labels()
            handles6, labels6 = ax6.get_legend_handles_labels()
            all_handles += handles5 + handles6
            all_labels += labels5 + labels6

    elif n_cols == 3:
        col1 = "Carbon neutrality"
        col2 = "Baseline"
        col3 = "Climate neutrality"
        if n_rows >= 1:
            if "DACCU" in scenario1:
                rows1 = "DACCU"
                text1 = "DACCU"
            elif "DACCS" in scenario1:
                rows1 = "FF"
                text1 = "DACCS"
            ax3 = ax.flatten()[2]
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6,
                                 np7,
                                 yaxis7, low_lim = low_lim1, upper_lim = up_lim1,  palette=palette,
                                 ylabel=ylabel, fuel=rows1, scenario=col1, ax=ax1,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6,
                                 np7,
                                 yaxis7, ylabel=ylabel, low_lim = low_lim1, upper_lim = up_lim1,  palette=palette,
                                 fuel=rows1, scenario=col2, ax=ax2,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6,
                                 np7,
                                 yaxis7, ylabel=ylabel, low_lim = low_lim1, upper_lim = up_lim1,  palette=palette,
                                 fuel=rows1, scenario=col3, ax=ax3,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            ax1.text(0.5, 1.1, col1, transform=ax1.transAxes, va="top", ha="center")
            ax2.text(0.5, 1.1, col2, transform=ax2.transAxes, va="top", ha="center")
            ax3.text(0.5, 1.1, col3, transform=ax3.transAxes, va="top", ha="center")
            ax1.text(0.05, 0.95, text1, transform=ax1.transAxes, va="top", ha="left")
            ax2.text(0.05, 0.95, text1, transform=ax2.transAxes, va="top", ha="left")
            ax3.text(0.05, 0.95, text1, transform=ax3.transAxes, va="top", ha="left")
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles3, labels3 = ax3.get_legend_handles_labels()
            all_handles = handles1 + handles2 + handles3
            all_labels = labels1 + labels2 + labels3

        if n_rows >= 2:
            if "DACCU" in scenario2:
                rows2 = "DACCU"
                text2 = "DACCU"
            elif "DACCS" in scenario2:
                rows2 = "FF"
                text2 = "DACCS"
            ax4 = ax.flatten()[3]
            ax5 = ax.flatten()[4]
            ax6 = ax.flatten()[5]
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6,
                                 np7,
                                 yaxis7, low_lim = low_lim2, upper_lim = up_lim2,  palette=palette,
                                 ylabel=ylabel, fuel=rows2, scenario=col1, ax=ax4,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6,
                                 np7,
                                 yaxis7, ylabel=ylabel, low_lim = low_lim2, upper_lim = up_lim2,  palette=palette,
                                 fuel=rows2, scenario=col2, ax=ax5,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            plt_species_subplots(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4, yaxis4, np5, yaxis5, np6, yaxis6,
                                 np7,
                                 yaxis7, ylabel=ylabel, low_lim = low_lim2, upper_lim = up_lim2,  palette=palette,
                                 fuel=rows2, scenario=col3, ax=ax6,
                                 np1_min=np1_min, np1_max=np1_max, np2_min=np2_min, np2_max=np2_max, np3_min=np3_min,
                                 np3_max=np3_max,
                                 np4_min=np4_min, np4_max=np4_max, np5_min=np5_min, np5_max=np5_max, np6_min=np6_min,
                                 np6_max=np6_max,
                                 np7_min=np7_min, np7_max=np7_max
                                 )
            ax4.text(0.05, 0.95, text2, transform=ax4.transAxes, va="top", ha="left")
            ax5.text(0.05, 0.95, text2, transform=ax5.transAxes, va="top", ha="left")
            ax6.text(0.05, 0.95, text2, transform=ax6.transAxes, va="top", ha="left")
            handles1, labels1 = ax4.get_legend_handles_labels()
            handles2, labels2 = ax5.get_legend_handles_labels()
            handles3, labels3 = ax6.get_legend_handles_labels()
            all_handles += handles1 + handles2 + handles3
            all_labels += labels1 + labels2 + labels3

    # Remove duplicates
    unique_handles, unique_labels = [], []
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    if len(unique_handles) <= 3:
        leg_y = -0.05
        v_bottom = 0.05
        n_cols = 3
    elif len(unique_handles) > 3:
        if n_rows == 1:
            leg_y = -0.15
            v_bottom = 0.15
        elif n_rows > 1:
            leg_y = -0.075
            v_bottom = 0.1
        n_cols = 3

    if len(unique_handles) > 6:
        n_cols += 1
    fig.legend(handles=unique_handles, labels=unique_labels, bbox_to_anchor=(0.5, leg_y), loc="lower center",
               ncol=3, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=v_bottom)

    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 8.5
    plt.rcParams["xtick.labelsize"] = 8.5
    fig.savefig("Figures/Figures_final/TimeSeries_"+type+"_"+what.replace(' ', '_')+"_"+str(n_plots)+"_"+"scenarios.pdf", dpi=600, bbox_inches="tight")


def plot_heatmap_single_SA(df_sens_analysis, neutrality, comparison, baseline = None, policy = False):

    if baseline is not None:
        tmp1 = df_sens_analysis
        df_sens_analysis = (baseline - tmp1)/baseline*100
        pd.set_option('display.float_format', lambda x: f'{x:.1f}')
        scenario_name = "Change in \u0394 cost between DACCU and " + comparison + " [%]"
        unit = "percent"
    else:
        #scenario_name = "Difference between DACCU and " + comparison + " cost" + ' [trillion €]'
        scenario_name = 'DACCU cost penalty in 2050 [trillion €]'
        unit = "trillion€"

    if neutrality == "Climate neutrality":
        neutrality_save = "climate_neutral"
    elif neutrality == "Carbon neutrality":
        neutrality_save = "carbon_neutral"
    elif neutrality == "Baseline":
        neutrality_save = "baseline"
    else:
        neutrality_save = ""

    comparison_save = comparison.replace(' ', '_')

    custom_cmp = sns.diverging_palette(220, 20, as_cmap=True)

    plt.figure()

    sns.set(rc={'figure.figsize': (14, 11)})
    sns.set(font_scale=1.4)

    if policy is True:
        figure_heatmap_SA = sns.heatmap(df_sens_analysis, annot=True, linewidth=.5, center=0
                                        #, vmax=1.5, vmin=-1.5
                                        , cmap=custom_cmp, cbar_kws=
                                        {'label': "\n" + scenario_name, 'orientation': 'vertical'})
    else:
        figure_heatmap_SA = sns.heatmap(df_sens_analysis, annot=True, linewidth=.5, center=0
                                        #, vmax=1.5, vmin=-1.5
                                        , cmap=custom_cmp, cbar_kws=
                                        {'label': "\n" + scenario_name, 'orientation': 'horizontal'})
    figure_heatmap_SA.set_title(neutrality.replace('Carbon', 'CO$_2$') + " - DACCU vs. " + comparison + "\n", fontsize=18)
    figure_heatmap_SA.set_ylabel("\n Variation from assumed value \n", fontsize=16)


    figure_heatmap_SA
    if policy is True:
        figure_heatmap_SA.figure.savefig(
            'Figures/Figures_final/SA_single_factors_policy_DACCU_vs_' + comparison_save + "_" + neutrality_save + "_" + unit +
            '.png', bbox_inches='tight')
    else:
        figure_heatmap_SA.figure.savefig(
            'Figures/Figures_final/' + 'SA_single_factors_DACCU_vs_' + comparison_save + "_" + neutrality_save + "_" + unit +
            '.png', bbox_inches='tight')

def plot_tornado(df_sens_analysis, neutrality, comparison, baseline = None):
    if baseline is not None:
        tmp1 = df_sens_analysis
        df_sens_analysis = (baseline - tmp1)/baseline*100
        pd.set_option('display.float_format', lambda x: f'{x:.1f}')
        scenario_name = "Change in \u0394 cost between DACCU and " + comparison + " [%]"
        unit = "percent"
    else:
        scenario_name = "Difference between DACCU and " + comparison + " cost" + ' [trillion €]'
        unit = "trillion€"

    if neutrality == "Climate neutrality":
        neutrality_save = "climate_neutral"
    elif neutrality == "Carbon neutrality":
        neutrality_save = "carbon_neutral"
    elif neutrality == "Baseline":
        neutrality_save = "baseline"
    else:
        neutrality_save = ""

    comparison_save = comparison.replace(' ', '_')


    df_sens_analysis_T = df_sens_analysis.T

    col_names = list(df_sens_analysis.columns)
    column_names = [s.replace(' \n','') for s in col_names]
    df_sens_analysis_T.index = column_names
    df_ordered = df_sens_analysis_T.drop(columns=["-70%", "-30%", "0%", "+50%", "+150%", "+200%", "+250%", "+400%"]).sort_values(['-100%'], ascending = [True])
    df_sens_analysis_tornado = df_ordered.T
    category_names = list(df_ordered.columns)
    # Convert each column to a list and store them in the 'results' dictionary
    sens_analysis_dict = {col: df_sens_analysis_tornado[col].tolist() for col in df_sens_analysis_tornado.columns}

    # Convert the 'wished_dict' values into a NumPy array
    data = np.array(list(sens_analysis_dict.values()))
    labels  = list(df_sens_analysis_tornado.keys())

    data_cum = data.cumsum(axis=1)
    middle_index = data.shape[1]//2
    offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index] / 2

    category_colors = plt.get_cmap('coolwarm_r')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot Bars
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        if baseline is not None:
            starts = np.zeros((len(data[:,i])))
        else:
            starts = data_cum[:, i] - widths - offsets
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

    # Add Zero Reference Line
    ax.axvline(0, linestyle='--', color='black', alpha=.25)

    # X Axis
    #ax.set_xlim(-90, 90)
    #ax.set_xticks(np.arange(-90, 91, 10))
    ax.xaxis.set_major_formatter(lambda x, pos: str((int(x))))
    ax.set_xlabel(scenario_name)

    # Y Axis
    ax.invert_yaxis()

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Ledgend
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1), frameon = False,
              loc='lower left', fontsize='medium')

    # Set Background Color
    fig.set_facecolor('#FFFFFF')
    plt.show()

    fig.savefig(
        'Figures/Figures_final/' + 'SA_tornado_DACCU_vs_' + comparison_save + "_" + neutrality_save + "_" + unit +
        '.png', bbox_inches='tight')


def plot_single_variables_time_series(data_array, year, what = 'DAC', ylabel = 'DAC cost (€/tCO$_2$)', scenario1 = 'DACCU \n baseline', scenario2 = 'DACCS \n baseline',
                                      scenario3 = 'DACCU \n climate neutrality', scenario4 = 'DACCS \n climate neutrality',
                                      scenario5 = 'DACCU \n carbon neutrality', scenario6 = 'DACCS \n carbon neutrality',
                                      scenario = 'Both',
                                      colors = ['#ed7d31', '#3b87ce','#ce5d12','#2968a3','#f2a069','#6ca5da'],
                                      data_array_min = None, data_array_max = None):
    # Extract data for each time series
    DACCU_baseline = data_array[0, 0, :]  # Time series 1 from category 1 and subcategory 1
    DACCS_baseline = data_array[0, 1, :]  # Time series 2 from category 1 and subcategory 2
    DACCU_neutrality = data_array[1, 0, :]  # Time series 3 from category 2 and subcategory 1
    DACCS_neutrality = data_array[1, 1, :]  # Time series 4 from category 2 and subcategory 2
    DACCU_zerocarbon = data_array[2, 0, :]  # Time series 5 from category 3 and subcategory 1
    DACCS_zerocarbon = data_array[2, 1, :]  # Time series 6 from category 3 and subcategory 2

    if data_array_min is not None:
        # Extract data for each time series
        DACCU_baseline_min = data_array_min[0, 0, :]  # Time series 1 from category 1 and subcategory 1
        DACCS_baseline_min = data_array_min[0, 1, :]  # Time series 2 from category 1 and subcategory 2
        DACCU_neutrality_min = data_array_min[1, 0, :]  # Time series 3 from category 2 and subcategory 1
        DACCS_neutrality_min = data_array_min[1, 1, :]  # Time series 4 from category 2 and subcategory 2
        DACCU_zerocarbon_min = data_array_min[2, 0, :]  # Time series 5 from category 3 and subcategory 1
        DACCS_zerocarbon_min = data_array_min[2, 1, :]  # Time series 6 from category 3 and subcategory 2

    if data_array_max is not None:
        # Extract data for each time series
        DACCU_baseline_max = data_array_max[0, 0, :]  # Time series 1 from category 1 and subcategory 1
        DACCS_baseline_max = data_array_max[0, 1, :]  # Time series 2 from category 1 and subcategory 2
        DACCU_neutrality_max = data_array_max[1, 0, :]  # Time series 3 from category 2 and subcategory 1
        DACCS_neutrality_max = data_array_max[1, 1, :]  # Time series 4 from category 2 and subcategory 2
        DACCU_zerocarbon_max = data_array_max[2, 0, :]  # Time series 5 from category 3 and subcategory 1
        DACCS_zerocarbon_max = data_array_max[2, 1, :]  # Time series 6 from category 3 and subcategory 2

    # Create a time array for the x-axis (assuming 41 time steps)
    time_steps = np.arange(41)
    dates = np.arange(start=year+1-len(data_array[0,0]), stop=year + 1)


    # Plot each time series
    plt.figure(figsize=(8.5, 5))
    if scenario == 'Both':
        plt.plot(dates, DACCU_baseline, label=scenario1, color=colors[0])
        plt.plot(dates, DACCS_baseline, label=scenario2, color=colors[1])
        plt.plot(dates, DACCU_neutrality, label=scenario3, color=colors[2])
        plt.plot(dates, DACCS_neutrality, label=scenario4, color=colors[3])
        plt.plot(dates, DACCU_zerocarbon, label=scenario5, color=colors[4])
        plt.plot(dates, DACCS_zerocarbon, label=scenario6, color=colors[5])
        if data_array_min is not None:
            plt.fill_between(dates, DACCU_baseline_min, DACCU_baseline_max, label = scenario1, color=colors[0], alpha = 0.5)
            plt.fill_between(dates, DACCS_baseline_min, DACCS_baseline_max, label=scenario2, color=colors[1], alpha=0.5)
            plt.fill_between(dates, DACCU_neutrality_min, DACCU_neutrality_max, label=scenario3, color=colors[2], alpha=0.5)
            plt.fill_between(dates, DACCS_neutrality_min, DACCS_neutrality_max, label=scenario4, color=colors[3], alpha=0.5)
            plt.fill_between(dates, DACCU_zerocarbon_min, DACCU_zerocarbon_max, label=scenario5, color=colors[4], alpha=0.5)
            plt.fill_between(dates, DACCS_zerocarbon_min, DACCS_zerocarbon_max, label=scenario6, color=colors[5], alpha=0.5)
    elif scenario == 'Baseline':
        plt.plot(dates, DACCU_baseline, label=scenario1, color=colors[0])
        plt.plot(dates, DACCS_baseline, label=scenario2, color=colors[1])
        plt.plot(dates, DACCU_neutrality, label=scenario3, color=colors[4])
        plt.plot(dates, DACCS_neutrality, label=scenario4, color=colors[5])
        if data_array_min is not None:
            plt.fill_between(dates, DACCU_baseline_min, DACCU_baseline_max, label = scenario1, color=colors[0], alpha = 0.5)
            plt.fill_between(dates, DACCS_baseline_min, DACCS_baseline_max, label=scenario2, color=colors[1], alpha=0.5)
            plt.fill_between(dates, DACCU_neutrality_min, DACCU_neutrality_max, label=scenario3, color=colors[4], alpha=0.5)
            plt.fill_between(dates, DACCS_neutrality_min, DACCS_neutrality_max, label=scenario4, color=colors[5], alpha=0.5)
    elif scenario == 'Carbon neutrality':
        if data_array_min is not None:
            plt.fill_between(dates, DACCU_neutrality_min, DACCU_neutrality_max, label=scenario3, color=colors[0], alpha=0.5)
            plt.fill_between(dates, DACCS_neutrality_min, DACCS_neutrality_max, label=scenario4, color=colors[1], alpha=0.5)
            plt.fill_between(dates, DACCU_zerocarbon_min, DACCU_zerocarbon_max, label=scenario5, color=colors[4], alpha=0.5)
            plt.fill_between(dates, DACCS_zerocarbon_min, DACCS_zerocarbon_max, label=scenario6, color=colors[5], alpha=0.5)
        plt.plot(dates, DACCU_neutrality, label=scenario3, color=colors[0])
        plt.plot(dates, DACCS_neutrality, label=scenario4, color=colors[1])
        plt.plot(dates, DACCU_zerocarbon, label=scenario5, color=colors[4])
        plt.plot(dates, DACCS_zerocarbon, label=scenario6, color=colors[5])

    if DACCS_zerocarbon[-1] > DACCS_zerocarbon[0]:
        loc_leg = 'upper left'
    else:
        loc_leg = 'upper right'
    # Add labels and legend
    plt.ylabel(ylabel)
    plt.legend(frameon = False,
              loc=loc_leg, ncol = 2)

    # Show the plot
    plt.show()

    plt.savefig(
        'Figures/Figures_final/' + 'Single_timeseries_' + what.replace(' ','_') +
        '.png', bbox_inches='tight')


def plot_combinations(df_uncertain, what = 'Final cost', label = '(€/year)', cluster = False):
    # Extract the relevant columns
    #combinations = df_uncertain['Combination']
    df_uncertain_DACCU_baseline = df_uncertain[what+' DACCU baseline']
    df_uncertain_DACCS_baseline = df_uncertain[what+' DACCS baseline']
    df_uncertain_DACCU_neutrality = df_uncertain[what+' DACCU neutrality']
    df_uncertain_DACCS_neutrality = df_uncertain[what+' DACCS neutrality']
    df_uncertain_DACCU_carbon = df_uncertain[what+' DACCU carbon']
    df_uncertain_DACCS_carbon = df_uncertain[what+' DACCS carbon']

    min_baseline = min(df_uncertain_DACCU_baseline.min(), df_uncertain_DACCS_baseline.min())
    max_baseline = max(df_uncertain_DACCU_baseline.max(), df_uncertain_DACCS_baseline.max())
    min_neutrality = min(df_uncertain_DACCU_neutrality.min(), df_uncertain_DACCS_neutrality.min())
    max_neutrality = max(df_uncertain_DACCU_neutrality.max(), df_uncertain_DACCS_neutrality.max())
    min_carbon = min(df_uncertain_DACCU_carbon.min(), df_uncertain_DACCS_carbon.min())
    max_carbon = max(df_uncertain_DACCU_carbon.max(), df_uncertain_DACCS_carbon.max())


    num_intervals = 100

    # Create subplots for each scenario
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 7))
    plt.tight_layout()

    # Color mapping based on clusters
    if cluster is True:
        colors = df_uncertain['Cluster']

    # Plot scatter plots with color mapping
    scatter = axes[0].scatter(df_uncertain_DACCU_baseline, df_uncertain_DACCS_baseline, c=colors, cmap='gist_ncar',
                              alpha=0.5)

    # Plot scatter plots
    #axes[0].scatter(df_uncertain_DACCU_baseline, df_uncertain_DACCS_baseline, alpha=0.5)
    axes[0].set_title('DACCU baseline vs DACCS baseline')
    axes[0].plot(np.linspace(min_baseline, max_baseline, num_intervals), np.linspace(min_baseline, max_baseline, num_intervals), color = 'k')
    axes[0].set_xlim(min_baseline, max_baseline)
    axes[0].set_ylim(min_baseline, max_baseline)

    scatter = axes[1].scatter(df_uncertain_DACCU_neutrality, df_uncertain_DACCS_neutrality, c=colors, cmap='gist_ncar',
                              alpha=0.5)
    #axes[1].scatter(df_uncertain_DACCU_neutrality, df_uncertain_DACCS_neutrality, alpha=0.5)
    axes[1].set_title('DACCU neutrality vs DACCS neutrality')
    axes[1].plot(np.linspace(min_neutrality, max_neutrality, num_intervals), np.linspace(min_neutrality, max_neutrality, num_intervals), color = 'k')
    axes[1].set_xlim(min_neutrality, max_neutrality)
    axes[1].set_ylim(min_neutrality, max_neutrality)

    scatter = axes[2].scatter(df_uncertain_DACCU_carbon, df_uncertain_DACCS_carbon, c=colors, cmap='gist_ncar',
                              alpha=0.5)
    #axes[2].scatter(df_uncertain_DACCU_carbon, df_uncertain_DACCS_carbon, alpha=0.5)
    axes[2].set_title('DACCU carbon vs DACCS carbon')
    axes[2].plot(np.linspace(min_carbon, max_carbon, num_intervals), np.linspace(min_carbon, max_carbon, num_intervals), color = 'k')
    axes[2].set_xlim(min_carbon, max_carbon)
    axes[2].set_ylim(min_carbon, max_carbon)

    # Customize labels
    for ax in axes.flat:
        ax.set_xlabel('Final Cost DACCU' + ' ' + label)
        ax.set_ylabel('Final Cost DACCS'+ ' ' + label)

    # Show the plots
    plt.show()

def plot_mean_cluster_per_variable(mean_df, var1, var2):
    # Plot Final cost against cc_efficacy
    # Plot Final cost against cc_efficacy and electricity_cost using colors
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_df[var1], mean_df['Final cost DACCU baseline'], c=mean_df[var2], cmap='coolwarm',
                label='DACCU baseline')
    plt.scatter(mean_df[var1], mean_df['Final cost DACCS baseline'], c=mean_df[var2], cmap='coolwarm',
                marker='s', label='DACCS baseline')
    plt.colorbar(label=var2.replace('_', ' '))
    plt.xlabel(var1.replace('_', ' '))
    plt.ylabel('Final cost')
    plt.title('Final cost vs '+var1.replace('_',' '))
    plt.legend()
    plt.show()


def plot_heatmap_SA_II(df_sens_analysis, neutrality, comparison, var_para_1, var_para_2):
    if neutrality == "Climate neutrality":
        neutrality_save = "climate_neutral"
    elif neutrality == "Carbon neutrality":
        neutrality_save = "carbon_neutral"
    elif neutrality == "Baseline":
        neutrality_save = "baseline"
    else:
        neutrality_save = ""

    scenario_name = "Cost difference between DACCU and " + comparison
    comparison_save = comparison.replace(' ', '_')

    custom_cmp = sns.diverging_palette(220, 20, as_cmap=True)

    plt.figure()

    sns.set(rc={'figure.figsize': (14, 11)})
    sns.set(font_scale=1.4)

    figure_heatmap_SA = sns.heatmap(df_sens_analysis, annot=True, linewidth=.5, center=0
                                    #, vmax=0.7, vmin=-1.5
                                    , cmap=custom_cmp, cbar_kws=
                                    {'label': "\n" + scenario_name + ' [trillion €]', 'orientation': 'horizontal'})
    figure_heatmap_SA.set_title("\n" + scenario_name + "\n" + neutrality + "\n", fontsize=18)

    if var_para_1 == "electricity cost":
        figure_heatmap_SA.set_xlabel("\n Electricity cost [€/KWh]\n", fontsize=16)
        var1 = "electricity"
    elif var_para_1 == "learning rate":
        figure_heatmap_SA.set_xlabel("\n Learning rate \n", fontsize=16)
        var1 = "learning"
    elif var_para_1 == "growth rate":
        figure_heatmap_SA.set_xlabel("\n Demand growth rate \n", fontsize=16)
        var1 = "growth"
    elif var_para_1 == "fossil fuel cost":
        figure_heatmap_SA.set_xlabel("\n Fossil fuel cost [€/L] \n", fontsize=16)
        var1 = "FF"
    elif var_para_1 == "DAC initial cost":
        figure_heatmap_SA.set_xlabel("\n DAC initial cost [€/tCO$_2$] \n", fontsize=16)
        var1 = "DAC"
    elif var_para_1 == "jetfuel allocation":
        figure_heatmap_SA.set_xlabel("\n Share of jet fuel in Fischer-Tropsch output [%] \n", fontsize=16)
        var1 = "jetfuel_allocation"
    elif var_para_1 == "cc efficacy":
        figure_heatmap_SA.set_xlabel("\n Contrail cirruc efficacy [%] \n", fontsize=16)
        var1 = "cc_efficacy"

    if var_para_2 == "electricity cost":
        figure_heatmap_SA.set_ylabel("\n Electricity cost [€/KWh]\n", fontsize=16)
        var2 = "electricity"
    elif var_para_2 == "learning rate":
        figure_heatmap_SA.set_ylabel("\n Learning rate \n", fontsize=16)
        var2 = "learning"
    elif var_para_2 == "growth rate":
        figure_heatmap_SA.set_ylabel("\n Demand growth rate \n", fontsize=16)
        var2 = "growth"
    elif var_para_2 == "fossil fuel cost":
        figure_heatmap_SA.set_ylabel("\n Fossil fuel cost [€/L] \n", fontsize=16)
        var2 = "FF"
    elif var_para_2 == "DAC initial cost":
        figure_heatmap_SA.set_ylabel("\n DAC initial cost [€/tCO$_2$] \n", fontsize=16)
        var2 = "DAC"
    elif var_para_2 == "jetfuel allocation":
        figure_heatmap_SA.set_ylabel("\n Share of jet fuel in Fischer-Tropsch output [%] \n", fontsize=16)
        var2 = "jetfuel_allocation"
    elif var_para_2 == "cc efficacy":
        figure_heatmap_SA.set_ylabel("\n Contrail cirruc efficacy [%] \n", fontsize=16)
        var2 = "cc_efficacy"

    figure_heatmap_SA
    figure_heatmap_SA.figure.savefig(
        'Figures/Figures_final/' + 'SA_DACCU_vs_' + comparison_save + "_" + neutrality_save + "_" + var1 + "_" + var2 +
        '.png', bbox_inches='tight')


def plot_stacked_bars_four_scenarios(np1, np2, np3, year, yaxis1, yaxis2, yaxis3, np4=None, yaxis4=None,
                          np5=None, yaxis5=None, np6=None, yaxis6=None, np7=None, yaxis7=None,
                          scenario1="DACCU  \n climate \n neutrality", scenario2="DACCS \n climate \n neutrality",
                          scenario3="DACCU \n stagnant \n demand", scenario4="DACCS \n stagnant \n demand",
                            scenario5="DACCU \n decreasing \n demand", scenario6="DACCS \n decreasing \n demand",
                          scenario7 = None, scenario8 = None, scenario9 = None, scenario10 = None,
                          what="DAC_Gt", palette='magma', scenario = "Baseline", cumulative = False, stacked = True,
                         BAU_cost = None, np1_min=None, np1_max=None, np2_min=None, np2_max=None, np3_min=None, np3_max=None,
                         np4_min=None, np4_max=None, np5_min=None, np5_max=None, np6_min=None, np6_max=None,
                         np7_min=None, np7_max=None, hlines_divide = True, fmt_choice = 'none', error_label = None,
                                     BAU_label = 'Business-as-Usual'):

    n_colors = 3
    if np4 is not None:
        n_colors += 1
    if np5 is not None:
        n_colors += 1
    if np6 is not None:
        n_colors += 1
    if np7 is not None:
        n_colors += 1
    colors = sns.color_palette(palette, n_colors)

    if cumulative is True:
        tmp1 = np.zeros((3,2))
        tmp2 = np.zeros((3,2))
        tmp3 = np.zeros((3,2))
        if np4 is not None:
            tmp4 = np.zeros((3, 2))
        if np5 is not None:
            tmp5 = np.zeros((3, 2))
        if np6 is not None:
            tmp6 = np.zeros((3, 2))
        if np7 is not None:
            tmp7 = np.zeros((3, 2))
        for i in range(0, 3):
            for j in range(0, 2):
                tmp1[i, j] = np.sum(np1[i, j])
                tmp2[i, j] = np.sum(np2[i, j])
                tmp3[i, j] = np.sum(np3[i, j])
                if np4 is not None:
                    tmp4[i, j] = np.sum(np4[i, j])
                if np5 is not None:
                    tmp5[i, j] = np.sum(np5[i, j])
                if np6 is not None:
                    tmp6[i, j] = np.sum(np6[i, j])
                if np7 is not None:
                    tmp7[i, j] = np.sum(np7[i, j])
        data = {
            yaxis1: tmp1[0, 0],
            yaxis2: tmp2[0, 0],
            yaxis3: tmp3[0, 0]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[0, 0]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[0, 0]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[0, 0]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[0, 0]
        df1 = pd.DataFrame(data, index=[scenario1])

        data = {
            yaxis1: tmp1[1, 0],
            yaxis2: tmp2[1, 0],
            yaxis3: tmp3[1, 0]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[1, 0]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[1, 0]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[1, 0]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[1, 0]
        df2 = pd.DataFrame(data, index=[scenario2])

        data = {
            yaxis1: tmp1[0, 1],
            yaxis2: tmp2[0, 1],
            yaxis3: tmp3[0, 1]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[0, 1]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[0, 1]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[0, 1]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[0, 1]
        df3 = pd.DataFrame(data, index=[scenario3])

        data = {
            yaxis1: tmp1[1, 1],
            yaxis2: tmp2[1, 1],
            yaxis3: tmp3[1, 1]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[1, 1]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[1, 1]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[1, 1]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[1, 1]
        df4 = pd.DataFrame(data, index=[scenario4])

        data = {
            yaxis1: tmp1[2, 0],
            yaxis2: tmp2[2, 0],
            yaxis3: tmp3[2, 0]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[2, 0]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[2, 0]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[2, 0]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[2, 0]
        df5 = pd.DataFrame(data, index=[scenario5])

        data = {
            yaxis1: tmp1[2, 1],
            yaxis2: tmp2[2, 1],
            yaxis3: tmp3[2, 1]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = tmp4[2, 1]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = tmp5[2, 1]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = tmp6[2, 1]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = tmp7[2, 1]
        df6 = pd.DataFrame(data, index=[scenario6])

        if scenario7 is not None:
            data = {
                yaxis1: tmp1[3, 0],
                yaxis2: tmp2[3, 0],
                yaxis3: tmp3[3, 0]
            }
            if np4 is not None and yaxis4 is not None:
                data[yaxis4] = tmp4[3, 0]
            if np5 is not None and yaxis5 is not None:
                data[yaxis5] = tmp5[3, 0]
            if np6 is not None and yaxis6 is not None:
                data[yaxis6] = tmp6[3, 0]
            if np7 is not None and yaxis7 is not None:
                data[yaxis7] = tmp7[3, 0]
            df7 = pd.DataFrame(data, index=[scenario7])

        if scenario8 is not None:
            data = {
                yaxis1: tmp1[3, 1],
                yaxis2: tmp2[3, 1],
                yaxis3: tmp3[3, 1]
            }
            if np4 is not None and yaxis4 is not None:
                data[yaxis4] = tmp4[3, 1]
            if np5 is not None and yaxis5 is not None:
                data[yaxis5] = tmp5[3, 1]
            if np6 is not None and yaxis6 is not None:
                data[yaxis6] = tmp6[3, 1]
            if np7 is not None and yaxis7 is not None:
                data[yaxis7] = tmp7[3, 1]
            df8 = pd.DataFrame(data, index=[scenario8])

    else:
        index_year = year - 2020
        data = {
            yaxis1: np1[0, 0, index_year],
            yaxis2: np2[0, 0, index_year],
            yaxis3: np3[0, 0, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[0, 0, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[0, 0, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[0, 0, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[0, 0, index_year]
        df1 = pd.DataFrame(data, index=[scenario1])

        data = {
            yaxis1: np1[1, 0, index_year],
            yaxis2: np2[1, 0, index_year],
            yaxis3: np3[1, 0, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[1, 0, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[1, 0, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[1, 0, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[1, 0, index_year]
        df2 = pd.DataFrame(data, index=[scenario3])

        data = {
            yaxis1: np1[0, 1, index_year],
            yaxis2: np2[0, 1, index_year],
            yaxis3: np3[0, 1, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[0, 1, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[0, 1, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[0, 1, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[0, 1, index_year]
        df3 = pd.DataFrame(data, index=[scenario2])

        data = {
            yaxis1: np1[1, 1, index_year],
            yaxis2: np2[1, 1, index_year],
            yaxis3: np3[1, 1, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[1, 1, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[1,1, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[1, 1, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[1, 1, index_year]
        df4 = pd.DataFrame(data, index=[scenario4])

        data = {
            yaxis1: np1[2, 0, index_year],
            yaxis2: np2[2, 0, index_year],
            yaxis3: np3[2, 0, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[2, 0, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[2, 0, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[2, 0, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[2, 0, index_year]
        df5 = pd.DataFrame(data, index=[scenario5])

        data = {
            yaxis1: np1[2, 1, index_year],
            yaxis2: np2[2, 1, index_year],
            yaxis3: np3[2, 1, index_year]
        }
        if np4 is not None and yaxis4 is not None:
            data[yaxis4] = np4[2, 1, index_year]
        if np5 is not None and yaxis5 is not None:
            data[yaxis5] = np5[2, 1, index_year]
        if np6 is not None and yaxis6 is not None:
            data[yaxis6] = np6[2, 1, index_year]
        if np7 is not None and yaxis7 is not None:
            data[yaxis7] = np7[2, 1, index_year]
        df6 = pd.DataFrame(data, index=[scenario6])


        if scenario7 is not None:
            data = {
                yaxis1: np1[3, 0, index_year],
                yaxis2: np2[3, 0, index_year],
                yaxis3: np3[3, 0, index_year]
            }
            if np4 is not None and yaxis4 is not None:
                data[yaxis4] = np4[3, 0, index_year]
            if np5 is not None and yaxis5 is not None:
                data[yaxis5] = np5[3, 0, index_year]
            if np6 is not None and yaxis6 is not None:
                data[yaxis6] = np6[3, 0, index_year]
            if np7 is not None and yaxis7 is not None:
                data[yaxis7] = np7[3, 0, index_year]
            df7 = pd.DataFrame(data, index=[scenario7])

        if scenario8 is not None:
            data = {
                yaxis1: np1[3, 1, index_year],
                yaxis2: np2[3, 1, index_year],
                yaxis3: np3[3, 1, index_year]
            }
            if np4 is not None and yaxis4 is not None:
                data[yaxis4] = np4[3, 1, index_year]
            if np5 is not None and yaxis5 is not None:
                data[yaxis5] = np5[3, 1, index_year]
            if np6 is not None and yaxis6 is not None:
                data[yaxis6] = np6[3, 1, index_year]
            if np7 is not None and yaxis7 is not None:
                data[yaxis7] = np7[3, 1, index_year]
            df8 = pd.DataFrame(data, index=[scenario8])

        if scenario9 is not None:
            data = {
                yaxis1: np1[4, 0, index_year],
                yaxis2: np2[4, 0, index_year],
                yaxis3: np3[4, 0, index_year]
            }
            if np4 is not None and yaxis4 is not None:
                data[yaxis4] = np4[4, 0, index_year]
            if np5 is not None and yaxis5 is not None:
                data[yaxis5] = np5[4, 0, index_year]
            if np6 is not None and yaxis6 is not None:
                data[yaxis6] = np6[4, 0, index_year]
            if np7 is not None and yaxis7 is not None:
                data[yaxis7] = np7[4, 0, index_year]
            df9 = pd.DataFrame(data, index=[scenario9])

        if scenario10 is not None:
            data = {
                yaxis1: np1[4, 1, index_year],
                yaxis2: np2[4, 1, index_year],
                yaxis3: np3[4, 1, index_year]
            }
            if np4 is not None and yaxis4 is not None:
                data[yaxis4] = np4[4, 1, index_year]
            if np5 is not None and yaxis5 is not None:
                data[yaxis5] = np5[4, 1, index_year]
            if np6 is not None and yaxis6 is not None:
                data[yaxis6] = np6[4, 1, index_year]
            if np7 is not None and yaxis7 is not None:
                data[yaxis7] = np7[4, 1, index_year]
            df10 = pd.DataFrame(data, index=[scenario10])


    if scenario == "Both":
        merged_df = pd.concat([df1, df2, df3, df4, df5, df6])
        index_order = [scenario5, scenario6, scenario1, scenario3, scenario2, scenario4]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#2f609c", "#e26e02", "#2f609c", "#e26e02", "#2f609c", "#e26e02"]
    elif scenario == 'Baseline':
        merged_df = pd.concat([df2, df4, df1, df3])
        index_order = [scenario1, scenario3, scenario2,
                       scenario4]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#2f609c", "#e26e02", "#2f609c", "#e26e02"]
    elif scenario == 'Contrails options':
        merged_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
        index_order = [scenario1, scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#2f609c", "#e26e02", "#2f609c", "#e26e02", "#2f609c", "#e26e02", "#2f609c", "#e26e02"]
    elif scenario == 'Scenarios explanation':
        merged_df = pd.concat([df2, df3, df4, df5, df6])
        index_order = [scenario2, scenario3, scenario4,
                       scenario5, scenario6]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#FF5733", "#2f609c", "#e26e02", "#2f609c", "#e26e02"]
    elif scenario == 'Climate neutrality only':
        merged_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10])
        index_order = [scenario1, scenario2, scenario3,
                       scenario4, scenario5, scenario6,
                       scenario7, scenario8, scenario9, scenario10]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#FF5733", "#2f609c", "#e26e02", "#2f609c", "#e26e02"]
    else:
        merged_df = pd.concat([df2, df4, df5, df6])
        index_order = [scenario5, scenario6, scenario2,
                       scenario4]  # Specify the desired order of columns
        plot_df = merged_df.reindex(index_order)
        colors_xticks = ["#2f609c", "#e26e02", "#2f609c", "#e26e02"]

    # Plot the stacked barplot

    ax = plot_df.plot.bar(stacked=stacked, color=colors)

    if BAU_cost is not None:
        if cumulative is False:
            BAU_cost_date = BAU_cost[index_year]
        else:
            BAU_cost_date = np.sum(BAU_cost)
        ax.axhline(y=BAU_cost_date, color='grey', linestyle='--', label = BAU_label)  # You can customize the color and linestyle

    if cumulative is True:
        yerr_min = 0
        yerr_max = 0
        if np1_min is not None:
            tmp1_min = np.zeros((3,2))
            tmp1_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp1_min[i, j] = np.sum(np1_min[i, j])
                    tmp1_max[i, j] = np.sum(np1_max[i, j])
            yerr_min += np.abs(tmp1 - np.abs(tmp1_min))
            yerr_max += np.abs(np.abs(tmp1_max) - tmp1)

        if np2_min is not None:
            tmp2_min = np.zeros((3,2))
            tmp2_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp2_min[i, j] = np.sum(np2_min[i, j])
                    tmp2_max[i, j] = np.sum(np2_max[i, j])
            yerr_min += np.abs(tmp2 - np.abs(tmp2_min))
            yerr_max += np.abs(np.abs(tmp2_max) - tmp2)

        if np3_min is not None:
            tmp3_min = np.zeros((3,2))
            tmp3_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp3_min[i, j] = np.sum(np3_min[i, j])
                    tmp3_max[i, j] = np.sum(np3_max[i, j])
            yerr_min += np.abs(tmp3 - np.abs(tmp3_min))
            yerr_max += np.abs(np.abs(tmp3_max) - tmp3)

        if np4_min is not None:
            tmp4_min = np.zeros((3,2))
            tmp4_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp4_min[i, j] = np.sum(np4_min[i, j])
                    tmp4_max[i, j] = np.sum(np4_max[i, j])
            yerr_min += np.abs(tmp4 - np.abs(tmp4_min))
            yerr_max += np.abs(np.abs(tmp4_max) - tmp4)

        if np5_min is not None:
            tmp5_min = np.zeros((3,2))
            tmp5_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp5_min[i, j] = np.sum(np5_min[i, j])
                    tmp5_max[i, j] = np.sum(np5_max[i, j])
            yerr_min += np.abs(tmp5 - np.abs(tmp5_min))
            yerr_max += np.abs(np.abs(tmp5_max) - tmp5)

        if np6_min is not None:
            tmp6_min = np.zeros((3,2))
            tmp6_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp6_min[i, j] = np.sum(np6_min[i, j])
                    tmp6_max[i, j] = np.sum(np6_max[i, j])
            yerr_min += np.abs(tmp6 - np.abs(tmp6_min))
            yerr_max += np.abs(np.abs(tmp6_max) - tmp6)

        if np7_min is not None:
            tmp7_min = np.zeros((3,2))
            tmp7_max = np.zeros((3,2))
            for i in range(0, 3):
                for j in range(0, 2):
                    tmp7_min[i, j] = np.sum(np7_min[i, j])
                    tmp7_max[i, j] = np.sum(np7_max[i, j])
            yerr_min += np.abs(tmp7 - np.abs(tmp7_min))
            yerr_max += np.abs(np.abs(tmp7_max) - tmp7)

    else:
        if scenario == "Contrails options":
            yerr_min = np.zeros((4, 2, 41))
            yerr_max = np.zeros((4, 2, 41))
        elif scenario == 'Climate neutrality only':
            yerr_min = np.zeros((5, 2, 41))
            yerr_max = np.zeros((5, 2, 41))
        else:
            yerr_min = np.zeros((3,2,41))
            yerr_max = np.zeros((3,2,41))
        highest_np_err = 0
        if np1_min is not None and np1_max is not None:
            yerr_min += np.abs(np1 - np.abs(np1_min))
            yerr_max += np.abs(np.abs(np1_max)-np1)
            highest_np_err += 1
        if np2_min is not None and np2_max is not None:
            yerr_min += np.abs(np2 - np.abs(np2_min))
            yerr_max += np.abs(np.abs(np2_max)-np2)
            highest_np_err = 2
        if np3_min is not None and np3_max is not None:
            yerr_min += np.abs(np3 - np.abs(np3_min))
            yerr_max += np.abs(np.abs(np3_max)-np3)
            highest_np_err = 3
        if np4_min is not None and np4_max is not None:
            yerr_min += np.abs(np4 - np.abs(np4_min))
            yerr_max += np.abs(np.abs(np4_max)-np4)
            highest_np_err = 4
        if np5_min is not None and np5_max is not None:
            yerr_min += np.abs(np5 - np.abs(np5_min))
            yerr_max += np.abs(np.abs(np5_max)-np5)
            highest_np_err = 5
        if np6_min is not None and np6_max is not None:
            yerr_min += np.abs(np6 - np.abs(np6_min))
            yerr_max += np.abs(np.abs(np6_max)-np6)
            highest_np_err = 6
        if np7_min is not None and np7_max is not None:
            yerr_min += np.abs(np7 - np.abs(np7_min))
            yerr_max += np.abs(np.abs(np7_max)-np7)
            highest_np_err = 7
    if np1_min is not None or np2_min is not None or np3_min is not None or np3_min is not None or np4_min is not None or np5_min is not None or np6_min is not None or np7_min is not None:
        if scenario == 'Contrails options':
            final_yerr_min = np.zeros((4, 2))
            final_yerr_max = np.zeros((4, 2))
        elif scenario == 'Climate neutrality only':
            final_yerr_min = np.zeros((4, 2))
            final_yerr_max = np.zeros((4, 2))
        else:
            final_yerr_min = np.zeros((3, 2))
            final_yerr_max = np.zeros((3, 2))
        if cumulative is True:
            for i in range(0, 3):
                for j in range(0, 2):
                    final_yerr_min[i, j] = np.sum(yerr_min[i, j])
                    final_yerr_max[i, j] = np.sum(yerr_max[i, j])
        else:
            final_yerr_min = yerr_min[:,:,index_year]
            final_yerr_max = yerr_max[:,:,index_year]
        if scenario == "Contrails options":
            f_yerr_min = {
                scenario1: final_yerr_min[0, 0],
                scenario2: final_yerr_min[0, 1],
                scenario3: final_yerr_min[1, 0],
                scenario4: final_yerr_min[1, 1],
                scenario5: final_yerr_min[2, 0],
                scenario6: final_yerr_min[2, 1],
                scenario7: final_yerr_min[3, 0],
                scenario8: final_yerr_min[3, 1],
            }
            f_yerr_max = {
                scenario1: final_yerr_max[0, 0],
                scenario2: final_yerr_max[0, 1],
                scenario3: final_yerr_max[1, 0],
                scenario4: final_yerr_max[1, 1],
                scenario5: final_yerr_max[2, 0],
                scenario6: final_yerr_max[2, 1],
                scenario7: final_yerr_max[3, 0],
                scenario8: final_yerr_max[3, 1],
            }
        elif scenario == "Climate neutrality only":
            f_yerr_min = {
                scenario1: final_yerr_min[0, 0],
                scenario2: final_yerr_min[0, 1],
                scenario3: final_yerr_min[1, 0],
                scenario4: final_yerr_min[1, 1],
                scenario5: final_yerr_min[2, 0],
                scenario6: final_yerr_min[2, 1],
                scenario7: final_yerr_min[3, 0],
                scenario8: final_yerr_min[3, 1],
                scenario9: final_yerr_min[4, 0],
                scenario10: final_yerr_min[4, 1],
            }
            f_yerr_max = {
                scenario1: final_yerr_max[0, 0],
                scenario2: final_yerr_max[0, 1],
                scenario3: final_yerr_max[1, 0],
                scenario4: final_yerr_max[1, 1],
                scenario5: final_yerr_max[2, 0],
                scenario6: final_yerr_max[2, 1],
                scenario7: final_yerr_max[3, 0],
                scenario8: final_yerr_max[3, 1],
                scenario9: final_yerr_max[4, 0],
                scenario10: final_yerr_max[4, 1],
            }
        else:
            f_yerr_min = {
                scenario1: final_yerr_min[0, 0],
                scenario2: final_yerr_min[1, 0],
                scenario3: final_yerr_min[0, 1],
                scenario4: final_yerr_min[1, 1],
                scenario5: final_yerr_min[2,0],
                scenario6: final_yerr_min[2,1]
            }
            f_yerr_max = {
                scenario1: final_yerr_max[0, 0],
                scenario2: final_yerr_max[1, 0],
                scenario3: final_yerr_max[0, 1],
                scenario4: final_yerr_max[1, 1],
                scenario5: final_yerr_max[2, 0],
                scenario6: final_yerr_max[2, 1]
            }
        df_yerr_min = pd.DataFrame({'yerr_min': f_yerr_min})
        df_yerr_max = pd.DataFrame({'yerr_max': f_yerr_max})

        plot_yerr_df = pd.concat([df_yerr_min, df_yerr_max], axis = 1).reindex(index_order)

        # Define the x positions for the error bars (assuming the same order as the scenarios)
        x_positions = np.arange(len(index_order))
        # Extract error bar values from the combined DataFrame
        error_bars = plot_yerr_df.to_numpy()
        # Plot the error bars
        if yaxis7 is not None and 'Avoided' in yaxis7:
            plot_df_without_np7 = plot_df.drop(columns=[yaxis7])
            ax.errorbar(x_positions, plot_df_without_np7.sum(axis=1).values, yerr=error_bars.T, fmt=fmt_choice, color='black',
                        capsize=5, capthick=2, ecolor='black', label=error_label)
        else:
            ax.errorbar(x_positions, plot_df.sum(axis=1).values, yerr=error_bars.T, fmt=fmt_choice, color  = 'black', capsize=5, capthick=2, ecolor='black', label = error_label)

    # Remove the gray background
    ax.set_facecolor('white')

    # Set the title and labels
    if what == 'DAC_Gt' and cumulative is False:
        ylabel = "DAC rates (GtCO$_2$/year "+str(year)+")"
        leg_loc = 'upper left'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'DAC_Gt' and cumulative is True:
        ylabel = "Cumulative DAC rates (GtCO$_2$)"
        leg_loc = 'upper left'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Total cost" and cumulative is False:
        ylabel = "Cost (billion €/year in "+str(year)+")"
        leg_loc = 'upper center'
        if scenario == 'Contrails options':
            loc_text = [0.075, 0.55]
        else:
            loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Total cost" and cumulative is True:
        ylabel = "Cost (trillion €)"
        leg_loc = 'upper center'
        if scenario == 'Contrails options':
            loc_text = [0.075, 0.55]
        else:
            loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost" and cumulative is False:
        ylabel = "Cost (€/year in "+str(year)+")"
        leg_loc = 'upper center'
        if scenario == 'Contrails options':
            loc_text = [0.075, 0.55]
        else:
            loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Cost' and cumulative is True:
        ylabel = "Cumulative cost (€)"
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost per emissions" and cumulative is False:
        ylabel = "Cost per emissions (€/tCO$_2$e*/year in "+str(year)+")"
        leg_loc = 'upper center'
        loc_text = [0.1, 0.75]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Cost per emissions' and cumulative is True:
        ylabel = "Cost per emissions (€/tCO$_2$e*)"
        leg_loc = 'upper center'
        loc_text = [0.1, 0.75]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost per DAC" and cumulative is False:
        ylabel = "Cost per installed DAC (€/tCO$_2$/year in "+str(year)+")"
        leg_loc = 'upper center'
        loc_text = [0.25, 0.7]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Cost per DAC' and cumulative is True:
        ylabel = "Cost per installed DAC (€/tCO$_2$)"
        leg_loc = 'upper center'
        loc_text = [0.25, 0.7]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Cost per ton":
        ylabel = "Cost per ton (€/t) in "+ str(year)
        leg_loc = 'upper center'
        loc_text = [0.1, 0.75]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == "Emissions" and cumulative is False:
        ylabel = "Emissions (GtCO$_2$e*/year in "+str(year)+")"
        leg_loc = 'upper center'
        if scenario == 'Contrails options':
            loc_text = [0.075, 0.55]
        elif scenario == 'Scenarios explanation':
            loc_text = [0.025, 0.35, 0.7]
        else:
            loc_text = [0.1, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    elif what == 'Emissions' and cumulative is True:
        ylabel = "Emissions (GtCO$_2$e*)"
        leg_loc = 'upper left'
    elif what == "Electricity" or what == "Electricity_withDiesel" and cumulative is False:
        ylabel = "Electricity (MWh/year in "+str(year)+")"
        leg_loc = 'upper left'
    elif what == 'Electricity' or what == "Electricity_withDiesel"  and cumulative is True:
        ylabel = "Electricity (MWh)"
        leg_loc = 'upper left'
    elif what == "Change flight price" and cumulative is False:
        ylabel = "\u0394 flight price relative to BAU (%) in "+str(year)
        leg_loc = 'upper center'
        loc_text = [0.1, 0.65]
        num_columns = 3
        leg_box = (0.5, -0.075)
    elif what == "Cost neutrality per flight" and cumulative is False:
        ylabel = "Cost neutrality per flight (€/passenger) in "+str(year)
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 3
        leg_box = (0.5, -0.075)
    elif what == "Cost per liter fuel" and cumulative is False:
        ylabel = "Cost per liter fuel (€/L) in "+str(year)
        leg_loc = 'upper center'
        if scenario == 'Contrails options':
            loc_text = [0.075, 0.55]
        else:
            loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)
    else:
        ylabel = what
        leg_loc = 'upper center'
        loc_text = [0.2, 0.65]
        num_columns = 2
        leg_box = (0.5, -0.075)

    if scenario == 'decreasing demand':
        ax.text(0.45, 1.05, "-2% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'stagnating demand':
        ax.text(0.45, 1.05, "+0% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'historically observed demand':
        ax.text(0.45, 1.05, "+4% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'base demand growth':
        ax.text(0.45, 1.05, "+2% demand", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'allnonCO2':
        ax.text(0.35, 1.05, "Offsetting contrails", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'noCC':
        ax.text(0.35, 1.05, "Not offsetting contrails", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'rerouting':
        ax.text(0.35, 1.05, "With rerouting", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
    elif scenario == 'Contrails options':
        if what == 'Emissions':
            ax.text(0.3, 0.075, "Offsetting \ncontrails", transform=ax.transAxes, fontsize=10, color='black',
                    fontweight='bold',
                    multialignment='center')
            ax.text(0.55, 0.05, "Rerouting", transform=ax.transAxes, fontsize=10, color='black', fontweight='bold', multialignment = 'center')
            ax.text(0.8, 0.075, "Ignoring \ncontrails", transform=ax.transAxes, fontsize=10, color='black', fontweight='bold',
                    multialignment='center')
        else:
            ax.text(0.3, 0.75, "Offsetting \ncontrails", transform=ax.transAxes, fontsize=10, color='black', fontweight='bold',
                    multialignment='center')
            ax.text(0.55, 0.75, "Rerouting", transform=ax.transAxes, fontsize=10, color='black', fontweight='bold',
                    multialignment='center')
            ax.text(0.8, 0.75, "Ignoring \ncontrails", transform=ax.transAxes, fontsize=10, color='black',
                    fontweight='bold',
                    multialignment='center')
    elif scenario == 'Climate neutrality only':
        if what == 'Emissions':
            ax.text(0.05, 0.9, "Default", transform=ax.transAxes, fontsize=10, color='black',
                    fontweight='bold',
                    multialignment='center')
            ax.text(0.3, 0.9, "Demand reduction", transform=ax.transAxes, fontsize=10, color='black', fontweight='bold',
                    multialignment='center')
            ax.text(0.7, 0.9, "Contrails avoidance", transform=ax.transAxes, fontsize=10, color='black',
                    fontweight='bold',
                    multialignment='center')
            ax.text(0.25, 0.075, "Capped\ndemand", transform=ax.transAxes, fontsize=10, color='black',
                    #fontweight='bold',
                    multialignment='center')
            ax.text(0.42, 0.075, "Decreasing\ndemand", transform=ax.transAxes, fontsize=10, color='black',
                    # fontweight='bold',
                    multialignment='center')
            ax.text(0.65, 0.075, "Rerouting", transform=ax.transAxes, fontsize=10, color='black', # fontweight='bold',
                    multialignment = 'center')
            ax.text(0.85, 0.075, "Ignoring \ncontrails", transform=ax.transAxes, fontsize=10, color='black', #fontweight='bold',
                    multialignment='center')
        else:
            ax.text(0.05, 0.95, "Default", transform=ax.transAxes, fontsize=10, color='black',
                    fontweight='bold',
                    multialignment='center')
            ax.text(0.3, 0.95, "Demand reduction", transform=ax.transAxes, fontsize=10, color='black', fontweight='bold',
                    multialignment='center')
            ax.text(0.7, 0.95, "Contrails avoidance", transform=ax.transAxes, fontsize=10, color='black',
                    fontweight='bold',
                    multialignment='center')
            ax.text(0.25, 0.85, "Capped\ndemand", transform=ax.transAxes, fontsize=10, color='black',
                    #fontweight='bold',
                    multialignment='center')
            ax.text(0.42, 0.85, "Decreasing\ndemand", transform=ax.transAxes, fontsize=10, color='black',
                    # fontweight='bold',
                    multialignment='center')
            ax.text(0.65, 0.85, "Rerouting", transform=ax.transAxes, fontsize=10, color='black', # fontweight='bold',
                    multialignment = 'center')
            ax.text(0.85, 0.85, "Ignoring \ncontrails", transform=ax.transAxes, fontsize=10, color='black', #fontweight='bold',
                    multialignment='center')


    plt.ylabel(ylabel)

    new_labels = {
        'DACCS': 'Emit-and-\nremove',
        'DACCU': 'DAC-based\nfuels',
        'BAU': 'Business-as\n-Usual'
    }

    # Rotate x-axis labels horizontally
    ax.tick_params(axis='x', rotation=0)
    #tick_labels = [new_labels[label.split()[0]] if label.split()[0] in new_labels else 'Business-as\n-Usual' for label
    #               in index_order]
    tick_labels = [label.split()[0] if label.split()[0] in ["DACCS", "DACCU"] else "BAU" for label in index_order]
    ax.set_xticklabels(tick_labels, fontweight='bold', fontsize=9)

    # Iterate through tick labels and set custom colors
    for i, label in enumerate(tick_labels):
        if label == 'DACCU':
            ax.get_xticklabels()[i].set_color('#2f609c')  # Custom color for DACCS
        elif label == 'DACCS':
            ax.get_xticklabels()[i].set_color('#e26e02')  # Custom color for DACCU
        elif label == 'BAU':
            ax.get_xticklabels()[i].set_color('#ff5733')  # Custom color for DACCU

    if hlines_divide is True:
        if scenario == 'Contrails options':
            ax.axvline(1.5, ymin=0.05, ymax=0.95, color='grey')
            if what == 'Emissions':
                ax.axvline(3.5, ymin=0.1, ymax=0.9, color='grey')
                ax.axvline(5.5, ymin=0.1, ymax=0.9, color='grey')
            else:
                ax.axvline(3.5, ymin=0.05, ymax=0.8, color='grey')
                ax.axvline(5.5, ymin=0.05, ymax=0.8, color='grey')
        elif scenario == 'Climate neutrality only':
            ax.axvline(1.5, ymin=0.05, ymax=0.95, color='grey')
            if what == 'Emissions':
                ax.axvline(3.5, ymin=0.1, ymax=0.85, color='grey')
                ax.axvline(5.5, ymin=0.05, ymax=0.95, color='grey')
                ax.axvline(7.5, ymin=0.1, ymax=0.85, color='grey')
            else:
                ax.axvline(3.5, ymin=0.05, ymax=0.75, color='grey')
                ax.axvline(5.5, ymin=0.05, ymax=0.95, color='grey')
                ax.axvline(7.5, ymin=0.05, ymax=0.75, color='grey')
        # Add text to the upper left and upper right
        elif scenario == 'Scenarios explanation':
            ax.text(loc_text[0], 0.9, "Business-\nas-Usual", transform=ax.transAxes, fontsize=12, color='black',
                    fontweight='bold', multialignment='center')
            ax.text(loc_text[1], 0.9, "CO$_2$\nneutrality", transform=ax.transAxes, fontsize=12, color='black',
                    fontweight='bold', multialignment='center')
            ax.text(loc_text[2], 0.9, "Climate\nneutrality", transform=ax.transAxes, fontsize=12, color='black',
                    fontweight='bold', multialignment='center')
            ax.axvline(0.5, ymin=0.05, ymax=0.95, color='grey')
            ax.axvline(2.5, ymin=0.05, ymax=0.95, color='grey')


        elif scenario != 'Climate neutrality only':
            ax.text(loc_text[0], 0.85, "CO$_2$\nneutrality", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')
            ax.text(loc_text[1], 0.85, "Climate\nneutrality", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold', multialignment = 'center')


        # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc=leg_loc, ncol=num_columns, bbox_to_anchor=leg_box, frameon=False)
    if cumulative is False:
        ax.figure.savefig('Figures/Figures_final/'+what.replace(' ','_') + '_' + scenario.replace(' ','_') + "_" + str(year) +'.png', bbox_inches='tight', dpi=850, transparent = True)
    else:
        ax.figure.savefig('Figures/Figures_final/'+what.replace(' ','_') + '_' + scenario.replace(' ','_') + "_" + 'cumulative' +'.png', bbox_inches='tight', dpi=850, transparent = True)

def tornado_plot(df, positive_change, negative_change, standard_values,
                 neutrality = 'Carbon neutrality', comparison = 'DACCS', palette=None):
    """
    Creates a tornado plot from a sensitivity analysis DataFrame with standard parameter values.

    Parameters:
    df (pd.DataFrame): DataFrame containing the sensitivity analysis data.
    positive_change (str): The positive percentage change to plot (must match one of the index values).
    negative_change (str): The negative percentage change to plot (must match one of the index values).
    standard_values (dict): Dictionary containing the standard values of the parameters.
    """

    scenario_name = 'DACCU cost penalty in 2050 [trillion €]'
    unit = "trillion €"

    if neutrality == "Climate neutrality":
        neutrality_save = "climate_neutral"
    elif neutrality == "Carbon neutrality":
        neutrality_save = "carbon_neutral"
    elif neutrality == "Baseline":
        neutrality_save = "baseline"
    else:
        neutrality_save = ""

    comparison_save = comparison.replace(' ', '_')

    plt.figure()
    sns.set(rc={'figure.figsize': (8, 8)})
    sns.set(font_scale=1.4)

    # Check if the positive_change and negative_change exist in the DataFrame index
    if positive_change not in df.index:
        raise ValueError(
            f"The positive_change '{positive_change}' is not in the DataFrame index. Please choose from {df.index.tolist()}")
    if negative_change not in df.index:
        raise ValueError(
            f"The negative_change '{negative_change}' is not in the DataFrame index. Please choose from {df.index.tolist()}")

    # Extract the rows for the specified percentage changes
    data_positive = df.loc[positive_change]
    data_negative = df.loc[negative_change]

    # Combine the data for the tornado plot
    data_combined = pd.DataFrame({positive_change: data_positive, negative_change: data_negative})

    # Sort the data by the minimum impact value
    data_combined_sorted = data_combined.min(axis=1).sort_values().index

    # Prepare the data for the bar plot
    plot_data = pd.melt(data_combined.loc[data_combined_sorted].reset_index(), id_vars='index',
                        value_vars=[positive_change, negative_change],
                        var_name='Change', value_name='Impact')
    plot_data.rename(columns={'index': 'Parameter'}, inplace=True)
    plot_data['Change'] = plot_data['Change'].map({positive_change: 'Positive', negative_change: 'Negative'})

    # Calculate bar positions with a slight offset for negative bars
    bar_width = 0.4
    y_pos = np.arange(len(data_combined_sorted))
    y_pos_positive = y_pos + bar_width / 2
    y_pos_negative = y_pos - bar_width / 2

    # Create a color palette
    custom_cmp = sns.diverging_palette(220, 20, as_cmap=False)
    bar_colors = [custom_cmp[0], custom_cmp[-1]]

    # Create labels with standard values
    labels_with_values = [f"{param} (std: {standard_values[param]:.2f})" for param in data_combined_sorted]

    # Create a tornado plot
    #fig, ax = plt.subplots(figsize=(14, 11))
    #ax.barh(y_pos_positive, data_combined.loc[data_combined_sorted, positive_change], color=positive_color,
    #        height=bar_width, label=positive_change)
    #ax.barh(y_pos_negative, data_combined.loc[data_combined_sorted, negative_change], color=negative_color,
    #        height=bar_width, label=negative_change)

    bar_plot = sns.barplot(data=plot_data, x='Impact', y='Parameter', hue='Change', palette=bar_colors)

    # Add labels and title
    bar_plot.set_yticks(y_pos)
    bar_plot.set_yticklabels(labels_with_values)
    bar_plot.set_xlabel('Impact on cost penalty of DACCU')
    bar_plot.set_ylabel('Parameters')
    #ax.set_title(f'Tornado Plot for {positive_change} and {negative_change} Changes')

    # Set white background and frame around the plot area
    bar_plot.set_facecolor('white')
    for spine in bar_plot.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Remove grid lines
    bar_plot.grid(False)

    # Add legend without frame
    bar_plot.legend(frameon=False)

    bar_plot.figure.savefig(
        'Figures/Figures_final/' + 'Tornado_factors_DACCU_vs_' + comparison_save + "_" + neutrality_save + "_" + unit +
        '.png', bbox_inches='tight')

    return plot_data





def scatter_plot(df_optimization, neutrality = 'Carbon neutrality', comparison = 'DACCS', palette=None):
    """
    Creates a scatter plot for the sensitivity analysis optimization results.
    Parameters:
    df_optimization (pd.DataFrame): DataFrame containing the optimization results.
    palette (str): Color palette to use for the scatter plot (default is 'viridis').
    """

    scenario_name = 'DACCU cost penalty in 2050 [trillion €]'
    unit = "trillion €"

    if neutrality == "Climate neutrality":
        neutrality_save = "climate_neutral"
    elif neutrality == "Carbon neutrality":
        neutrality_save = "carbon_neutral"
    elif neutrality == "Baseline":
        neutrality_save = "baseline"
    else:
        neutrality_save = ""

    comparison_save = comparison.replace(' ', '_')

    if palette is None:
        palette = sns.diverging_palette(220, 20, as_cmap=False)
    custom_cmp = sns.diverging_palette(220, 20, as_cmap=True)

    norm = plt.Normalize(df_optimization['Minimized Difference (€ Trillion)'].min(),
                         df_optimization['Minimized Difference (€ Trillion)'].max())

    # Sort the DataFrame by 'Minimized Difference (€ Trillion)' to get the minimal values on top
    df_sorted = df_optimization.sort_values(by='Minimized Difference (€ Trillion)')

    # Create the scatter plot
    plt.figure()
    sns.set(rc={'figure.figsize': (5, 8)})
    sns.set(font_scale=1.4)

    # Create a color palette
    n_colors = len(df_optimization["Variable"])
    colors = sns.color_palette(palette, n_colors)
    scatter_optimum = sns.scatterplot(data = df_sorted,
                                    x='Percentage Change of Optimal Value', y= 'Variable',
                                        hue='Minimized Difference (€ Trillion)', #errorbar = None, marker='o',
                                    palette = custom_cmp, s=100, legend=None, norm=norm)
                                    #palette = palette, norm=norm, dodge = True, legend = None)
    scatter_optimum.set_title(neutrality.replace('Carbon', 'CO$_2$') + " - DACCU vs. " + comparison + "\n", fontsize=18)
    scatter_optimum.set_xlabel("\n Variation from assumed value (\%) \n", fontsize=16)

    #TODO: add hline for 0% and order the parameters based on 'ordered' tornado values
    #TODO: add standard values too in bold and remove them from tornado figure
    #Remove legend
    #scatter_optimum.legend_.remove()

    # Create color bar
    #norm = plt.Normalize(df_optimization['Minimized Difference (€ Trillion)'].min(), df_optimization['Minimized Difference (€ Trillion)'].max())
    sm = plt.cm.ScalarMappable(cmap=custom_cmp, norm=norm)
    sm.set_array([])

    cbar = scatter_optimum.figure.colorbar(sm)
    cbar.set_label('Minimized Difference (€ Trillion)', rotation=270, labelpad=20)

    # Remove connectors
    #for line in scatter_optimum.get_lines():
    #    line.set_linestyle('')

    # Set white background and frame around the plot area
    scatter_optimum.set_facecolor('white')
    # Set a frame around the plot area
    for spine in scatter_optimum.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    scatter_optimum.set_xticks(np.arange(-400,401,200))
    scatter_optimum.set_xticklabels([f"{x}%" for x in np.arange(-400,401,200)])

    scatter_optimum.figure.savefig(
        'Figures/Figures_final/' + 'Optimal_factors_DACCU_vs_' + comparison_save + "_" + neutrality_save + "_" + unit +
        '.png', bbox_inches='tight')



def combined_plot(df, df_optimization, positive_change, negative_change, standard_values,
                  original_diff = None, hue_scatter = 'Minimized Difference (€ Trillion)',
                  neutrality='Carbon neutrality', comparison='DACCS', reference = 'DACCU',
                  palette=None,
                  percentage_penalty_change = False):
    """
    Creates a combined tornado and scatter plot from a sensitivity analysis DataFrame and optimization results.

    Parameters:
    df (pd.DataFrame): DataFrame containing the sensitivity analysis data.
    df_optimization (pd.DataFrame): DataFrame containing the optimization results.
    positive_change (str): The positive percentage change to plot (must match one of the index values).
    negative_change (str): The negative percentage change to plot (must match one of the index values).
    standard_values (dict): Dictionary containing the standard values of the parameters.
    """

    if percentage_penalty_change is True:
        unit = "%"
        scenario_name =' Change in '+ reference + ' cost penalty in 2050 (' + unit + ')'
        scenario_name_bar = scenario_name

    else:
        unit = "trillion €"
        scenario_name = reference + ' cost penalty in 2050 (' + unit + ')'
        scenario_name_bar = hue_scatter

    if neutrality == "Climate neutrality":
        neutrality_save = "climate_neutral"
    elif neutrality == "Carbon neutrality":
        neutrality_save = "carbon_neutral"
    elif neutrality == "Baseline":
        neutrality_save = "baseline"
    else:
        neutrality_save = ""

    comparison_save = comparison.replace(' ', '_')

    custom_cmp = sns.diverging_palette(220, 20, as_cmap=True)
    # Create a color palette
    custom_cmp_discrete = sns.diverging_palette(220, 20, as_cmap=False)
    bar_colors = [custom_cmp_discrete[0], custom_cmp_discrete[-1]]

    # Check if the positive_change and negative_change exist in the DataFrame index
    if positive_change not in df.index:
        raise ValueError(
            f"The positive_change '{positive_change}' is not in the DataFrame index. Please choose from {df.index.tolist()}")
    if negative_change not in df.index:
        raise ValueError(
            f"The negative_change '{negative_change}' is not in the DataFrame index. Please choose from {df.index.tolist()}")

    # Extract the rows for the specified percentage changes
    data_positive = df.loc[positive_change]
    data_negative = df.loc[negative_change]

    if original_diff is None:
        original_diff = df.loc['0%'].values
        if reference == 'DACCS':
            original_diff = -original_diff

    if percentage_penalty_change is True:
        data_positive = (data_positive/df.loc['0%'] - 1) * 100
        data_negative = (data_negative / df.loc['0%'] - 1) * 100

    # Combine the data for the tornado plot
    data_combined = pd.DataFrame({positive_change: data_positive, negative_change: data_negative})


    # Sort the data by the minimum impact value
    data_combined_sorted = data_combined.min(axis=1).sort_values().index

    # Prepare the data for the bar plot
    plot_data = pd.melt(data_combined.loc[data_combined_sorted].reset_index(), id_vars='index',
                        value_vars=[positive_change, negative_change],
                        var_name='Change', value_name='Impact')
    plot_data.rename(columns={'index': 'Parameter'}, inplace=True)
    plot_data['Change'] = plot_data['Change'].map({positive_change: positive_change, negative_change: negative_change})

    # Ensure the order and names match the new columns for optimization data
    name_mapping = {
        'electricity cost': 'levelized cost\nof electricity',
        'learning rate': 'all learning\nrates',
        'learning rate H2': 'learning\nrate H$_2$',
        'learning rate DAC': 'learning\nrate DAC',
        'growth rate': 'demand\ngrowth',
        'fossil fuel cost': 'price of\nfossil kerosene',
        'DAC initial cost': 'DAC initial\nCAPEX',
        'fuel efficiency': 'fuel\nefficiency',
        'cc efficacy': 'contrails\nefficacy'
    }

    df_sorted = df_optimization.copy()
    # Map the names and ensure there are no duplicates
    df_sorted['Variable'] = df_sorted['Variable'].map(name_mapping)
    df_sorted = df_sorted.drop_duplicates(subset='Variable')
    # Ensure the Variable column has unique values and matches the order
    df_sorted = df_sorted.set_index('Variable').reindex(data_combined_sorted).reset_index()
    df_sorted.rename(columns={'index': 'Variable'}, inplace=True)

    # Same thing with standard values
    standard_values_sorted = standard_values.copy()
    # Map the names and ensure there are no duplicates
    standard_values_sorted_mapped = remap_standard_values(standard_values_sorted, name_mapping)
    # Ensure the Variable column has unique values and matches the order
    #standard_values_sorted = standard_values_sorted.set_index('Variable').reindex(data_combined_sorted).reset_index()

    # Add the percentage change in minimized difference relative to the default
    df_sorted[hue_scatter] = [
        (row["Minimized Difference (€ Trillion)"] / original_diff[idx] - 1) * 100
        if row["Minimized Difference (€ Trillion)"] != 'NA' else 'NA'
        for idx, row in df_sorted.iterrows()
    ]
    # Add percentage change of the optimal value relative to the standard value
    df_sorted[
        "Percentage Change of Optimal Value"] = df_sorted.apply(
        lambda row: (row["Optimal Value"] / standard_values_sorted_mapped[row["Variable"]] - 1) * 100 if row[
                                                                                               "Optimal Value"] != 'NA' else 'NA',
        axis=1
    )


    norm = plt.Normalize(df_sorted[hue_scatter].min(),
                         df_sorted[hue_scatter].max())

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharey=True)

    # Tornado plot
    axes[0].set_title('Impact of \u00B1'+positive_change.replace('+','')+ ' variation in parameters ')
    axes[0].set_facecolor('white')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    axes[0].grid(False)
    if percentage_penalty_change is True:
        if comparison == 'BAU':
            min_lim = plot_data['Impact'].min()*2
        else:
            min_lim = plot_data['Impact'].min() * 1.5
        max_lim = plot_data['Impact'].max()*1.1
        axes[0].axvline(0, color='grey', linestyle='--', label='Default')
        axes[0].axvspan(min_lim, -100, color='grey', alpha=0.5)
        axes[0].text(min_lim*0.95, len(data_combined_sorted)/1.05, "No " + reference + "\ncost penalty",
                     verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=12, weight='bold')
        axes[0].set_xlim(min_lim, max_lim)
        sns.barplot(ax=axes[0], data=plot_data, x='Impact', y='Parameter', hue='Change', dodge=False, palette=bar_colors)
    else:
        axes[0].axvline(original_diff[0], color='grey', linestyle = '--', label = 'Default')
        sns.barplot(ax=axes[0], data=plot_data, x='Impact', y='Parameter', hue='Change', palette=bar_colors)
    axes[0].legend(frameon=False, loc = 'lower right')
    axes[0].set_xlabel(scenario_name)
    axes[0].set_ylabel('Parameters')

    # Scatter plot
    scatter_optimum = sns.scatterplot(ax=axes[1], data=df_sorted, x='Percentage Change of Optimal Value', y='Variable',
                                      hue=hue_scatter, palette=custom_cmp, s=150, legend=None, norm=norm)
    scatter_optimum.set_xlabel("Variation from assumed value (%)")
    scatter_optimum.set_xticks(np.arange(-400, 401, 200))
    scatter_optimum.set_xticklabels([f"{x}%" for x in np.arange(-400, 401, 200)])
    scatter_optimum.set_facecolor('white')
    scatter_optimum.set_title('Variation that minimizes the gap\nbetween '+ reference+ ' and '+comparison)
    # Add vertical dashed line at 0% in the scatter plot
    scatter_optimum.axvline(0, color='grey', linestyle='--')
    for spine in scatter_optimum.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    sm = plt.cm.ScalarMappable(cmap=custom_cmp, norm=norm)
    sm.set_array([])
    cbar = scatter_optimum.figure.colorbar(sm, ax=axes[1])
    cbar.set_label(scenario_name_bar, rotation=270, labelpad=20)

    # Add labels with standard values to scatter plot
    #for i, param in enumerate(data_combined_sorted):
    #    scatter_optimum.text(df_sorted.loc[i, 'Percentage Change of Optimal Value'], i+0.5,
    #                         f"{standard_values_sorted_mapped[param]:.2f}",
    #                         verticalalignment='bottom', horizontalalignment='center', weight='bold')

    # Add labels with optimal values to scatter plot
    for i, param in enumerate(data_combined_sorted):
        scatter_optimum.text(df_sorted.loc[i, 'Percentage Change of Optimal Value'], i + 0.5,
                             f"{df_sorted.loc[i, 'Optimal Value']:.2f}",
                             verticalalignment='bottom', horizontalalignment='center', weight='bold')

    # Create a custom legend entry
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='% change',
                   markerfacecolor=bar_colors[0], markersize=10, markeredgecolor='b')
    ]
    # Add the colorbar legend
    scatter_optimum.legend(handles=legend_elements, frameon=False, loc='lower left')

    fig.suptitle(neutrality.replace('Carbon', 'CO$_2$') + " - " + reference +" vs. " + comparison)
    # Adjust layout
    plt.tight_layout()
    fig.savefig(
        'Figures/Figures_final/' + 'Combined_factors_DACCU_vs_' + comparison_save + "_" + neutrality_save + "_" + unit +
        '.png', bbox_inches='tight')

    plt.show()






