import matplotlib.pyplot as plt
from RLS import RLS
num_params = 4  # 参数个数
forgetting_factor = 0.99  # 遗忘因子


def get_charging_OCV(training_df, charging_df, show_plot=True):
    rls = RLS(num_params, forgetting_factor)
    OCV_list_full = []
    OCV_list_charge = []
    theta_0_list = []
    theta_1_list = []
    SOC_list = []
    real_voltage_list = []

    for index, row in training_df.iterrows():
        x = [1, row['Voltage'], row['Current'], row['Current -1']]
        y = row['Voltage']
        theta = rls.update(x, y)
        OCV = theta[0]/(1-theta[1])
        OCV_list_full.append(OCV)
        theta_0_list.append(theta[0])
        theta_1_list.append(theta[1])
    for i, r in charging_df.iterrows():
        x = [1, r['Total Voltage'], r['Total Current'], r['Total Current -1']]
        y = r['Total Voltage']
        real_voltage_list.append(y/96)
        SOC_list.append(r['SOC(%)'])
        theta = rls.update(x, y)
        OCV = theta[0]/(1-theta[1])
        OCV_list_full.append(OCV)
        OCV_list_charge.append(OCV/96)
        theta_0_list.append(theta[0])
        theta_1_list.append(theta[1])
    if show_plot:
        plt.plot(SOC_list, OCV_list_charge, label='OCV by RLS')
        plt.plot(SOC_list, real_voltage_list, label='Real Charging Voltage')
        plt.title(f'OCV after {len(training_df)} RLS Iteration')
        plt.grid()
        plt.legend()
        plt.text(10, 3.80, f'start OCV: {OCV_list_charge[0]}')
        plt.text(10, 3.78, f'end OCV: {OCV_list_charge[-1]}')
        plt.show()
    return OCV_list_charge


def calculate_charging_dod(OCV_list):
    start_soc = transfer_OCV_SOC(OCV_list[0])
    end_soc = transfer_OCV_SOC(OCV_list[-1])
    return (end_soc - start_soc)/100


def transfer_OCV_SOC(OCV_value):
    return OCV_value