import boto3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

s3 = boto3.resource('s3')
bucket_name = 'apacdlcdc-rms-ted-sample/lifecycle'


columns_name_a = [
    "VIN",
    "Sending Time",
    "Receive Time",
    "Reissue Report",
    "Vehicle State",
    "Charging State",
    "Operation Mode",
    "Vehicle Speed(km/h)",
    "Accumulated Mileage",
    "Total Voltage",
    "Total Current",
    "SOC(%)",
    "DC-DC State",
    "With Driving Force",
    "With Braking Force",
    "Gears Position",
    "Insulation Resistance",
    "Accelerated Pedal Stroke Value",
    "Brake Pedal Status",
    "Engine State",
    "Crankshaft Speed(rpm)",
    "Fuel Consumption(L/100km)",
    "No. of Battery Subsystem With Max. Voltage",
    "No. of Cell With Maximum Voltage",
    "Max. Cell Voltage(V)",
    "No. of Battery Subsystem With Min. Voltage",
    "No. of Cell With Minimum Voltage",
    "Min. Cell Voltage(V)",
    "No. of Subsystem With Max. Temperature",
    "No. of Probe With Maximum Temperature",
    "Max. Temperature Value",
    "No. of Subsystem With Min Temperature",
    "No of Probe With Minimum Temperature",
    "Min. Temperature Value",
    "Highest Warning Level",
    "Total Quantities of RESS’s Faults",
    "Total Number of Motor Failures",
    "Total Engine Failure",
    "Total Number of Other Faults",
    "List of Fault Codes For RESS",
    "Drive Motor Fault Code List",
    "Engine Fault List",
    "List of Other Fault Codes",
    "Tem Difference Warning",
    "Battery High Temp Warning",
    "Energy Storage Over Voltage Warning",
    "Energy Storage Under Voltage Warning",
    "Low SOC Warning",
    "Cell Over Voltage Warning",
    "Cell Under Voltage Warning",
    "Excessively High SOC Warning",
    "SOC Jump Warning",
    "Energy Storage Unmatched",
    "Cell Poor Consistency Warning",
    "Insulation Warning",
    "DC-DC Temperature Warning",
    "Brake System Warning",
    "DC-DC State Warning",
    "Electrical Temperature Warning",
    "Highvoltage Interlocking Warning",
    "Electrical Temp Warning",
    "Energy Storage Over Charging",
    "Voltage Vol Rechargeable System Number",
    "Frame Start Cell Number",
    "Voltage of Ress",
    "Total number of single cells in this frame",
    "Current of Ress",
    "Driving Motor Data",
    "Total Number of Single Cells",
    "Temperature Rechargeable System Number",
    "Total Temperature Probe",
    "Battery Pack Voltage Temperature"
]


columns_name_b = [
    "VIN",
    "Sending Time",
    "Receive Time",
    "Reissue Report",
    "Vehicle State",
    "Charging State",
    "Operation Mode",
    "Vehicle Speed(km/h)",
    "Accumulated Mileage",
    "Total Voltage",
    "Total Current",
    "SOC(%)",
    "DC-DC State",
    "With Driving Force",
    "With Braking Force",
    "Gears Position",
    "Insulation Resistance",
    "Accelerated Pedal Stroke Value",
    "Brake Pedal Status",
    "Engine State",
    "Crankshaft Speed(rpm)",
    "Fuel Consumption(L/100km)",
    "No. of Battery Subsystem With Max. Voltage",
    "No. of Cell With Maximum Voltage",
    "Max. Cell Voltage(V)",
    "No. of Battery Subsystem With Min. Voltage",
    "No. of Cell With Minimum Voltage",
    "Min. Cell Voltage(V)",
    "No. of Subsystem With Max. Temperature",
    "No. of Probe With Maximum Temperature",
    "Max. Temperature Value",
    "No. of Subsystem With Min Temperature",
    "No of Probe With Minimum Temperature",
    "Min. Temperature Value",
    "Highest Warning Level",
    "Total Quantities of RESS’s Faults",
    "Total Number of Motor Failures",
    "Total Engine Failure",
    "Total Number of Other Faults",
    "List of Fault Codes For RESS",
    "Drive Motor Fault Code List",
    "Engine Fault List",
    "List of Other Fault Codes",
    "Tem Difference Warning",
    "Battery High Temp Warning",
    "Energy Storage Over Voltage Warning",
    "Energy Storage Under Voltage Warning",
    "Low SOC Warning",
    "Cell Over Voltage Warning",
    "Cell Under Voltage Warning",
    "Excessively High SOC Warning",
    "SOC Jump Warning",
    "Energy Storage Unmatched",
    "Cell Poor Consistency Warning",
    "Insulation Warning",
    "DC-DC Temperature Warning",
    "Brake System Warning",
    "DC-DC State Warning",
    "Electrical Temperature Warning",
    "Highvoltage Interlocking Warning",
    "Electrical Temp Warning",
    "Energy Storage Over Charging",
    "Voltage Vol Rechargeable System Number",
    "Frame Start Cell Number",
    "Voltage of Ress",
    "Total number of single cells in this frame",
    "Current of Ress",
    "Driving Motor Data"
]


def process_s3_csv(file_path):
    # read file from merged csv in S3 and use | to separate data in first column
    df = pd.read_csv(f's3://{bucket_name}/{file_path}')
    subset_data = df.iloc[:, :1]
    data = pd.DataFrame({
        'column_name': subset_data.iloc[:, 0]
    })
    split_data = data['column_name'].str.split('|', expand=True)
    if split_data.shape[1] == 72:
        split_data.columns = columns_name_a
    else:
        split_data.columns = columns_name_b
    # convert data and date into correct format for plotting
    split_data['Receive Time'] = pd.to_datetime(split_data['Receive Time'])
    split_data['Sending Time'] = pd.to_datetime(split_data['Sending Time'])
    split_data['Total Current'] = pd.to_numeric(split_data['Total Current'], errors='coerce')
    split_data['Total Voltage'] = pd.to_numeric(split_data['Total Voltage'], errors='coerce')
    split_data['Accumulated Mileage'] = pd.to_numeric(split_data['Accumulated Mileage'])
    split_data['SOC(%)'] = pd.to_numeric(split_data['SOC(%)'], errors='coerce')
    return split_data


def charging_voltage_plot(dataframe):
    # select onlt parking charge status for plotting
    parking_charge_data = dataframe[dataframe['Charging State'] == 'Parking charge'].copy()
    fig = px.scatter(x=parking_charge_data['Sending Time'], y=parking_charge_data['Total Voltage'], title = 'Charging Voltage(v) Scatter Plot', labels={'x':'Charging Date', 'y':'Charging Voltage(v)'})
    fig.update_traces(marker_size=2)
    fig.show()
    return parking_charge_data


def filter_charging_event_2(df, voltage_start=345, voltage_end=375, stop_threshold=360):
    df_range = df[df['Total Voltage'].between(voltage_start, voltage_end, inclusive='both')].copy()
    # filter continuous charging event if soc is continuous increasing by 1%
    # store index for each charing event
    continuous_segments = []
    # store index for current charging event
    current_segment = []
    # indexing filtered charing event
    df_range['group_index'] = None
    group_index = 0

    for index, row in df_range.iterrows():
        soc = row['SOC(%)']
        voltage = row['Total Voltage']
        if not current_segment:
            if voltage == voltage_start:
                current_segment.append(index)
            else:
                continue
        elif soc == df_range.loc[current_segment[-1], 'SOC(%)'] or soc == (df_range.loc[current_segment[-1],'SOC(%)'] + 1):
            time_diff = (row['Sending Time'] - df_range.loc[current_segment[-1], 'Sending Time']).total_seconds()
            # exclude charging event with stop > stop_threshold
            if time_diff > stop_threshold:
                current_segment = []
                continue
            current_segment.append(index)
            if voltage == voltage_end:
                df_range.loc[current_segment, 'group_index'] = group_index
                continuous_segments.append(current_segment)
                current_segment = []
                group_index += 1
        else:
            current_segment = []
    df_range_concat = pd.concat([df_range.loc[seg] for seg in continuous_segments])
    return df_range_concat


def calculate_area(group_df):
    time_diff = ((group_df['Sending Time'].diff().dt.seconds)/3600).fillna(0)
    current_values = group_df['Total Current']
    Integral_area = ((current_values[:-1] + current_values[1:])/2*time_diff[1:]).sum()
    return Integral_area


def charging_group_info(df):
    area_per_group = df.groupby('group_index').apply(calculate_area)
    start_per_group = df.groupby('group_index')['Sending Time'].min()
    duration_group = (df.groupby('group_index')['Sending Time'].max() - df.groupby('group_index')['Sending Time'].min()).dt.total_seconds()/3600
    current_group = df.groupby('group_index')['Total Current'].mean()
    group_df = pd.DataFrame({
        'Start Time': start_per_group.values,
        'Electric Charge(Ah)': area_per_group.values*-1,
        'Charging Duration(hour)': duration_group.values,
        'Average Current': current_group.values*-1
    })
    return group_df


def exclude_soh_outlier(group_df, soh_jump_threshold=1):
    filter_capacity = []
    for index, row in group_df.iterrows():
        if index == 0:
            filter_capacity.append(index)
        else:
            capacity = row['Electric Charge(Ah)']
            diff = abs(capacity - group_df.iloc[filter_capacity[-1]]['Electric Charge(Ah)'])
            if diff > soh_jump_threshold:
                continue
            filter_capacity.append(index)
    group_exclusion = group_df.iloc[filter_capacity].copy().reset_index()
    first_record = group_exclusion.iloc[0]
    group_exclusion['Capacity Percentage(%)'] = group_exclusion['Electric Charge(Ah)']/first_record['Electric Charge(Ah)'] * 100
    group_exclusion['Days'] = (group_exclusion['Start Time'] - first_record['Start Time']).dt.days
    return group_exclusion


def calculate_three_prediction(df):
    X = df['Days'].values.reshape(-1, 1)
    y = df['Capacity Percentage(%)'].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    # get slope and intercept from linear regression
    slope = model.coef_[0]
    intercept = model.intercept_
    residuals = y - model.predict(X)
    # find lowest and highest record above/below the regression line
    index_high = np.argmax(residuals)
    index_low = np.argmin(residuals)
    pred = slope * df['Days'].values + intercept
    low = slope * df['Days'].values - df.iloc[index_low]['Days']*slope + df.iloc[index_low]['Capacity Percentage(%)']
    high = slope * df['Days'].values - df.iloc[index_high]['Days']*slope + df.iloc[index_high]['Capacity Percentage(%)']
    return high, pred, low


def plot_three_lines_NEW(df, high, pred, low, vin):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Days'],
                             y=df['Capacity Percentage(%)'],
                             mode='lines+markers',
                             name='SOH data points'))
    fig.add_trace(go.Scatter(x=df['Days'],
                             y=high,
                             mode='lines',
                             name='Upper Prediction Line'))
    fig.add_trace(go.Scatter(x=df['Days'],
                             y=pred,
                             mode='lines',
                             name='Prediction Line'))
    fig.add_trace(go.Scatter(x=df['Days'],
                             y=low,
                             mode='lines',
                             name='Lower Prediction Line'))
    print(vin)
    fig.update_layout(
        title=f'Prediction for VIN {vin} Battery Capacity Aging',
        xaxis_title="Driving Days",
        yaxis_title="Capacity Percentage(%)"
    )
    fig.show()