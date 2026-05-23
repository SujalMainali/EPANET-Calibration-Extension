import pandas as pd
import json

# Load calibration data
with open('outputs/reports/best_params.json') as f:
    bp = json.load(f)

# Show pressure-related parameters
pda = bp['best_raw_params']['pda']
print('PDA Parameters:')
print(f'  minimum_pressure: {pda["minimum_pressure"]} (units)')
print(f'  required_pressure: {pda["required_pressure"]} (units)')
print(f'  pressure_exponent: {pda["pressure_exponent"]}')

# Load observed data to see pressure ranges
obs_data = pd.read_csv('Data/HourlyData_2025-12-18.csv')
print(f'\nObserved pressure statistics (from Data CSV):')
print(obs_data.describe())
print(f'\nColumn names:')
print(obs_data.columns.tolist())
