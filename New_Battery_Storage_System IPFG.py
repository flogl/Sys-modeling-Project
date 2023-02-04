#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:46:43 2022

@author: florianglogger
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:46:43 2022

@author: florianglogger
"""

# Imports
#import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
from sklearn.linear_model import LinearRegression

# Input data
#1. Fixed data / Assumptions

# Storage
StorageCapacity = 5 # given by source / data sheet of actually available storage. One data sheet uploaded in One Drive.
StorageCapacity_PVonly = 0 # for system simulation PV only
StoragePower = 3.4 # in kW, given by source / data sheet, usually smaller than capacity.
StoragePower_PVonly = 0 # for system simulation PV only
efficiency_charge = 0.95 # efficiency of charge and discharge given by source
efficiency_discharge = 0.95
efficiency_storage = 0.996 # self discharge rate ranging from 2-3 % per month (see source). So 0.004 % per hour equals 2.88 % self-discharge per month.
CAPEXstorage = 1000 # in € / kWh
LifetimeStorage = 20 # in years

# PV
Tilt_angle = 35 
CAPEXpv = 1400 # in € / kW
LifetimePV = 20 # in years

# Grid
RetailPrice = 0.35 # in € / kWh, evtl. mit Liste und fluktuierenden Preisen, Prof fragen
FeedInPrice = 0.08 # in € / kWh, evtl. mit Excel Liste und Börsenstrompreisen

# General
InterestRate = 0.02

# 2. Spatial conditions and pv setup

City = input('The cities of interest are Freiburg and Hamburg: ')

while City != 'Freiburg' and City != 'Hamburg':
    print('Please type in a valid city name.')
    City = input('The cities of interest are Freiburg and Hamburg: ')

print('The installed PV system power is predefined by PolySun to 8.4 kW.') # solar cell efficiency = 21 % (2023)
installed_pv_question = input('Do you want to scale down / up the installed power of PV? (y / n)? ')

if installed_pv_question == 'y':
    installed_pv = float(input('The installed PV system power [in kW; max. 10 kW]: '))
if installed_pv_question == 'n':
    installed_pv = 8.4

while float(installed_pv) <= 0 or installed_pv > 10:
    print('Please type in a valid PV system power.')
    installed_pv = float(input('The installed PV system power [in kW; max. 10 kW]: '))

#3. import pv generation from Polysun
pv_polysun_df = pd.read_csv(f'inputs/pv_polysun_{City}.csv', sep=',', decimal='.',thousands = ',', header=0, index_col=('Hour'))
pv_gen = pd.DataFrame({'Gen [kWh]':pv_polysun_df[f'Energy_production_AC_{Tilt_angle}']})
pv_gen.index.names = ['Timesteps']
pv_gen['Gen [kWh]'] /= (1000*(8.4/installed_pv)) # PolySun values for 8.4 kW in Wh, dividing by 1000 for kWh and ratio for chosen installed power.

#4. import demand from LoadProfileGenerator
demand = pd.read_csv(f'inputs/load_profile_{City}.csv', sep=';', decimal='.', header=0)
demand.index=pv_gen.index #same index
del demand['Time']
del demand['Electricity.Timestep']
demand.rename(columns={'Sum [kWh]': 'Load [kWh]'}, inplace=True)

#5. calculation of total demand, generation and costs
pv_gen_total = float(pv_gen.sum())
demand_household_total = float(demand.sum())

#6. time horizon
N = 8760

#7. defining model
def rule_energy_balance(model, i):
    exp = demand['Load [kWh]'].iat[i]==pv_gen['Gen [kWh]'].iat[i]-model.feed_in[i]-model.p_charge[i]+model.p_discharge[i]+model.con_grid[i]
    return exp

def rule_ss_share(model,i): # share of self sufficiency, could also be done via pv self-consumption rate: sc_share = 1-(m.feed_in[i]/pv_gen[i])
    exp = model.ss_share[i]== 1-(model.con_grid[i]/demand['Load [kWh]'].iat[i])
    return exp

def rule_obj(model): # objective rule for maximising self sufficency rate for given storage size, could also be done via self-consumption rate from pv
    exp = (pyo.quicksum(model.ss_share[i] for i in model.T)/N)
    return exp

def rule_storage_SOC(model,i):
    if i<N-1:
        exp = model.SOC[i+1]==(model.SOC[i]*efficiency_storage+model.p_charge[i]*efficiency_charge
                            -model.p_discharge[i]/efficiency_discharge)
        return exp
    else:
        exp = model.SOC[i] == model.SOC[0]
        return exp

def rule_capacity(model,i):
    exp = model.SOC[i] <= StorageCapacity
    return exp

def storageSystemOperationModel(StoragePower, StorageCapacity, efficiency_storage, efficiency_charge, efficiency_discharge, demand, pv_gen, N):
    model = pyo.ConcreteModel()
    
    # sets
    model.T = pyo.RangeSet(0, N-1)
    
    # variables
    model.con_grid = pyo.Var(model.T, domain = pyo.NonNegativeReals)
    model.feed_in = pyo.Var(model.T, domain = pyo.NonNegativeReals)
    model.p_charge = pyo.Var(model.T, domain = pyo.NonNegativeReals, bounds = (0,StoragePower))
    model.p_discharge = pyo.Var(model.T, domain = pyo.NonNegativeReals, bounds = (0,StoragePower))
    model.ss_share = pyo.Var(model.T, domain = pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.T, domain = pyo.NonNegativeReals, bounds = (0, StorageCapacity))
    
    # constraints
    model.constr_eb = pyo.Constraint(model.T, rule=rule_energy_balance)
    model.constr_soc = pyo.Constraint(model.T, rule=rule_storage_SOC)
    model.constr_ss = pyo.Constraint(model.T, rule=rule_ss_share)
    model.constr_cap = pyo.Constraint(model.T, rule=rule_capacity)
    
    #objective
    model.obj = pyo.Objective(rule=rule_obj, sense = pyo.maximize)
    
    #solve model
    opt = SolverFactory('glpk')
    opt.solve(model)
    
    #store variables in a list
    con_grid=[]
    feed_in=[]
    p_charge=[]
    p_discharge=[]
    ss_share=[]
    soc=[]
    for i in model.T:
        con_grid.append(pyo.value(model.con_grid[i]))
        feed_in.append(pyo.value(model.feed_in[i]))
        p_charge.append(pyo.value(model.p_charge[i]))
        p_discharge.append(pyo.value(model.p_discharge[i]))
        ss_share.append(pyo.value(model.ss_share[i]))
        soc.append(pyo.value(model.SOC[i]))

    #store lists in one DataFrame
    operation_df = pv_gen.copy()
    operation_df['Load [kWh]'] = demand['Load [kWh]']
    operation_df['grid consumption'] = con_grid
    operation_df['grid feed in'] = feed_in
    operation_df['charge of storage'] = p_charge
    operation_df['discharge of storage'] = p_discharge
    operation_df['state of charge'] = soc
    operation_df['self sufficiency rate'] = ss_share
    return model, operation_df

# executing model PV + battery system and storing all lists from model operation into dataframe PV + battery system
model, operation_pv_bat_df = storageSystemOperationModel(StoragePower, StorageCapacity, efficiency_storage, efficiency_charge, 
                              efficiency_discharge, demand, pv_gen, N)

model, operation_pv_only_df = storageSystemOperationModel(StoragePower_PVonly, StorageCapacity_PVonly, efficiency_storage, efficiency_charge, 
                              efficiency_discharge, demand, pv_gen, N)

# function for npv calculation a) PV + Battery b) PV only
# 1. calculating  grid costs for one year of operation
# 2. defining and calculating investment and operational cost
# 3. creation of DataFrame for NPV calculation and determining cash flow, discounted cash flow and accumulated cash flow
# 4. break even analysis
# 5. internal rate of return analysis
def npv(operation_pv_bat_df, StorageCapacity):
    # calculating electricity costs for consumption: a) PV + Battery or PV only (depending on Input df) b) Grid only
    grid_costs_a = (operation_pv_bat_df['grid consumption'].sum()*RetailPrice) - (operation_pv_bat_df['grid feed in'].sum()*FeedInPrice) # negative grid costs indicate positive cash flow due to more remuneration from feed in than payment from consumption!
    grid_costs_b = demand_household_total * RetailPrice
    
    # investment and operational costs for PV and battery
    CAPEXpv_total = CAPEXpv * installed_pv
    CAPEXstorage_total = CAPEXstorage * StorageCapacity
    OPEXpvbatsystem = 0.02 * (CAPEXpv_total + CAPEXstorage_total) # is OPEX = 0 or > 0?
    
    # new DataFrame for NPV calculation
    npv = pd.DataFrame(columns=['Investment', 'Cash Flow', 'Discounted Cash Flow', 'Accumulated Cash Flow'], index = np.arange(20))
    npv.index.name = 'Year'
    npv.index += 1
    
    for i in range(len(npv)):
        # investment costs for year 0
        if i == 0:
            npv['Investment'].iat[i] = -(CAPEXpv_total + CAPEXstorage_total)
        else:
            npv['Investment'].iat[i] = 0
        # cash flow = grid costs for system with no PV + battery - grid costs for PV + battery system - operational costs for PV + battery system
        npv['Cash Flow'] = grid_costs_b - grid_costs_a - OPEXpvbatsystem # cash flow equals savings in the electricity bill
        npv['Discounted Cash Flow'].iat[i]= npv['Cash Flow'].iat[i] / ((1+InterestRate)**npv.index[i])
        # accumulated cash flow
        if i == 0:
            npv['Accumulated Cash Flow'].iat[i]=npv['Investment'].iat[i]+npv['Discounted Cash Flow'].iat[i]
        else:
            npv['Accumulated Cash Flow'].iat[i]=npv['Accumulated Cash Flow'].iat[i-1]+npv['Investment'].iat[i]+npv['Discounted Cash Flow'].iat[i]
            
    # break even analysis with new for loop for breaking when break even reached
    for i in range(len(npv)):
        if npv['Accumulated Cash Flow'].iat[i] >= 0:
            break_even_year = npv.index[i]
            break
    
    # optional: calculation of internal rate of return (IRR). First new list for storing cash out- and inflows, then calculating IRR
    cash_flows_irr = []
    for i in range(len(npv)):
        if i == 0:
            cash_flows_irr.append(npv['Investment'].iat[i])
        else:
            cash_flows_irr.append(npv['Cash Flow'].iat[i])
    irr = round(npf.irr(cash_flows_irr), 4)
    return npv, break_even_year, irr

# plotting npv calculations
#def plot_npv(npv_a, npv_b):
    #ax = npv_a.plot(kind='line',use_index=True,y='Accumulated Cash Flow')
    #npv_b.plot(ax=ax, kind='line',use_index=True,y='Accumulated Cash Flow')
    
# plotting SOC and load duration curve (total PV gen and load), first sort value function -> new dataframe with sorted values and then plotting them into graphs
#def plot_soc_pv_load(output_df):
   # plot_df = pv_gen.copy()
   # del plot_df['Gen [kWh]']
    #plot_df
    
    #output_df.sort_values(by=['Gen [kWh]'], inplace=True, ascending=False)
    #output_df.plot(kind='line', use_index=True, y='Gen [kWh]')

# plotting system operation for one exemplary day

# execution of plots
#plot_npv(npv_a, npv_b)
#plot_soc_pv_load(operation_pv_bat_df)

# print most important system performance indicators / KPIs - pv output total, load total, ss rate, sc rate, average SOC, total feed in, total grid consumption, npv after 20 years, break even, IRR
#system a) (PV + battery)
print('Here are the KPIs for the PV + battery system:')
print('Self-sufficency rate:', round((operation_pv_bat_df['self sufficiency rate'].mean()*100), 0),  '%')
#system b) (PV only)
print('Here are the KPIs for the PV only system:')
print('Self-sufficency rate:', round((operation_pv_only_df['self sufficiency rate'].mean()*100), 0),  '%')
#%%
'''
# CO2 comparison with linear regression

CO2_intensity_grid_2012 = 339 # [g CO2 eq/ kWh]
CO2_intensity_grid_2018 = 289

Primary_energy_factor_gird = 2.1 # this PEF is from 2018
PEF_gird_2012 = 2.5
PEF_gird_2018 = 2.1
PEF_gird_2050 = 1
PEF_gird_2060 = 1
PEF_gird_2070 = 1

X = ([[2012],
      [2018],
      [2050],
      [2060],
      [2070]])

y = np.log([847.5,606.9,0.001,0.0001,0.00001])


model_PEF = LinearRegression()
model_PEF.fit(X,y)

model_PEF.coef_, model_PEF.intercept_

xs = np.arange(2010,2080,1)
ys = np.exp(xs*model_PEF.coef_[0] + model_PEF.intercept_)

plt.scatter(X, [847.5,606.9,0.001,0.0001,0.00001], alpha=0.3)
plt.plot(xs,ys,'r-',lw=3)
df_PEF=pd.DataFrame()
years=[2023]
for i in range(20):
    years.append(i+2024)
    
df_PEF['Years'] = years

# df_PEF['PEF']=np.exp(df_PEF['Years']*-0.01710544495 + 35.2609642)'''
# CO2 intensity of electricity of the EU 27 countries in 2021 is 275 g CO2 eq/kWh
# the assumption here is that in 2050 the carbon intensity of the grid electricity reach 0, positing the carbon neutrality by 2050

X = ([[2021],[2050]])
y = [275.0,0]

model_GHG_grid = LinearRegression()
model_GHG_grid.fit(X,y)

xs = np.arange(2010,2070,1)
ys = xs*model_GHG_grid.coef_[0] + model_GHG_grid.intercept_

plt.scatter(X, y, alpha=0.3)
plt.plot(xs,ys,'r-',lw=3)
df_GHG=pd.DataFrame()

GWP_pv = 55 # global warming potential of pv [g CO2 eq/kWh_generated]
GWP_battery = 55 # global warming potential of battery [g CO2 eq/kWh_stored]

# Now we assess the GHG emissions for 1 kWh of electricity
years=[2023]
for i in range(20):
    years.append(i+2024)

df_GHG['Years'] = years

Carbon_intensity_of_each_year = []

for i in range(len(years)):
    Carbon_intensity_of_each_year.append(years[i]*model_GHG_grid.coef_[0] + model_GHG_grid.intercept_)
    
df_GHG['CO2_intensity_grid [g CO2 eq/kWh]'] = Carbon_intensity_of_each_year

GHG_from_pv_only_system = []
GHG_from_pvbat_system = []
for i in range(len(years)):
    GHG_from_pv_only_system.append(((operation_pv_only_df['grid consumption'].sum()*df_GHG['CO2_intensity_grid [g CO2 eq/kWh]'].iloc[i]
                                     +(operation_pv_only_df['Gen [kWh]']).sum()*GWP_pv))/(operation_pv_only_df['demand'].sum()+operation_pv_only_df['grid feed in'].sum()))
    GHG_from_pvbat_system.append(((operation_pv_bat_df['grid consumption'].sum()*df_GHG['CO2_intensity_grid [g CO2 eq/kWh]'].iloc[i]+
                                   (operation_pv_bat_df['Gen [kWh]']).sum()*GWP_pv+operation_pv_bat_df['charge of storage'].sum()*GWP_battery))//(operation_pv_bat_df['demand'].sum()+operation_pv_bat_df['grid feed in'].sum()))
df_GHG['GHG from pv-only system [g CO2 eq]'] = GHG_from_pv_only_system
df_GHG['GHG from pv-bat system [g CO2 eq]'] = GHG_from_pvbat_system



# As the renewable energy mix increases in the grid, it's expected for the manufacturing process of pv and battery to involve less carbon emissions, and, therefore, less global warming potential of the integration of them.

 
