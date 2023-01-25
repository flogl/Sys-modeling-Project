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

# Input data
#1. Fixed data / Assumptions
# Storage
StorageCapacity = 10 # in kWh, maximal storage capacity within optimisation model.
StoragePower = 10 # in kW, maximal storage power within optimisation model.
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

def rule_ss_share(model,i): # share of self sufficiency
    exp = model.ss_share[i]== 1-(model.con_grid[i]/demand['Load [kWh]'].iat[i])
    return exp

def rule_storage_SOC(model,i):
    if i<N-1:
        exp = model.SOC[i+1]==(model.SOC[i]*efficiency_storage+model.p_charge[i]*efficiency_charge
                            -model.p_discharge[i]/efficiency_discharge)
        return exp
    else:
        exp = model.SOC[i] == model.SOC[0]
        return exp
    
def rule_min_share(model):
    exp = (pyo.quicksum(model.ss_share[i] for i in model.T)/N) >= model.share
    return exp

def rule_capacity(model,i):
    exp = model.SOC[i] <= model.storageCapacity
    return exp

def objective_rule(model):
    return model.storageCapacity

def storageOptShare(StoragePower, StorageCapacity, efficiency_storage, efficiency_charge, efficiency_discharge, demand, pv_gen, N):
    model = pyo.ConcreteModel()
    
    # sets
    model.T = pyo.RangeSet(0, N-1)
    model.share = 0.90
    
    # variables
    model.con_grid = pyo.Var(model.T, domain = pyo.NonNegativeReals)
    model.feed_in = pyo.Var(model.T, domain = pyo.NonNegativeReals)
    model.ss_share = pyo.Var(model.T, domain = pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.T, domain = pyo.NonNegativeReals)
    model.storageCapacity = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0,StorageCapacity))
    model.p_charge = pyo.Var(model.T, domain = pyo.NonNegativeReals, bounds = (0,StoragePower))
    model.p_discharge = pyo.Var(model.T, domain = pyo.NonNegativeReals, bounds = (0,StoragePower))
    
    # constraints
    model.constr_eb = pyo.Constraint(model.T, rule=rule_energy_balance)
    model.constr_soc = pyo.Constraint(model.T, rule=rule_storage_SOC)
    model.constr_ss = pyo.Constraint(model.T, rule=rule_ss_share)
    model.constr_cap = pyo.Constraint(model.T, rule=rule_capacity)
    model.constr_share = pyo.Constraint(rule=rule_min_share)
    
    #objective
    model.obj = pyo.Objective(rule=objective_rule, sense = pyo.minimize)
    
    #solve model
    opt = SolverFactory('glpk')
    opt.solve(model)

    return model

model = storageOptShare(StoragePower, StorageCapacity, efficiency_storage, efficiency_charge, 
                              efficiency_discharge, demand, pv_gen, N)

print(f'Storage size needed for covering {pyo.value(model.share)*100} % self-sufficiency is: {round(pyo.value(model.obj),2)} kWh.')
print(round(pv_gen.sum(),2))
print(round(demand.sum(),2))