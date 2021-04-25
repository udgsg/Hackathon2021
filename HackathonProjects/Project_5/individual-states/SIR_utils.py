#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:01:10 2021

@author: jerome
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, least_squares, minimize

def sir_ode(t, x, a, b):
    """
    ODE function for SIR model.
    Conforms to required input to solve_ivp
    WARNING: this problem is stiff! Use a stiff ODE solver!

    Parameters
    ----------
    t : float
        time.
    x : array of length 3 of floats
        x[0] = S - susceptible
        x[1] = I - infectious
        x[2] = R - removed (dead + in treatment).
    a : float > 0
        first parameter for SIR,
        reflects rate of infection.
    b : float > 0
        second parameter for SIR,
        reflects rate of removal (removing one's self from 
                                  those who can infect others).

    Returns
    -------
    dxdt : array of length 3 of floats
        dxdt[i] = dx[i] / dt.  
        dS / dt = -a * S * I
        dI / dt = a * S * I - b * I
        dR / dt = b * I

    """
    dxdt = np.zeros(3)
    dxdt[0] = -a * x[0] * x[1]
    dxdt[1] = a * x[0] * x[1] - b * x[1]
    dxdt[2] = b * x[1]
    
    return dxdt

def compute_number_infectious(current_infected, averaging_range):
    """
    Estimate the number of infectious individuals at a given time

    Parameters
    ----------
    current_infected : array of length n of ints
        the number of cumulative infected individuals at each time node.
    averaging_range : int > 0
        length of moving average window.

    Returns
    -------
    infected_avg : array of length n of floats
        approximate number of infectious individuals at each time node.
        Note values near the beginning and the end are corrupted due 
        to the boundary effects of np.convolve.
        See np.convolve() for more details on this.
    infected_roc_filled : array of length n of floats
        approximate rate of change of infectious individuals.
        Built using second order centered differences 
        using a window of length averaging_range

    """
    averager = np.ones(averaging_range) / averaging_range
    N = len(current_infected)
    
    # build number of infected individuals
    infected = current_infected[1:] - current_infected[:-1]
    infected_avg = np.convolve(averager, infected, mode="same")
    infected_avg_good = np.convolve(averager, infected, mode="valid")
    
    # rate of change of infected
    infected_roc = np.zeros_like(infected_avg_good)
    infected_roc[1:] = (infected[averaging_range:] - 
                    infected[:-averaging_range]) / averaging_range
    infected_roc = np.convolve(averager, infected_roc, mode="same")
    print(len(infected_roc))
    infected_roc[0] = infected_roc[1]
    
    return infected_avg_good, infected_roc
    
def compute_sir_coefs(times, susceptible, infected, removed, 
                      a_guess=1, b_guess=1, 
                      ode_options={}, opt_options={}):
    """
    Compute the coefficients (a, b) for the SIR model 
    a corresponds to infectivity / virilance, 
    b corresponds to removal rate from infectious population 
        (e.g. getting tested and quarentining)
        
    Parameters
    ----------
    times : array of length n
        time nodes.
    susceptible : array of length n
        susceptible population at time nodes.
    infected : array of length n
        infectious population at time nodes.
    removed : array of length n
        removed population at time nodes.
    a_guess : float > 0, optional
        initial guess for a. The default is 1.
    b_guess : float > 0, optional
        initial guess for b. The default is 1.
    ode_options : dictionary, optional
        options to pass to the ODE solver. The default is {} (no options).
        See scipy.integrate.solve_ivp for more details
    opt_options : dictionary, optional
        options to pass to the least squares solver. The default is {} 
        (no options).
        See scipy.optimize.least_squares for more details

    Returns
    -------
    opt_res : array of length 2 of floats
        opt_res = [a_fitted, b_fitted].

    """
    # store true values in one array
    truth = np.zeros([len(times), 3])
    truth[:, 0] = susceptible
    truth[:, 1] = infected
    truth[:, 2] = removed
    # extra parameters for ode solving
    init_cond = [susceptible[0], infected[0], removed[0]]
    tspan = [0, times[-1]]
    
    # run the ode solver
    test_fun = lambda t, a, b: solve_ivp(sir_ode, [0, times[-1]], init_cond, 
                                        t_eval=t, args=(a, b), 
                                        method="LSODA", **ode_options)
    
    error = lambda coef: test_fun(times, coef[0], coef[1]).y[2, :] - removed
    
    
    
    #opt_res = curve_fit(test_fun, times, removed)
    #opt_res = least_squares(error, (a_guess, b_guess), **opt_options)
    opt_res = minimize(lambda coef: np.linalg.norm(error(coef), 2), 
                       (a_guess, b_guess),
                       **opt_options)
    return opt_res
        
def get_a_b_guesses(sus, inf, rem):
    
    a_guess = -np.mean((sus[1:] - sus[:-1]) / (inf[1:] * sus[1:]))
    b_guess = np.mean((rem[1:] - rem[:-1]) / inf[1:])
    return a_guess, b_guess
    
def moving_averages_fits(times, sus, inf, rem, window=14, 
                         a_guess=1, b_guess=1, 
                         ode_options={}, opt_options={}):
    """
    Wrapper for performing multiple fits on a time series
    
    This method uses the previous values for a and b as 
    initial guess for the next fitting

    Parameters
    ----------
    times : array of length n
        time nodes.
    susceptible : array of length n
        susceptible population at time nodes.
    infected : array of length n
        infectious population at time nodes.
    removed : array of length n
        removed population at time nodes.
    window : int > 0, optional
        window averaging size for fitting. The default is 14.
    a_guess : float > 0, optional
        initial guess for a. The default is 1.
    b_guess : float > 0, optional
        initial guess for b. The default is 1.
    ode_options : dictionary, optional
        options to pass to the ODE solver. The default is {} (no options).
        See scipy.integrate.solve_ivp for more details
    opt_options : dictionary, optional
        options to pass to the least squares solver. The default is {} 
        (no options).
        See scipy.optimize.least_squares for more details

    Returns
    -------
    fitting_values : m x 2 array of floats
        the first column corresponds to the fitted a values,
        the second column corresponds to the fitted b values.

    """
    a_prev = a_guess
    b_prev = b_guess
    
    num_fits = int(len(times) / window)
    fitting_values = np.zeros([num_fits, 2])
    
    for j in range(num_fits):
        # window for indexing
        start = window * j
        stop = start + window
        a_guess, b_guess = get_a_b_guesses(sus[start:stop], 
                                           inf[start:stop], 
                                           rem[start:stop])
        opt_res = compute_sir_coefs(times[start:stop], sus[start:stop], 
                                    inf[start:stop], rem[start:stop], 
                                    a_guess=a_guess, b_guess=b_guess, 
                                    ode_options=ode_options, 
                                    opt_options=opt_options)
        
        # update a, b guesses
        a_prev, b_prev = opt_res.x
        # update array of solutions
        fitting_values[j, :] = opt_res.x
        
    return fitting_values
        
        
    
    
    
    


if __name__ == "__main__":
    print("Executing Tests")
    
    # testing params
    fname = "new-jersey-history.csv"
    averaging_range = 14
    
    # raw data frame contains all fields and many NaNs
    raw_dataframe = pd.read_csv(fname)
    print(raw_dataframe.keys())
    
    # what to look at
    keys_to_use = ["date", "positiveCasesViral"]
    # flip order to go chronologically
    dataframe = raw_dataframe[keys_to_use].dropna(axis="rows")[::-1]
    
    # get the number of infectious individuals at each time
    # book-keeping
    infectious_key = "infectious"
    infectious_roc_key = "infectiousROC"
    keys_to_use.append(infectious_key)
    keys_to_use.append(infectious_roc_key)
    # first and second derivative
    infectious, infectious_roc = compute_number_infectious(
        np.array(dataframe[keys_to_use[1]]), 
        averaging_range)
    
    # assign to dataframe
    #dataframe[infectious_key] = infectious
    #dataframe[infectious_roc_key] = infectious_roc

    trim = 360
    #plt.plot(dataframe[keys_to_use[2]][:trim], dataframe[keys_to_use[3]][:trim], "*")
    plt.plot(infectious, infectious_roc, "*-")
    plt.show()
    start = 0
    stop = 14
    pop = 8e7
    rem = np.convolve(np.ones(averaging_range), 
                      np.array(dataframe[keys_to_use[1]]), 
                      mode="valid")[:-1] / averaging_range
    sus = pop - rem - infectious
    
    
    plt.plot(sus, infectious)
    plt.show()
    plt.plot(infectious, rem)
    plt.show()
    plt.plot(sus + rem + infectious, infectious_roc)
    plt.show()
    
    times = np.arange(len(sus))
    
    res = compute_sir_coefs(times[start:stop], sus[start:stop], 
                                infectious[start:stop], rem[start:stop], 
                                a_guess=1.25e-8, b_guess=0.68, 
                                opt_options={"method": "TNC"})
        
    print(res)
    sol = solve_ivp(sir_ode, [times[start], times[stop]], 
                    [sus[0], infectious[0], rem[0]], 
                    args=res.x, method="LSODA")
    
    plt.plot(sol.t, sol.y[2, :], "--")
    plt.plot(times[start:stop], rem[start:stop], "*")
    plt.show()
    
    fitting = moving_averages_fits(times, sus, infectious, rem, window=14, 
                                   a_guess=1.25e-8, b_guess=0.68)
    
    plt.plot(fitting[:, 0])
    plt.show()
    plt.plot(fitting[:, 1])
    plt.show()

    
    