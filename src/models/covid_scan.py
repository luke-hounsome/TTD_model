import matplotlib.pyplot as plt
import multiprocessing
from scipy.stats import gaussian_kde
from scipy.stats import norm, truncnorm, beta

# from copy import copy
from itertools import repeat
from tqdm import tqdm
from time import time
from model import Model
import pandas as pd
import os
import random
import numpy as np
import sys
import itertools

lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(""))), "models")


sys.path.append(lib_path)

flight_df = pd.read_csv(
    os.path.join("..", "..", "data", "processed", "country_flights.csv")
)


def screening_analysis(testing_params, disease_params):
    """Simulates the model 1000 for a given screening coverages at border, hospitals and community testing. Other
    parameters are sampled from distributions, or left as default."""
    pop, seats, phi_bd, phi_hd, phi_cd, phi_hs, phi_cs = testing_params
    R0, ihr = disease_params

    ps_mu = 0.459
    ps_sig2 = 0.168**2
    ps_alpha = ((1 - ps_mu) / ps_sig2 - 1 / ps_mu) * ps_mu**2
    ps_beta = ps_alpha * ((1 / ps_mu) - 1)
    # make parameter set generator object
    param_generator = (
        {
            "y1_0": [pop, 1, 0, 0, 0, 0],
            "pd": random.uniform(0.1, 0.6),
            "n": seats,
            "ps": beta(ps_alpha, ps_beta).rvs(),
            "delta": 1 / random.uniform(3, 28),
            "phi_bd": phi_bd,
            "phi_hd": phi_hd,
            "phi_cd": phi_cd,
            "phi_hs": phi_hs,
            "phi_cs": phi_cs,
            "R0": R0,
            "ihr": ihr,
            "eta": 0.85,
        }
        for i in range(10000)
    )

    MC_results = []

    for i in range(1000):

        params = next(param_generator)

        sim = Model(**params)
        t = time()
        sim.simulate()
        print("sim time", time() - t)

        t_seed = (
            np.argmax(np.array(sim.IF_timeseries) > 0)
            if len(sim.IF_timeseries) > 0
            else len(sim.IF_timeseries) - 1
        )

        # get time of detections
        t_detect = (
            sim.wd_timeseries.index(1)
            if 1 in sim.wd_timeseries
            else len(sim.wd_timeseries) - 1
        )
        t_screen = (
            sim.bd_timeseries.index(1)
            if 1 in sim.bd_timeseries
            else len(sim.bd_timeseries) - 1
        )
        t_screen_h = (
            sim.hd_timeseries.index(1)
            if 1 in sim.hd_timeseries
            else len(sim.hd_timeseries) - 1
        )
        t_screen_c = (
            sim.cd_timeseries.index(1)
            if 1 in sim.cd_timeseries
            else len(sim.cd_timeseries) - 1
        )

        t_screen_hs = (
            sim.hs_timeseries.index(1)
            if 1 in sim.hs_timeseries
            else len(sim.hs_timeseries) - 1
        )
        t_screen_cs = (
            sim.cs_timeseries.index(1)
            if 1 in sim.cs_timeseries
            else len(sim.cs_timeseries) - 1
        )

        # get culmulative infections at detection
        c_detect = sim.solution_2[t_detect][-1]
        c_screen = sim.solution_2[t_screen][-1]
        c_screen_h = sim.solution_2[t_screen_h][-1]
        c_screen_c = sim.solution_2[t_screen_c][-1]

        # get culmulative number of tests done at detection
        n_detect = sim.wd_count[t_detect]
        n_screen = sim.bd_count[t_screen]
        n_screen_h = sim.hd_count[t_screen_h]
        n_screen_c = sim.cd_count[t_screen_c]

        results_dict = dict(
            simulation=i,
            phi_bd=phi_bd,
            phi_hd=phi_hd,
            phi_cd=phi_cd,
            phi_hs=phi_hs,
            phi_cs=phi_cs,
            R0=R0,
            ihr=ihr,
            seed_t=t_seed,
            ww_time=t_detect,
            bd_time=t_screen,
            hd_time=t_screen_h,
            cd_time=t_screen_c,
            hs_time=t_screen_hs,
            cs_time=t_screen_cs,
            ww_c_infected=c_detect,
            bs_c_infected=c_screen,
            hs_c_infected=c_screen_h,
            cs_c_infected=c_screen_c,
            ww_n_tests=n_detect,
            bs_n_tests=n_screen,
            hs_n_tests=n_screen_h,
            cs_n_tests=n_screen_c,
        )
        results_dict.update(params)
        MC_results.append(results_dict)

    return MC_results


save_dir = os.path.join("..", "..", "models", "simulated_data", "covid_scan")

# ['USA', 'Spain', 'Netherlands', 'India', 'Italy', 'Germany', 'France', 'China']
for source_country in reversed(
    ["USA", "Spain", "Netherlands", "India", "Italy", "Germany", "France", "China"]
):
    print(source_country)
    # ATM assuming that the arriving and departing number of seat is symmetric as only have incoming flights to UK
    seats, pop = flight_df.loc[
        flight_df["Name"] == source_country, ["Seats", "Population"]
    ].values[0]
    seats //= 31  # get average number of seats per day
    seats = int(seats)
    pop = int(pop)

    disease_params = list(itertools.product(R0s, ihrs))
    testing_params = list(
        zip(repeat(pop), repeat(seats), phi_bds, phi_hds, phi_cds, phi_hss, phi_css)
    )
    all_params = itertools.product(testing_params, disease_params)

    ti = time()
    with multiprocessing.Pool() as pool:
        res = pool.starmap(screening_analysis, all_params)
        MC_results = list(itertools.chain.from_iterable(res))

    print(source_country, time() - ti)

    sim_df = pd.DataFrame(MC_results)
    os.makedirs(save_dir, exist_ok=True)
    sim_df.to_csv(os.path.join(save_dir, source_country + ".csv"))
