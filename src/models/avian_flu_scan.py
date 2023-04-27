import matplotlib.pyplot as plt
import multiprocessing
from scipy.stats import gaussian_kde
from scipy.stats import norm, truncnorm, beta
import copy
from itertools import repeat
from tqdm import tqdm
from time import time
import pandas as pd
import os
import random
import numpy as np
import sys
import itertools

lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(""))), "models")


sys.path.append(lib_path)
from model_age_stratified import Model


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

    # from england
    age_groups = np.array([3076945, 9980931, 33030917, 10401308])
    age_props = age_groups / 56490101

    y1_0 = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    y1_0[0, :] = (
        age_props * pop
    )  # for now assume this is the same as in england as doesnt affect anything, but for future modelling can have different age distributions in source country

    # make parameter set generator object
    param_generator = (
        {
            "y1_0": y1_0,
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
            "alpha": [1 / np.random.lognormal(1.21, 0.23)] * 4,
            "gamma": [1 / np.random.lognormal(1.50, 0.33)] * 4,
        }
        for i in range(10000)
    )

    MC_results = []

    for i in range(1000):
        params = next(param_generator)

        sim = Model(**params)

        sim.simulate()
        IF_timeseries = np.array(sim.IF_timeseries).sum(axis=1) > 0

        t_seed = (
            IF_timeseries.nonzero()[0][0]
            if np.sum(IF_timeseries) > 0
            else len(IF_timeseries) - 1
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

        # get culmulative number of tests done at detection
        n_detect = sim.wd_count[t_detect]
        n_screen = sim.bd_count[t_screen]
        n_screen_h = sim.hd_count[t_screen_h]
        n_screen_c = sim.cd_count[t_screen_c]

        # get culmulative number of tests done in total
        n_ww = sim.wd_count[-1]
        n_bd = sim.bd_count[-1]
        n_hd = sim.hd_count[-1]
        n_cd = sim.cd_count[-1]

        # total hospitilisations
        c_hosp_1, c_hosp_2, c_hosp_3, c_hosp_4 = sim.c_hosp[-1]

        # total infections
        c_inf_1, c_inf_2, c_inf_3, c_inf_4 = sim.solution_2[-1][-1]

        # get culmulative infections at detection
        c_detect_1, c_detect_2, c_detect_3, c_detect_4 = sim.solution_2[t_detect][-1]
        c_screen_1, c_screen_2, c_screen_3, c_screen_4 = sim.solution_2[t_screen][-1]
        (c_screen_h_1, c_screen_h_2, c_screen_h_3, c_screen_h_4,) = sim.solution_2[
            t_screen_h
        ][-1]
        c_screen_c_1, c_screen_c_2, c_screen_c_3, c_screen_c_4 = sim.solution_2[
            t_screen_c
        ][-1]

        results_dict = dict(
            simulation=i,
            phi_bd=phi_bd,
            phi_hd=phi_hd,
            phi_cd=phi_cd,
            phi_hs=phi_hs,
            phi_cs=phi_cs,
            R0=R0,
            ihr_1=ihr[0],
            ihr_2=ihr[1],
            ihr_3=ihr[2],
            ihr_4=ihr[3],
            seed_t=t_seed,
            ww_time=t_detect,
            bd_time=t_screen,
            hd_time=t_screen_h,
            cd_time=t_screen_c,
            hs_time=t_screen_hs,
            cs_time=t_screen_cs,
            ww_nd_tests=n_detect,
            bs_nd_tests=n_screen,
            hs_nd_tests=n_screen_h,
            cs_nd_tests=n_screen_c,
            ww_nt_tests=n_ww,
            bs_nt_tests=n_bd,
            hs_nt_tests=n_hd,
            cs_nt_tests=n_cd,
            c_hosp_1=c_hosp_1,
            c_hosp_2=c_hosp_2,
            c_hosp_3=c_hosp_3,
            c_hosp_4=c_hosp_4,
            c_inf_1=c_inf_1,
            c_inf_2=c_inf_2,
            c_inf_3=c_inf_3,
            c_inf_4=c_inf_4,
            ww_c_infected_1=c_detect_1,
            ww_c_infected_2=c_detect_2,
            ww_c_infected_3=c_detect_3,
            ww_c_infected_4=c_detect_4,
            bs_c_infected_1=c_screen_1,
            bs_c_infected_2=c_screen_2,
            bs_c_infected_3=c_screen_3,
            bs_c_infected_4=c_screen_4,
            hs_c_infected_1=c_screen_h_1,
            hs_c_infected_2=c_screen_h_2,
            hs_c_infected_3=c_screen_h_3,
            hs_c_infected_4=c_screen_h_4,
            cs_c_infected_1=c_screen_c_1,
            cs_c_infected_2=c_screen_c_2,
            cs_c_infected_3=c_screen_c_3,
            cs_c_infected_4=c_screen_c_4,
        )
        results_dict.update(params)
        # convert vector params that are all the same to scalars for the dataframe
        results_dict["alpha"] = results_dict["alpha"][0]
        results_dict["gamma"] = results_dict["gamma"][0]
        results_dict["pop"] = pop  # add population explicitly

        MC_results.append(results_dict)

    return MC_results


phi_bds = [0.1, 0.3, 0.5]
phi_hds = [0.1, 0.3, 0.5]
# phi_bds = [0.1, 0.2]
# phi_hds = [0.1, 0.2]

# between 10000 and 100000 tests per week in simulation doc
tests_per_week = [15000, 50000, 100000]
# tests_per_week = [10000, 25000]
phi_cds = [
    t / 7 / 67e6 for t in tests_per_week
]  # convert to proportion of UK pop. per day
print(phi_cds)

phi_hss = [0.1, 0.3, 0.5]
phi_css = copy.deepcopy(phi_cds)  # same as the UK ones
# phi_hss = [0.1, 0.2]
# phi_css = [2.1321961620469086e-05, 5.330490405117271e-05] # same as the UK ones

R0s = [1.2, 1.6, 2.0]
ihrs_es = [
    [10, 10, 10, 10],
    [0.0409, 0.4091, 4.0913, 40.9133],
    [38.152, 3.815, 0.382, 38.152],
]
ihrs_s = [
    [5, 5, 5, 5],
    [0.0205, 0.2046, 2.0457, 20.4566],
    [33.552, 3.355, 3.355, 3.355],
]
ihrs_m = [[1, 1, 1, 1], [0.0041, 0.0409, 0.4091, 4.0913], [3.82, 0.38, 0.04, 3.82]]
ihrs = ihrs_es
ihrs.extend(ihrs_s)
ihrs.extend(ihrs_m)

ihrs = np.array(ihrs) / 100


save_dir = os.path.join("..", "..", "models", "simulated_data", "avian_flu")


# ['USA', 'Spain', 'Netherlands', 'India', 'Italy', 'Germany', 'France', 'China']
for source_country in reversed(["France"]):
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

    print(len(list(itertools.product(testing_params, disease_params))))
    ti = time()

    with multiprocessing.Pool() as pool:
        res = pool.starmap(screening_analysis, all_params)
        MC_results = list(itertools.chain.from_iterable(res))
    """

    # non parellel for testing
    for p in all_params:
        res = screening_analysis(p[0], p[1])
    """

    print(source_country, time() - ti)

    res_df = pd.DataFrame(MC_results)
    os.makedirs(save_dir, exist_ok=True)
    res_df.to_csv(os.path.join(save_dir, source_country + ".csv"))
