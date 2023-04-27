#commit on 27/4/2023

import numpy as np
from scipy.integrate import odeint
import random
from time import time


def ode(y, t, beta, alpha, gamma, delta, N0):
    """Define the system of ODEs as above. We will include a cumulative infections
    compartment, Ic, which simply counts all transitions out of E.

    parameters
    ------
    y : list
        current state of system
    t : float
        current time step
    beta : float
        model parameter (infectivity)
    alpha : float
        model parameter (1/latent period)
    gamma : float
        model parameter (1/infectious period)
    delta : float
        model parameter (1/post-symptomatic fecal shedding period)
    N0 : int
        Population

    returns
    -------
    dydt : list
        solution of ode at next timestep.
    """
    S, E, I, F, R, Ic = y
    dydt = [
        -beta * S * I / N0,  # dS
        beta * S * I / N0 - alpha * E,  # dE
        alpha * E - gamma * I,  # dI
        gamma * I - delta * F,  # dF
        delta * F,  # dR
        alpha * E,  # dIc (cummulative infections)
    ]
    return dydt


class Flight:
    """Simulates flights between the seed country and the UK, and the detection of pathogen in
    wastewater and border screening.

    Attributes
    ----------
    departure_epi_state : list
        List of int values of length 5 detailing the number of people in [S,E,I,F,R]
        states in the country of origin at the point of departure.
    model_params : dict
        parameters of model, must include number on flight (n), defecation probability (p_d_,
        faecal shaedding probability (p_s), screening coverage (phi), screening sensitivity (eta)
        and limit of detection (L).
    nS, nE, nI, nF, nR : ints
        Number of people in flight in each disease state
    states : list
        possible disease states of the model [S, E, I, F, R]

    Properties
    ----------
    ww_detected : Bool
        True if pathogen has been detected in wastewater
    rs_detected : Bool
        True if pathogen detected in respiratory swabbing
    passenger_states : list
        list of passengers by disease state, e.g. [S, S, S, E, E, E, I , F, R, R, R...]

    """

    states = ["S", "E", "I", "F", "R"]

    def __init__(self, departure_epi_state: list, phi, **model_params):
        """
        Parameters
        ----------
        departure_epi_state : list
            List of int values of length 5 detailing the number of people in [S,E,I,F,R]
            states in the country of origin at the point of departure.
        model_params : dict or kwargs
            parameters of model, must include number on flight (n), defecation probability (p_d_,
            faecal shaedding probability (p_s), screening coverage (phi), screening sensitivity (eta)
            and limit of detection (L).
        """
        self.departure_epi_state = departure_epi_state
        model_params["phi"] = phi
        self.model_params = self.validate_model_params(model_params)

        # self.model_params = model_params

        N = sum(self.departure_epi_state)
        state_prevalence = [state / N for state in self.departure_epi_state]

        rng = np.random.default_rng()
        try:
            self.nS, self.nE, self.nI, self.nF, self.nR = rng.multinomial(
                self.model_params["n"], state_prevalence
            )
        except Exception as e:
            print(
                "numpy random number generator error, probably small -ve number",
                state_prevalence,
            )

    def __len__(self):
        return self.model_params["n"]

    @property
    def state_counts(self):
        return [self.nS, self.nE, self.nI, self.nF, self.nR]

    @property
    def ww_detected(self):
        """Checks if pathogen detected in wastewater returns True if detected, or None if not."""
        total_stools = np.random.binomial(len(self), self.model_params["pd"])
        IF_stools = np.random.binomial(total_stools, (self.nI + self.nF) / len(self))
        positive_stools = np.random.binomial(IF_stools, self.model_params["ps"])
        # check if limit of detection exceeded
        try:
            if positive_stools / total_stools > self.model_params["L"]:
                return True
        except ZeroDivisionError:
            return False

    @property
    def rs_detected(self):
        """Checks if pathogen detected in resp screening returns True if detected, or None if not."""
        I_tested = np.random.binomial(self.nI, self.model_params["phi"])
        positive_tests = np.random.binomial(I_tested, self.model_params["eta"])
        if positive_tests > 0:
            return True

    def validate_model_params(self, model_params):
        """Checks supplied parameters are sensible
        parameters
        ----------
        model_params : dict
            Dictionary of model parameters to be checked
        returns
        -------
        model_params : dict
            The same dictionary with extraneuous parameters removed.
            An error will have been thrown if it didn't pass the checks.
        """
        # Remove any parameters which are not needed by the model
        required_params = ["n", "ps", "pd", "phi", "eta", "L"]
        model_params = {k: v for k, v in model_params.items() if k in required_params}
        self.check_parameters_present(model_params)
        self.check_parameter_types(model_params)
        self.check_parameter_bounds(model_params)
        return model_params

    def check_parameters_present(self, model_params):
        """Checks all parameters are present
        Parameters
        ----------
        model_params : Dict
            Dictionary of model parameters to be checked
        Raises
        ------
        ValueError if any required parameters are missing
        """
        for param in ["n", "ps", "pd", "phi", "eta", "L"]:
            if not param in model_params.keys():
                raise ValueError(f"Ensure {param} in **model_params")

    def check_parameter_types(self, model_params):
        """Checks all parameters are of correct type.
        Parameters
        ----------
        model_params : Dict
            Dictionary of model parameters to be checked
        Raises
        ------
        ValueError if any parameters are not correct type.
        """
        # check types
        p_types = {
            "n": int,
            "ps": float,
            "pd": float,
            "phi": float,
            "eta": float,
            "L": float,
        }
        for p in model_params:
            if not isinstance(model_params[p], p_types[p]):
                raise ValueError(f"Ensure {p} is {p_types[p]}")

    def check_parameter_bounds(self, model_params):
        """Checks all parameters are within correct bounds.
        Parameters
        ----------
        model_params : Dict
            Dictionary of model parameters to be checked
        Raises
        ------
        ValueError if any =parameters are outside of required bounds.
        """
        p_bounds = {
            "n": (0, np.inf),
            "ps": (0, 1),
            "pd": (0, 1),
            "phi": (0, 1),
            "eta": (0, 1),
            "L": (0, 1),
        }
        for p in model_params.keys():
            if (model_params[p] < p_bounds[p][0]) or (model_params[p] > p_bounds[p][1]):
                raise ValueError(f"Ensure {p_bounds[p][0]} <= {p} <= {p_bounds[p][1]}")


class Model:
    """Object for simulating the ODE above.
    kwargs
    ----
    R0 : float
        Reproductive number
    alpha : float < 1
        ODE parameter latent rate
    gamma : float < 1
        ODE parameter recovery rate
    y0_1 : list
        initial conditions for seed country [S, E, I , F, R, Ic]
    y0_2 : list
        initial conditions for UK
    t_max : list
        time step to integrate up to.
    n : int
        number of return passengers per day
    pd : float < 1
        probability of an individual defecating on flight
    ps : float < 1
        probability of an infectious individual shedding viral RNA in faeces
    L : float < 1
        Limit of detection in infectious stools/total stools
    phi : float < 1
        Proportion of passengers screened via respiratory swabs (phi in model)
    eta : float < 1
        Sensitivity of respiratory swabbing (eta in model)
    ihr: float < 1
        Infection hospitilisation rate

    Properties
    ----------
    default_params : dict
        Defaults for the model. These will be used for params not specified by user.
    params : dict
        The same as the default parameter dictionary, however any user-defined parameters
        have been updated. Derived parameters (beta, N1, N2) also added.
    t_steps : np.array
        List of time steps in the model.
    """

    states = ["S", "E", "I", "F", "R", "Ic"]

    def __init__(self, **params):
        # set params and update from **kwargs
        self.user_params = params
        self.reset_results()

    @property
    def default_params(self):
        """Sets the default parameters for the model"""
        return dict(
            y1_0=[
                10e6,
                1,
                0,
                0,
                0,
                0,
            ],  # intial state of seed country S, E, I, F, R, Ic
            y2_0=[6.7e7, 0, 0, 0, 0, 0],  # initial state of UK S, E, I, F, R, Ic
            R0=3.0,  # Reproductive number
            ihr=0.0686,  # infection, hospitilisation ratio
            alpha=1 / 5.2,  # latency rate
            gamma=1 / 8,  # recovery rate
            delta=1 / 10,  # faecal shedding recovery rate
            t_max=500,  # number of time steps (days)
            n=250,  # number on board flight
            pd=0.36,  # defecation probability
            ps=0.459,  # faecal shedding probability
            L=0.0,  # Limit of detection in wastewater (infectious stools/stool)
            phi_hs=0.2,  # respiratory screening coverage (in hospitals in source country)
            phi_cs=0.0002,  # respiratory screening coverage (in community in source country)
            phi_bd=0.2,  # respiratory screening coverage (at border in destination country)
            phi_hd=0.2,  # respiratory screening coverage (in hospitals in destination country)
            phi_cd=0.0002,  # respiratory screening coverage (in community in destination country)
            eta=0.85,  # respiratory screening sensitivity
        )

    @property
    def params(self):
        """Updates parameters to include user defined and derived params."""
        params = self.default_params
        params.update(self.user_params)
        # derived params
        params.update(
            {
                "N1": sum(params["y1_0"]),
                "N2": sum(params["y2_0"]),
                "beta": params["gamma"] * params["R0"],
            }
        )
        return params

    @property
    def t_steps(self):
        """Time steps for the model"""
        T = np.linspace(0, self.params["t_max"], self.params["t_max"] + 1)
        return [int(t) for t in T]

    def reset_results(self):
        """Resets the results of the model"""
        self.solution_1 = []  # epi curve for seed country
        self.solution_2 = []  # epi curve for UK
        self.wd_timeseries = []  # has RNA entered wastewater timeseries
        self.bd_timeseries = (
            []
        )  # has respiatory swabbing detected any infectiond at border in destination country
        self.hd_timeseries = (
            []
        )  # has respiatory swabbing detected any infectiond in hostpitals in destination country
        self.cd_timeseries = (
            []
        )  # has respiatory swabbing detected any infectiond in community in destination country

        self.hs_timeseries = (
            []
        )  # has respiatory swabbing detected any infectiond in hostpitals in source country
        self.cs_timeseries = (
            []
        )  # has respiatory swabbing detected any infectiond in community in source country

        self.IF_timeseries = (
            []
        )  # number of infectious passenger from seed country to UK

        self.wd_count = (
            []
        )  # culmulative count of number of wastewater tests done in destination country
        self.bd_count = (
            []
        )  # culmulative count of number of respiritory swabs done at border in destination country
        self.hd_count = (
            []
        )  # culmulative count of number of respiritory swabs done at border in hostpitals in destination country
        self.cd_count = (
            []
        )  # culmulative count of number of respiritory swabs done at border in community in destination country

        self.hs_count = (
            []
        )  # culmulative count of number of respiritory swabs done at border in hostpitals in source country
        self.cs_count = (
            []
        )  # culmulative count of number of respiritory swabs done at border in community in source country

    def test(self, I, phi):
        """Carries out hospital or community testing"""
        I_tested = np.random.binomial(I, phi)
        positive_tests = np.random.binomial(I_tested, self.params["eta"])
        if positive_tests > 0:
            return True

    def simulate(self):
        """Intergrates the model over the stated time steps"""

        # initialise yt as y0
        yt_1 = self.params["y1_0"]
        yt_2 = self.params["y2_0"]
        # loop over time steps
        for t in self.t_steps[:-1]:
            # get current solutions - returns [S(t+1), E(t+1), I(t+1), F(t+1), R(t+1), Ic(t+1)]

            sol_1 = odeint(
                ode,
                yt_1,
                self.t_steps[t : t + 2],  # need to simulate 2+ steps
                args=(
                    self.params["beta"],
                    self.params["alpha"],
                    self.params["gamma"],
                    self.params["delta"],
                    self.params["N1"],
                ),
            )[
                1
            ]  # get solution on second step

            sol_2 = odeint(
                ode,
                yt_2,
                self.t_steps[t : t + 2],
                args=(
                    self.params["beta"],
                    self.params["alpha"],
                    self.params["gamma"],
                    self.params["delta"],
                    self.params["N2"],
                ),
            )[1]

            # sometimes some very small -ve values here, maybe solver inprecision. Gives errors for np random sampling when initialising flights
            """
            if any(sol_1 < 0):
                print('-ve values found in sim, probably solver inprecision:', sol_1, self.params)
            if any(sol_2 < 0):
                print('-ve values found in sim, probably solver inprecision:', sol_2, self.params)
            """
            sol_1[sol_1 < 0] = 0
            sol_2[sol_2 < 0] = 0

            S1, E1, I1, F1, R1, Ic1 = sol_1  # number in each state in seed country
            (
                S2,
                E2,
                I2,
                F2,
                R2,
                Ic2,
            ) = sol_2  # number in each state in destination country

            # simulate detection using Flight class
            flight1 = Flight(
                [S1, E1, I1, F1, R1], phi=self.params["phi_bd"], **self.params
            )
            flight2 = Flight([S2, E2, I2, F2, R2], phi=0.0, **self.params)

            # testing in source country
            new_cases = (
                (Ic1 - self.solution_1[-1][-1]) if len(self.solution_1) > 0 else Ic1
            )  # find out which new infections result in hospitilisation (only use new infections to avoid repeatedly testing the same individual for hospitilisation)
            I_hosp = np.random.binomial(new_cases, self.params["ihr"])
            self.hs_timeseries.append(
                self.test(I_hosp, self.params["phi_hs"])
            )  # hospital screening
            self.cs_timeseries.append(
                self.test(I1, self.params["phi_cs"])
            )  # community screening

            # testing in destination country
            self.wd_timeseries.append(flight1.ww_detected)  # wastewater detection
            self.bd_timeseries.append(flight1.rs_detected)  # resp screening detection
            new_cases = (
                (Ic2 - self.solution_2[-1][-1]) if len(self.solution_2) > 0 else Ic2
            )  # find out which new infections result in hospitilisation (only use new infections to avoid repeatedly testing the same individual for hospitilisation)
            I_hosp = np.random.binomial(new_cases, self.params["ihr"])
            self.hd_timeseries.append(
                self.test(I_hosp, self.params["phi_hd"])
            )  # hospital screening
            self.cd_timeseries.append(
                self.test(I2, self.params["phi_cd"])
            )  # community screening

            self.IF_timeseries.append(
                flight1.nE + flight1.nI
            )  # number of infectious on board

            self.wd_count.append(
                (1 + self.wd_count[-1]) if len(self.wd_count) > 0 else 1
            )
            self.bd_count.append(
                int(self.params["n"] * self.params["phi_bd"] + self.bd_count[-1])
                if len(self.bd_count) > 0
                else int(self.params["n"] * self.params["phi_bd"])
            )
            n_beds_uk = 158000  # from statistica
            n_hosp_admissions = 43552 + 23557  # from NHS
            self.hd_count.append(
                int(n_hosp_admissions * self.params["phi_hd"] + self.hd_count[-1])
                if len(self.hd_count) > 0
                else int(n_hosp_admissions * self.params["phi_hd"])
            )
            self.cd_count.append(
                int(self.params["y2_0"][0] * self.params["phi_cd"] + self.cd_count[-1])
                if len(self.cd_count) > 0
                else int(self.params["y2_0"][0] * self.params["phi_cd"])
            )

            flight1_list = flight1.state_counts + [flight1.nI]  # adding nI to to Ic
            flight2_list = flight2.state_counts + [flight2.nI]  # adding nI to to Ic

            # Update solution at current time step with flight disease state totals
            yt_1 = sol_1 - flight1_list + flight2_list
            yt_2 = sol_2 + flight1_list - flight2_list
            S2, E2, I2, F2, R2, Ic2 = yt_2

            # the continuous simulation and discrete sampling of flights can
            # sometimes cause very small -ve pops for low population states
            yt_1[yt_1 < 0] = 0
            yt_2[yt_2 < 0] = 0

            self.solution_1.append(yt_1)
            self.solution_2.append(yt_2)
