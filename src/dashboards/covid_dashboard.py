import streamlit as st
import pandas as pd
import os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


countries = []

data_dir = os.path.join(
    "..", "..", "models", "simulated_data", "230315_disease_severity_scan"
)

for file in os.listdir(data_dir):

    if file[0] != ".":  # avoid hidden files
        country = file[:-4]  # assumes country.csv
        countries.append(country)


st.sidebar.markdown("## Testing parameters")
country = st.sidebar.selectbox("Choose country", countries)
phi_bd = st.sidebar.slider("Rate of border testing (UK)", 0.1, 0.5, step=0.1)
phi_hd = st.sidebar.slider("Rate of hospital testing (UK)", 0.1, 0.5, step=0.1)
com_tests = st.sidebar.selectbox(
    "Number of community tests per week (UK)", [10000, 25000, 50000, 100000, 250000]
)
phi_cd = com_tests / 7 / 67e6


phi_hs = st.sidebar.slider("Rate of hospital testing (source)", 0.1, 0.5, step=0.1)
phi_cs = st.sidebar.selectbox(
    "Rate of community tests per week (source)",
    [
        2.1321961620469086e-05,
        5.330490405117271e-05,
        0.00010660980810234542,
        0.00021321961620469085,
        0.0005330490405117272,
    ],
)  # same as the U


st.sidebar.markdown("## Disease parameters")
R0 = st.sidebar.slider("R0", 2, 5, step=1)
ihr = st.sidebar.slider("ihr", 0.05, 0.25, step=0.05)

st.sidebar.markdown("## Select plots")
incursion = st.sidebar.checkbox("First incursion into UK", key="i", value=True)
ww_detect = st.sidebar.checkbox("Wastewater detection (UK)", key="ww", value=True)
bd_detect = st.sidebar.checkbox("Border swab detection (UK)", key="b", value=True)
hd_detect = st.sidebar.checkbox("Hospital swab detection (UK)", key="hd", value=True)
cd_detect = st.sidebar.checkbox("Community swab detection (UK)", key="sd", value=True)
hs_detect = st.sidebar.checkbox(
    "Hospital swab detection (source)", key="hs", value=True
)
cs_detect = st.sidebar.checkbox(
    "Community swab detection (source)", key="ss", value=True
)


df = pd.read_csv(os.path.join(data_dir, country + ".csv"))


X = np.linspace(0, 200, 200)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
if incursion:
    # PLOT RESULTS
    data = df.loc[(df.R0 == R0) & (df.ihr == ihr)]
    t_seed = data["seed_t"]
    sq05, sq25, sq50, sq75, sq95 = np.quantile(t_seed, [0.05, 0.25, 0.5, 0.75, 0.95])
    # seed kde
    seed_kde = gaussian_kde(t_seed)
    seed_kde.covariance_factor = lambda: 0.25
    seed_kde._compute_covariance()
    ax.plot(
        X, seed_kde(X), color="k", ls="-", alpha=0.5, label="First incursion into UK"
    )
    ax.fill_between(X, seed_kde(X), color="k", alpha=0.2)
    ax.axvline(sq50, color="k", ls="--", lw=3)

if ww_detect:
    data = df.loc[(df.R0 == R0) & (df.ihr == ihr)]
    t_detect = data["ww_time"]
    wq05, wq25, wq50, wq75, wq95 = np.quantile(t_detect, [0.05, 0.25, 0.5, 0.75, 0.95])

    # detect kde
    detect_kde = gaussian_kde(t_detect)
    detect_kde.covariance_factor = lambda: 0.25
    detect_kde._compute_covariance()

    ax.plot(
        X, detect_kde(X), color="C0", ls="-", alpha=0.5, label="Wastewater detection"
    )
    ax.fill_between(X, detect_kde(X), color="C0", alpha=0.2)
    ax.axvline(wq50, color="C0", ls="--", lw=3)


phi_names = ["phi_bd", "phi_hd", "phi_cd", "phi_hs", "phi_cs"]
t_names = ["bd_time", "hd_time", "cd_time", "hs_time", "cs_time"]
phis = [phi_bd, phi_hd, phi_cd, phi_hs, phi_cs]
plot_flags = [bd_detect, hd_detect, cd_detect, hs_detect, cs_detect]
labels = [
    "Border swab (UK)",
    "Hospital swab (UK)",
    "Community swab (UK)",
    "Hospital swab (source)",
    "Community swab (source)",
]

for i in range(len(phis)):

    phi_name, t_name, phi, plot_flag = phi_names[i], t_names[i], phis[i], plot_flags[i]

    if plot_flag:
        data = df.loc[np.isclose(df[phi_name], phi) & (df.R0 == R0) & (df.ihr == ihr)]
        t_screen = data[t_name]
        bq05, bq25, bq50, bq75, bq95 = np.quantile(
            t_screen, [0.05, 0.25, 0.5, 0.75, 0.95]
        )
        screen_b_kde = gaussian_kde(t_screen)
        screen_b_kde.covariance_factor = lambda: 0.25
        screen_b_kde._compute_covariance()

        ax.plot(
            X,
            screen_b_kde(X),
            color="C" + str(i + 1),
            ls="-",
            alpha=0.5,
            label=labels[i],
        )
        ax.fill_between(X, screen_b_kde(X), color="C" + str(i + 1), alpha=0.2)
        ax.axvline(bq50, color="C" + str(i + 1), ls="--", lw=3)


ax.set_ylabel("Density")
ax.set_xlabel("Days since index case in seed country")
ax.legend()

st.pyplot(fig)
