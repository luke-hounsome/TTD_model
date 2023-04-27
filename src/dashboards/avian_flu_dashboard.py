import streamlit as st
import pandas as pd
import os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns


def get_countries(data_dir):
    countries = []

    for file in os.listdir(data_dir):

        if file[0] != ".":  # avoid hidden files
            country = file[:-4]  # assumes country.csv
            countries.append(country)
    return countries


data_dir = os.path.join("..", "..", "models", "simulated_data", "230320_avian_flu")
countries = get_countries(data_dir)


# Initialise settings toggles


st.sidebar.markdown("## Testing parameters")
country = st.sidebar.selectbox("Choose country", countries)
phi_bd = st.sidebar.slider("Rate of border testing (UK)", 0.1, 0.5, step=0.2)
phi_hd = st.sidebar.slider("Rate of hospital testing (UK)", 0.1, 0.5, step=0.2)
com_tests = st.sidebar.selectbox(
    "Number of community tests per week (UK)", [15000, 50000, 100000]
)
phi_cd = com_tests / 7 / 67e6


phi_hs = st.sidebar.slider("Rate of hospital testing (source)", 0.1, 0.5, step=0.2)
phi_cs = st.sidebar.selectbox(
    "Rate of community tests per week (source)",
    [3.198294243070362e-05, 0.00010660980810234542, 0.00021321961620469085],
)  # same as the U

st.sidebar.markdown("## Disease parameters")
R0 = st.sidebar.slider("R0", 1.2, 2.0, step=0.4)

ihr_scenarios = {
    "Mild flat": [1, 1, 1, 1],
    " Mild monotonic": [0.0041, 0.0409, 0.4091, 4.0913],
    "Mild young risk": [3.82, 0.38, 0.04, 3.82],
    "Severe flat": [5, 5, 5, 5],
    "Severe monotonic": [0.0205, 0.2046, 2.0457, 20.4566],
    "Severe young risk": [33.552, 3.355, 3.355, 3.355],
    "Severe flat": [10, 10, 10, 10],
    "Severe monotonic": [0.0409, 0.4091, 4.0913, 40.9133],
    "Severe young risk": [33.552, 3.355, 3.355, 3.355],
    "Extra severe flat": [10, 10, 10, 10],
    " Extra severe monotonic": [0.0409, 0.4091, 4.0913, 40.9133],
    "Extra severe young risk": [38.152, 3.815, 0.382, 38.152],
}
ihr_scenario = st.sidebar.selectbox("IHR scenario", ihr_scenarios.keys())


ihrs = np.array(ihr_scenarios[ihr_scenario]) / 100  # because IHRS are not percentages

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


# Plot the distribution results

st.markdown("## Time to detection")

df = pd.read_csv(os.path.join(data_dir, country + ".csv"))


X = np.linspace(0, 100, 200)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
if incursion:

    data = df.loc[
        (df.phi_bd == phi_bd)
        & (df.R0 == R0)
        & np.isclose(df.ihr_1, ihrs[0])
        & np.isclose(df.ihr_2, ihrs[1])
        & np.isclose(df.ihr_3, ihrs[2])
        & np.isclose(df.ihr_4, ihrs[3])
    ]
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
    data = df.loc[
        (df.phi_bd == phi_bd)
        & (df.R0 == R0)
        & np.isclose(df.ihr_1, ihrs[0])
        & np.isclose(df.ihr_2, ihrs[1])
        & np.isclose(df.ihr_3, ihrs[2])
        & np.isclose(df.ihr_4, ihrs[3])
    ]
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
        data = df.loc[
            np.isclose(df[phi_name], phi)
            & (df.R0 == R0)
            & np.isclose(df.ihr_1, ihrs[0])
            & np.isclose(df.ihr_2, ihrs[1])
            & np.isclose(df.ihr_3, ihrs[2])
            & np.isclose(df.ihr_4, ihrs[3])
        ]
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


# Plot the histogram results


def plot_hists(data, titles, colours=["C0", "C0", "C0", "C0"], xlabel="Number"):

    fig, axs = plt.subplots(
        4, 2, figsize=(12, 8), gridspec_kw={"height_ratios": (0.2, 0.8, 0.2, 0.8)}
    )

    for i, d in enumerate(data):
        ax_box = axs[i // 2 * 2, i % 2]
        ax_hist = axs[i // 2 * 2 + 1, i % 2]

        # ax_box.boxplot(d)
        # ax_hist.hist(d)
        sns.boxplot(d.values, ax=ax_box, orient="h", color=colours[i])
        sns.histplot(d.values, ax=ax_hist, kde=True, color=colours[i])

        ax_box.set(yticks=[], xticks=[])
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True)
        ax_box.set_title(titles[i])
        ax_hist.set_xlabel(xlabel)
    plt.tight_layout()

    st.pyplot(fig)


st.markdown("## Detection time relative to index case in the UK")


data = df.loc[
    (df.R0 == R0)
    & np.isclose(df.ihr_1, ihrs[0])
    & np.isclose(df.ihr_2, ihrs[1])
    & np.isclose(df.ihr_3, ihrs[2])
    & np.isclose(df.ihr_4, ihrs[3])
]

d = []

for i, k in enumerate(["ww_time", "bd_time", "hd_time", "cd_time"]):
    phi_name, phi = phi_names[i], phis[i]
    detect_time = data.loc[np.isclose(data[phi_name], phi)][k]
    seed_time = data.loc[np.isclose(data[phi_name], phi)]["seed_t"]
    d.append(detect_time - seed_time)
titles = [
    "Wastewater detection",
    "Border swab (UK)",
    "Hospital swab (UK)",
    "Community swab (UK)",
]

plot_hists(d, titles, colours=["C0", "C1", "C2", "C3"], xlabel="Time (days)")


st.markdown("## Total tests used at detection")

titles = [
    "n ww tests at detection",
    "n border tests at detection",
    "n hospital tests at detection",
    "n community tests at detection",
]


d = [
    data.loc[(data.phi_bd == phi_bd)]["ww_nd_tests"],
    data.loc[(data.phi_bd == phi_bd)]["bs_nd_tests"],
    data.loc[(data.phi_hd == phi_hd)]["hs_nd_tests"],
    data.loc[np.isclose(data.phi_cd, phi_cd)]["cs_nd_tests"],
]
print(df.phi_cd.unique())
print(phi_cd)
plot_hists(d, titles)


st.markdown("## Total hospitilisations")

titles = [
    "0-4 hospitalisations",
    "5-19 hospitalisations",
    "20-64 hospitalisations",
    "65+ hospitalisations",
]

d = [
    data.loc[(data.phi_bd == phi_bd)]["c_hosp_1"],
    data.loc[(data.phi_bd == phi_bd)]["c_hosp_2"],
    data.loc[(data.phi_bd == phi_bd)]["c_hosp_3"],
    data.loc[(data.phi_bd == phi_bd)]["c_hosp_4"],
]


plot_hists(d, titles)


st.markdown("## Total infections")

titles = ["0-4 infections", "5-19 infections", "20-64 infections", "65+ infections"]

d = [
    data.loc[(data.phi_bd == phi_bd)]["c_inf_1"],
    data.loc[(data.phi_bd == phi_bd)]["c_inf_2"],
    data.loc[(data.phi_bd == phi_bd)]["c_inf_3"],
    data.loc[(data.phi_bd == phi_bd)]["c_inf_4"],
]


plot_hists(d, titles)


st.markdown("## Summary tables for time to detection for given disease parameters")

data = df.loc[
    (df.R0 == R0)
    & np.isclose(df.ihr_1, ihrs[0])
    & np.isclose(df.ihr_2, ihrs[1])
    & np.isclose(df.ihr_3, ihrs[2])
    & np.isclose(df.ihr_4, ihrs[3])
]

for phi, key in zip(["phi_bd", "phi_hd", "phi_cd"], ["bd_time", "hd_time", "cd_time"]):
    dat = data.groupby(phi)[[key]]

    quantiles = dat.quantile(q=[0.05, 0.25, 0.5, 0.75, 0.95])
    quantiles = quantiles.rename_axis([phi, "quantile"]).reset_index()
    # quantiles.rename(columns={1 :'quantile', 2:'time'}, inplace=True )
    quantiles = quantiles.pivot(index=phi, columns=["quantile"])

    st.dataframe(quantiles)
