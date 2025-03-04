# README

Package developed by Neythen Treloar during the Faculty fellowship. The aim of this project was to develop a model incorporating three previously developed models at the UKHSA to predict the time to detection of a new disease under different testing assumptions. The main deliverables of this project are the new model and the dashboard to display simulated results.

## Usage 
To run the dashboard excecute `streamlit run <dashboard>` in the `source/dashboards` directory. To run the code used to generate the synthetic data run the relevant Python file in the `src/models` directory. Let me know if you have any problems. 

## Main assumptions
- Because of data access issues this is done without the full international flights model, only direct flights are currently considered
- Because flights are modelled as one transfer of people per day (effectively one big flight) the limit of detection of feacal tests was set to 0. This is jusitfied by the LOD being orders of magnitude lower that the feasible mimimum
- As the previous flihgt model was modelling the whole of the UK but the age stratifiation assumptions were made for just England I scaled up all the age groups so that the total population was that of the UK
- I assumed the age stratification of the source countries were proportionally the same as in England. Because we assumped homogenous mixing and no age dependance on getting on flights this doesnt affect anything currently, but might if the model is further developed
- Avain flu parameter distribution assumptions: 


## Dependencies
Standard datascience libraries. `streamlit` is required to run the dashboards.



Package directory structure:
```
├── README.md
├── data
│   ├── processed
│   │   └── country_flights.csv                <- the dataset used to model flights
│   └── raw                                    <- the raw datasets used to generate country_flights.csv
├── models
│   ├── model.py                               <- model without age stratification used for covid
│   ├── model_age_stratified.py                <- model with age stratification used for avain flu
│   └── simulated_data                         <- directory containing simualted data generated by the models
├── notebooks                                  <- rough EDA notebooks
└── src
    ├── dashboards
    │   ├── avian_flu_dashboard.py             <- dashboard to display the avain flu results
    │   └── covid_dashboard.py                 <- dashboard to display the covid results
    ├── data
    │   └── make_flight_dataset.py             <- processes raw flight data and add contry populations
    ├── models
    │   ├── avian_flu_scan.py                  <- generates the avain flu simulated data
    │   └── covid_scan.py                      <- generates the covid simulated data
    └── visualisation                          <- some rough notebooks for plotting
```
