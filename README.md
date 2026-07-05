# Analyzing Daily CO₂ Emissions Across Countries

Anomaly detection, structural break analysis, clustering, and LSTM forecasting on daily CO₂ emissions data (2019–2025).

**Course project** — Data Mining & Machine Learning, University of Naples Federico II (Oct 2025)
Group: Francesco Stellato, Daniele Cavuto, Parisa Izadiyar, Aashi Sudhanshu Mehrotra
Instructors: Giuseppe Longo & Roberta Siciliano

## Overview

This project analyzes daily CO₂ emissions across multiple countries, combining data mining and machine learning techniques to understand both descriptive and predictive patterns in global emissions.

After preprocessing and normalizing emissions by population and GDP, the analysis explores patterns through visualization, symbolic representation (SAX), and statistical decomposition. Anomaly detection surfaces rare temporal behaviors, while CUSUM structural break analysis confirms major disruptions in early 2020 tied to the COVID-19 pandemic. Clustering (K-Means and Hierarchical) groups countries by emission dynamics and magnitude. Finally, LSTM neural networks forecast short-term emission trends at both the global and country level.

## Dataset

- **Source:** [Carbon Monitor](https://carbonmonitor.org/) — near-real-time daily CO₂ emissions by country and sector
- **Size:** 204,540 observations across multiple countries and sectors, Jan 2019 – Aug 2025
- **Variables:** country, date, sector, value (MtCO₂/day)
- **Enrichment:** population and GDP data pulled from the World Bank Open Data API (`wbgapi`) to compute per-capita and per-GDP normalized emissions

## Methodology

1. **Data Preparation** — cleaning, missing data/duplicate checks, and normalization of emissions by population and GDP for fair cross-country comparison
2. **Data Visualization** — global and country-level emission trends, 30-day rolling averages to smooth seasonal noise
3. **Symbolic Aggregate approXimation (SAX)** — symbolic representation of time series for pattern discovery
4. **Anomaly Detection** — identifying rare temporal deviations in emission behavior
5. **Structural Break Detection (CUSUM)** — statistically confirming the COVID-19-driven disruption in early 2020
6. **Clustering Analysis** — K-Means and Hierarchical clustering, grouping countries by temporal evolution and by emissions magnitude
7. **LSTM Forecasting** — recurrent neural networks forecasting short-term emissions at the global level and for China, the United States, and Russia

## Results

- Clustering revealed meaningful country groupings — e.g., European nations sharing synchronized seasonal patterns, versus countries like China and the U.S. standing apart by emissions magnitude.
- CUSUM confirmed a clear structural break in emissions coinciding with the 2020 pandemic lockdowns.
- LSTM forecasts achieved consistently low error across regions:

| Region | RMSE (MtCO₂/day) | MAE (MtCO₂/day) | MAPE |
|---|---|---|---|
| China | 0.83 | 0.62 | 2.1% |
| United States | 0.70 | 0.54 | 4.0% |
| Russia | 0.15 | 0.11 | 3.0% |

The global and Chinese models showed near-complete overlap between predicted and actual emissions, while the U.S. and Russian models showed slightly larger deviations around sudden peaks but still delivered reliable short-term forecasts.

## Tools & Techniques

- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistics:** statsmodels (CUSUM, seasonal decomposition), SciPy
- **Machine Learning:** scikit-learn (K-Means, DBSCAN, PCA, hierarchical clustering)
- **Deep Learning:** TensorFlow/Keras (LSTM)
- **External data:** World Bank API via `wbgapi`, GeoPandas

## Repository Structure

```
├── CO2_emissions_analysis.ipynb   # Full analysis notebook
├── requirements.txt               # Python dependencies
└── README.md
```

## Running the Notebook

```bash
pip install -r requirements.txt
jupyter notebook CO2_emissions_analysis.ipynb
```

You'll need the Carbon Monitor global emissions CSV (`carbonmonitor-global_datas_*.csv`) placed in the working directory — download it from [carbonmonitor.org](https://carbonmonitor.org/).

## Future Work

- Incorporate exogenous variables (temperature, industrial output, mobility indices) to improve forecasting accuracy and interpretability
- Extend the LSTM forecasting horizon beyond 30 days
- Combine SAX symbolic methods with deep learning for earlier anomaly signal detection
