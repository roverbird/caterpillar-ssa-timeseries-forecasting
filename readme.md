# Singular Spectrum Analysis for Time Series Forecasting

This repository contains a slightly updated version of the `pssa` repository, originally developed by [aj-cloete](https://github.com/aj-cloete/pssa) and broadly based on the research of [Nina Golyandina](https://scholar.google.it/citations?user=1druVRYAAAAJ&hl=it) from St. Petersburg, Russia. The original script is in `original` directory, and its updated version is in the `scripts` dir. The purpose of this repo is to have an up-to-date and user-friendly Python implementation of Singular Spectrum Analysis (SSA) for time series forecasting, so that anyone could easily experiment with their datasets and quickly see an SSA forecast for their data.

![Caterpillar SSA](https://github.com/roverbird/caterpillar-ssa-timeseries-forecasting/blob/main/data/Operophtera_brumata_caterpillar.jpg)

_The Caterpillar SSA (Гусеница) appeared independently in Russia and other countries in the 1960s and 1970s._

## What is Singular Spectrum Analysis (SSA)?

Singular Spectrum Analysis (SSA) is a non-parametric technique in machine learning used to analyze and forecast time series data. SSA decomposes a time series into a sum of interpretable components such as trends, oscillatory patterns, and noise. The "Caterpillar" method of SSA originated independently in Russia under the name “Гусеница” and in other countries the method is known as SSA, or Eigen Decomposition.

By decomposing a time series into a sum of components—such as trend, seasonal, and noise—SSA enables the identification of patterns and structures that might be obscured in the raw data. This decomposition is achieved through Singular Value Decomposition (SVD) applied to a Hankel matrix constructed from the time series. SSA's ability to model complex, non-linear time series dynamics makes it valuable for applications from economic forecasting to environmental monitoring, where understanding and predicting trends and cycles is needed.

This script for Singular Spectrum Analysis (SSA) forecasting offers several advantages over traditional time series models like ARIMA and [SARIMA](https://github.com/roverbird/time-series-forecasting-adriatic-tide/). Unlike the Seasonal Autoregressive Integrated Moving Average (SARIMA), which rely on linear assumptions and parameters to capture trends, seasonality, and noise, SSA decomposes the time series into additive components using matrix factorization techniques. We can view the contribution of each of the signals (corresponding to each singular value).

Non-parametric approach allows SSA to capture complex, non-linear patterns and dynamics that linear models might miss. SSA is particularly useful when dealing with irregular or noisy data, as it can decompose the time series into a series of components that can be individually analyzed and forecasted. SSA also provides a more flexible framework for handling data with varying periodicities and structures, which can be a limitation for ARIMA and SARIMA models that assume constant seasonality and trend components.

## SSA vs SARIMA

[SARIMA](https://github.com/roverbird/time-series-forecasting-adriatic-tide) and SSA are both valid methods for analyzing time series data with seasonal patterns, but they approach the problem differently, and are not directly comparable. SARIMA (Seasonal ARIMA, or Seasonal Autoregressive Integrated Moving Average) is specifically designed to handle seasonal effects by incorporating seasonal differencing and autoregressive terms into its model. In contrast, SSA offers a more advanced and flexible approach by decomposing the time series into additive components such as trends, seasonal patterns, and noise through matrix decomposition. While these methods are not directly comparable due to their distinct methodologies, they can be evaluated against each other in practical applications to determine which best captures the underlying patterns and performs more accurately for a given real-world time series. See some examples of such a comparison [here](https://www.researchgate.net/publication/348820189_Comparison_of_SSA_and_SARIMA_in_Forecasting_the_Rainfall_in_Sumatera_Barat/).

Please, check out [python implementation of SARIMA](https://github.com/roverbird/time-series-forecasting-adriatic-tide) with tidal data from Koper, Slovenia. 

## Historical Background

The ["Caterpillar"-SSA](https://en.wikipedia.org/wiki/Singular_spectrum_analysis) approach emerged in the 1980s and 1990s across various fields of both theoretical and applied science. SSA, or Singular Spectrum Analysis, is designed to solve a wide array of problems related to the study of one-dimensional time series. While there are extensions for two-dimensional time series and random fields, this implementation focuses solely on one-dimensional time series analysis and forecasting.

The method's foundation lies in representing a time series as a matrix, which is then decomposed into a sum of matrices using Singular Value Decomposition (SVD). Each matrix corresponds to an additive component of the original time series. Thus, the series is decomposed into components, with information about each component contained in singular values and vectors.

The "Caterpillar" (Гусеница) appeared independently in Russia and other countries in the 1960s and 1970s. The Saint Petersburg State University (then Leningrad State University) was instrumental in its development, with significant contributions from the Department of Statistical Modeling under the leadership of S.M. Ermakov. Nowadays, [Nina Golyandina](https://scholar.google.it/citations?user=1druVRYAAAAJ&hl=it) is knows for her extensive publications and research of the Caterpillar-SSA.

### Script Features

1. **Embedding**:
   - The script uses a Hankel matrix to embed the time series, which is consistent with SSA's general approach.
   - It calculates the embedding matrix with a specified dimension or one based on the suspected frequency.

2. **Decomposition**:
   - Singular Value Decomposition (SVD) is applied to the embedded matrix, extracting components and their contributions.
   - It calculates the rank and performs various matrix operations to decompose the time series.
   - Utilize SVD to extract significant patterns from the time series.
   - Decompose time series into trend, seasonal components, and noise.

3. **Reconstruction and Forecasting**:
   - The script provides methods to reconstruct components from the SVD and to forecast future values using these components.
   - Split / test functionality: Compare the forecasted values against actual values to assess model performance.

The script demonstrates core SSA functionality, including time series embedding, SVD decomposition, and reconstruction. It also includes diagonal averaging, which is a feature of the Caterpillar SSA approach.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `numpy`, `pandas`, `scipy`, `matplotlib`

You can install the required packages using the following command:

```bash
pip install numpy pandas scipy matplotlib
```

### Usage

1. Clone the repository:

```bash
git clone https://github.com/roverbird/caterpillar-ssa-timeseries-forecasting.git
```

2. Prepare your time series data as a CSV file with a timestamp column and a value column (input should contain these two columns); manutally adjust your timestamp format in the script (look for comments).

3. Run the script:

```bash
python ~/caterpillar-ssa-timeseries-forecasting/scripts/mySSAtest2.py
```

### Parameters

You can change some values in the scripts to get better results (or any results at all - try a different dataset if you are stuck with NaNs). So, notice this line of code:

```python
ssa.embed(embedding_dimension=36, suspected_frequency=12, verbose=True)
```

It calls the `embed` method of the `mySSA` class with specific values that you can tweak.

1. **`embedding_dimension=36`**:
   - **Purpose**: This specifies the window size for embedding the time series into a Hankel matrix. The embedding dimension is crucial because it determines the size of the matrix used for Singular Spectrum Analysis (SSA).
   - **How**: Setting `embedding_dimension` to 36 means that the time series will be transformed into a matrix with 36 rows (or columns, depending on how you define the Hankel matrix). The number of columns (or rows) will be adjusted based on the length of the time series.
   - **Why**: The choice of embedding dimension affects the resolution and effectiveness of SSA. A larger embedding dimension can capture more detailed patterns in the time series but may also introduce more noise or complexity. Conversely, a smaller dimension may oversimplify the time series, potentially missing important patterns. It’s essential to balance between capturing sufficient detail and avoiding overfitting. If the time series exhibits complex seasonal patterns or long-term trends, a larger embedding dimension might be more appropriate. Conversely, for simpler patterns, a smaller dimension could be sufficient and computationally more efficient.

2. **`suspected_frequency=12`**:
   - **Purpose**: This parameter adjusts the embedding dimension to be divisible by the suspected frequency of the time series. Suspected frequency usually refers to the periodicity of the data (e.g., monthly data with a suspected yearly cycle would have a frequency of 12).
   - **How**: When `suspected_frequency` is set to 12, the `embedding_dimension` will be adjusted so that it is a multiple of 12. This helps in capturing periodic components effectively.
   - **Why**: Aligning the embedding dimension with the suspected frequency ensures that the embedding matrix respects the periodicity of the time series. This can enhance the model’s ability to capture and reconstruct seasonal or cyclic patterns accurately. Accurate frequency detection helps in capturing periodic patterns effectively. If your time series has different periodic components, adjusting this parameter accordingly ensures that these components are well-represented in the analysis.

## Outputs and Results

- **Plots**:
  - Original time series are in `time_series_plot.png`
  - Singular value contribution: shows the contribution of each of the signals, corresponding to each singular value, in plot `singular_contributions_plot.png`
  - Reconstructed components: now that we have the signal components we loop over that range and look at each one individually, output to plots `reconstruction_plot{N}.png`
  - Forecast: forecasted vs. original time series is in `forecast_plot.png` (set the length of forecast in `steps_ahead`)
  - Split and test: comparison of forecast vs. actual data is in `forecast_vs_actual.png`


![Plot example 1](https://github.com/roverbird/caterpillar-ssa-timeseries-forecasting/blob/main/data/time_series_plot.png)
_Original tidal data from Koper, Slovenia_

![Plot example 2](https://github.com/roverbird/caterpillar-ssa-timeseries-forecasting/blob/main/data/singular_contributions_plot.png)
_Singular value contribution shows the contribution of each of the signals_

![Plot example 3](https://github.com/roverbird/caterpillar-ssa-timeseries-forecasting/blob/main/data/forecast_vs_actual.png)
_Split and test results comparing predictions with historic data, the plot also shows absolute error in prediction_

![Plot example 4](https://github.com/roverbird/caterpillar-ssa-timeseries-forecasting/blob/main/data/forecast_plot.png)
_Forecast plot_

- **Data**:
  - DataFrames of the reconstructed components and forecasts are additionally output when running the `mySSAtest2.py` script.

### Explanation of Outputs and Results

Running the script with `TideHourly.csv` dataset (30 days of Adriatic sea tidal data from Koper, Slovenia) will give output which includes plots and some results. Let's take a look at them.

Let's set the values in the script as follows: 

`ssa.embed(embedding_dimension=36, suspected_frequency=12, verbose=True)`

Running the script with the Koper tide level dataset will give results below, and here are some explanations.

#### 1. **Embedding Summary**

```
----------------------------------------
EMBEDDING SUMMARY:
Embedding dimension	:  36
Trajectory dimensions	: (36, 1116)
Complete dimension	: (36, 1116)
Missing dimension     	: (36, 0)
----------------------------------------
```

- **Embedding Dimension**: The time series was embedded into a Hankel matrix with 36 rows. This means each window of 36 time points was used to form columns of the matrix.

- **Trajectory Dimensions**: The dimensions of the trajectory matrix after embedding are `(36, 1116)`. This means the matrix has 36 rows and 1116 columns.

- **Complete Dimension**: Indicates that there are no missing values in the matrix. The dimensions remain `(36, 1116)` for the complete matrix.

- **Missing Dimension**: There are no missing values in the embedded data, hence the missing dimension is `(36, 0)`.

#### 2. **Decomposition Summary**

```
----------------------------------------
DECOMPOSITION SUMMARY:
Rank of trajectory		: 36
Dimension of projection space	: 5
Characteristic of projection	: 0.9999
----------------------------------------
```

- **Rank of Trajectory**: The rank of the trajectory matrix is 36. This suggests that the trajectory matrix has full rank, meaning all 36 dimensions contribute to the reconstruction.

- **Dimension of Projection Space**: The SSA method identified 5 significant components (singular values) that capture the essential structure of the data. These components are used for forecasting.

- **Characteristic of Projection**: The characteristic of the projection is 0.9999, which indicates that the 5 components capture nearly all of the variance in the time series. This high value suggests a good fit of the components to the data.

#### 3. **Forecast vs. Actual Data**

```
                     Actual    Forecast
timestamp                              
2024-08-18 01:00:00     209  209.525328
2024-08-18 01:30:00     198  197.690202
2024-08-18 02:00:00     186  187.299544
2024-08-18 02:30:00     172  179.201117
2024-08-18 03:00:00     165  173.499721
...                     ...         ...
2024-08-23 22:30:00     241  241.287378
2024-08-23 23:00:00     248  246.540306
2024-08-23 23:30:00     254  250.547880
2024-08-24 00:00:00     257  252.968838
2024-08-24 00:30:00     258  253.572991

[288 rows x 2 columns]
```

- **Actual**: The actual tidal values, in mm, recorded in the `TideHourly.csv` file.

- **Forecast**: The predicted tidal values generated by the SSA model for the same timestamps.

#### Example Explanation

- The **forecast values** closely follow the **actual values**, indicating that the SSA model has done a good job of capturing the tidal patterns and predicting future values.
  
- **Accuracy**: You can assess the accuracy of the forecast by comparing the `Actual` and `Forecast` columns. Minor differences between these values are expected due to inherent variability and the forecasting method’s limitations.

- **Plot Interpretation**: If you view the plots generated by the script, you would typically see how well the forecast matches the actual time series. Ideally, the forecast should closely track the actual values, indicating that the SSA model effectively learned and extrapolated from historical data.

The results show that the SSA implementation successfully embedded the time series, decomposed it into components, and produced forecasts that align well with the actual observed values. The high characteristic of projection suggests that the model has captured most of the time series' variability with a small number of components.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Note: This repository is an educational and experimental project. Use it at your own risk.*
