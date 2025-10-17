# Laboratory 3 homework

Homework is the preparation of NYC taxi data for demand forecasting, implemented as regression-based
time series forecasting. The goal is to predict the daily number of taxi rides in the future, based
on historical features. This requires building a full data engineering pipeline: reading, cleaning,
transforming, and extracting features.

Instructions:
- use a single Jupyter Notebook, properly formatted and with text cells for readability
- if you need any additional libraries, use [uv commands](https://docs.astral.sh/uv/getting-started/features/)
- when properly implemented with lazy execution, everything should run smoothly, but in critical
  cases you can reduce the data to latest months (note that in the notebook)
- use Polars exclusively, except for just final data extraction to Pandas or plotting
- you can use any software you want for visualization, but Pandas + Matplotlib/Seaborn/Plotly are recommended
- all plots should have proper formatting and title, and also axis descriptions and legend if necessary

- in lab notebook, the entire 2024 data was downloaded, which will be used here
- you can use older  data, if you want and your hardware allows it
- using lazy mode for reading and processing (as much as possible) is highly recommended
- when needed, make additional DataFrames / partial calculations, but the final pipeline should
  be written in one place and use lazy execution as much as possible
- use [data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) as needed
- in descriptions below, we assume money-related columns to be: `fare_amount`, `extra`, `mta_tax`, `tip_amount`,
   `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `Airport_fee`

1. Data reading
   - load all 2024 months of taxi rides
   - also load taxi zone lookup data
   - include only rides starting in 2024 and ending at most at 01.01.2025
   - optimize data types, particularly for integers and categorical strings
2. Data cleaning and filtering
   - fill NULL values in `passengers_count` with 1
   - remove rides with zero passengers
   - if a ride has over 6 passengers, replace the value with 6
   - remove rides lasting over 2 hours
   - in all money-related columns, replace values with absolute value to fix negative amounts
   - remove rides with over 1000 dollars in any money-related column
   - remove rows with values of `RatecodeID` or `VendorID` missing or outside those defined in the data dictionary
3. Data transformation
   - combine payment type values for `"No charge"`, `"Dispute"`, `"Unknown"` and `"Voided trip"` into one type,
     so we have only credit card, cash, or other
   - replace `PULocationID` and `DOLocationID` with borough names by joining taxi zone lookup and removing
     unnecessary columns
   - add boolean variable `is_airport_ride`, true if there is non-zero airport fee
   - add boolean variable `is_rush_hour`, rush hours are defined as 6:30–9:30 and 15:30-20:00 during weekdays
4. Feature extraction
   - apply dummy encoding to features:
     - payment type
     - pickup borough
     - dropoff borough
   - add integer variables for counting daily events:
     - total number of rides (target variable)
     - number of airport rides
     - number of rush hour rides
   - add features aggregating daily rides information:
     - average fare amount
     - median distance
     - sum of total amounts
     - total amount paid by card, cash, and other
     - total congestion surcharge
     - total number of passengers
   - add time features:
     - `quarter`
     - `month`
     - `day_of_month`
     - `day_of_week`
     - `is_weekend` (boolean)
   - add column `date`, indicating day with given features
   - make sure to exclude other columns, unnecessary for machine learning, e.g. IDs, timestamps, unused financial information
   - properly name all columns
   - save results as `dataset.parquet` file
5. Data analysis
   - print shape
   - show top rows
   - describe statistics
   - print schema
   - plot target variable (daily number of rides):
     - histogram of values
     - line plot, date vs number of rides (remember to sort by date)

Grading:
- section 1: 1 point
- section 2: 1.5 points
- section 3: 1.5 points
- section 4: 5 points
- section 5: 1 point
