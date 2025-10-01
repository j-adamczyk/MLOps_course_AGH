# Laboratory 2 homework

Homework will be a deeper analysis of Rijden de Treinen data, similar to real-world OLAP queries and analytics
use cases. Such steps are typical first steps in MLOps tasks and data engineering.

Instructions:
- use a single Jupyter Notebook, properly formatted and with text cells for readability
- if necessary, run the commands in the CLI and note them in Jupyter Notebook, or you can run them
[directly in Jupyter Notebook](https://saturncloud.io/blog/how-to-execute-terminal-commands-in-jupyter-notebook/)
- if you need any additional libraries, use [uv commands](https://docs.astral.sh/uv/getting-started/features/)
- if you run into any memory problems, you can reduce the number of data to newer years, but note that
  in the notebook
- use DuckDB as your primary analytics database/engine, and Postgres as indicated by instructions
below
- if necessary, export data to Pandas or Python and process it further there (e.g. for plotting)
- work on fresh Postgres and DuckDB instances, removing any volumes and files created during lab
- you can use any software you want for visualization, but Pandas + Matplotlib/Seaborn/Plotly are recommended
- all plots should have proper formatting and title, and also axis descriptions and legend if necessary

1. Download data:
   - [inter-station distances in 2022](https://www.rijdendetreinen.nl/en/open-data/station-distances)
   - [railway stations in 2023](https://www.rijdendetreinen.nl/en/open-data/stations)
   - [train disruptions between 2011 and 2023](https://www.rijdendetreinen.nl/en/open-data/disruptions)
   - [train services between 2019 and 2025](https://www.rijdendetreinen.nl/en/open-data/train-archive)
2. Put stations data into `stations` table in DuckDB. This changes rarely, so we treat it as a almost constant file.
3. Based on [DuckDB tutorial](https://duckdb.org/2024/05/31/analyzing-railway-traffic-in-the-netherlands.html#largest-distance-between-train-stations-in-the-netherlands),
   create tables `distances` and `distances_long`. We treat this similarly to `stations` table.
4. Put train disruptions into `disruptions` table in the Postgres database. We expect this data to change
   regularly, and thus treat it as a typical OLTP table.
5. Transform train services CSV files into a single Parquet file. Make table `services` from it. We treat
   this as a big data batch input, created rarely but regularly for analytics purposes.
6. In all following questions, use tables `services`, `distances_long`, `disruptions` and `stations` as necessary.
   Remember that each row represents single ride between stations, while the whole end-to-end train service has the
   same value in `Service:RDT-ID` column. Note that starting station for a service has NULL value in `Stop:Arrival time`
   column, while the end station has NULL value in `Stop:Departure time` column.
7. Make queries to answer the following questions:
   1. How many trains departed from Amsterdam Central station overall?
   2. Calculate the average arrival delay of different service types (`Service:Type`). Order results descending by average delay.
   3. What was the most common disruption cause in different years? [MODE function](https://duckdb.org/docs/stable/sql/functions/aggregates.html#modex) may be useful.
   4. How many trains started their overall service in any Amsterdam station?
   5. What fraction of services was run to final destinations outside the Netherlands?
   6. What is the largest distance between stations in the Netherlands (code `NL`)?
   7. Compare the average arrival delay between different train operators (`Service:Company`) on a bar plot. Sort them appropriately.
   8. How many services were disrupted in different years? Make a line plot.
   9. What fraction of all services were cancelled (`Service:Completely cancelled`) in different years? Make a line plot.
8. Currently, `services` table does not provide information about service lengths, neither between pairs of stations
   nor for the end-to-end service. Prepare this information:
   1. Note that each service has the same `Service:RDT-ID`, and stations can be ordered by `Stop:Departure time`,
      with the last one being NULL. Using [window functions](https://duckdb.org/docs/stable/sql/functions/window_functions.html),
      specifically [LAG() or LEAD()](https://duckdb.org/docs/stable/sql/functions/window_functions.html),
      you can get next row. [This example](https://stackoverflow.com/a/62584847/9472066) may also be useful.
   2. Create table `station_connections`, with columns `Service:RDT-ID`, `start_station_code` and `end_station_code`
      (pair of stations on a route), and `distance` between them. Note that you should deduplicate the data on station
      codes, so that every station pair appears only once. Create temporary tables, use a subquery, or any other
      similar techniques if necessary.
   3. What is the largest distance between a pair of stations?
   4. Plot a histogram of inter-station distances run by trains.

Grading:
- sections 1-5: 1.5 points
- section 7: 6 points
- section 8: 2.5 points
