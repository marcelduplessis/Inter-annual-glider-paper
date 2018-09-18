# READ ME

This document outlines the publication "Southern Ocean seasonal restratification delayed by submesoscale wind-front interactions"

Authors: Marcel du Plessis, Sebastiaan Swart, Isabelle Ansorge, Andrew F. Thompson, Amala Mahadevan

## Glider data

The glider data from this experiment was obtained by the Southern Ocean Carbon and Climate Observatory. To know more about the data, you can contact the lead author Marcel du Plessis at marceldavidduplessis@gmail.com

## Reading of the data and objective mapping

The coding language used in this paper is Python. To read the netcdf Seaglider data into Python, I use a toolkit for processing Seaglider basestation NetCDF files created by Luke Gregor ( https://github.com/luke-gregor/BuoyancyGliderUtils ). The raw glider data is then objectively mapped using the MATLAB toolbox (objmap) created by Scripps Oceanographic Institute ( http://mooring.ucsd.edu/ ). A comparison of the temperature at 10 m depth between the raw Seaglider profiles and objectively mapped data (2hr horizontal grid, 5m vertical grid) is shown below:

![Semi-variogram](https://github.com/marcelduplessis/Inter-annual-glider-paper/blob/master/sensativity_tests/Sens_ML_temp_S1.png)

Along track tracers of temperature and salinity reveal that the objective mapping technique is successfully able to capture the along-track submesoscale variability.

![Semi-variogram](https://github.com/marcelduplessis/Inter-annual-glider-paper/blob/master/sensativity_tests/Sens_temp_section_S1.pdf)