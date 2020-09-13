# slocum glider data tools
basic scripts for processing and visualizing slocum glider data in python.
- `dbd2asc.py`: converts raw binary data (.dbd, .ebd, .cac) into ascii arrays (.dat) and matlab files containing header data (.m). This script requires utilities for converting binary data provided by Teledyne Webb and available [here](https://marine.rutgers.edu/~kerfoot/slocum/data/readme/wrc_exe/windoze-bin/).
- `readascs.py`: converts files created by `dbd2asc.py` (.dat, .m) to a single .csv file with header and only keeping selected parameters.
- `cGliderData.py`: a class for reading in .csv produced by `readascs.py`, with methods for subsetting data, converting timezones, plotting, etc.
