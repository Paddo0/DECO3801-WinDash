# DECO3801 - Virtual Meter

## Overview

The Virtual Meter Simulation is a Python-based project designed to simulate the operation of a virtual energy meter. It processes and inserts energy consumption data into a Firebase Firestore database, allowing users to simulate energy usage data at specific times and analyse consumption trends over time.

### Key Features:
- **Simulates Historical and Real-Time Data**: The virtual meter can be set up with historical data or start from a specified point in time.
- **Flexible Start Time**: Allows running the simulation from any specific point in time, including testing scenarios for the future or the past.
- **Data Extraction and Processing**: Extracts data from CSV files and processes it to generate daily summaries.
- **Integration with Firebase Firestore**: Stores energy consumption data, daily summaries, and historical records in a Firestore database.
- **Resumption Capabilities**: Can resume the simulation from the last recorded time, ensuring continuity in data processing.

## Project Structure

The project contains the following components:

- `main.py`: Entry point of the project, manages command-line input to run different commands and functionalities.
- `constants.py`: Stores project constants such as the meter ID, file paths, and other configuration variables.
- `helpers/`: Contains helper files that break down the project's functionality into manageable modules:
  - `CalculationHelper.py`: Handles calculations for daily summaries based on minute-level data.
  - `CsvDataHelper.py`: Manages data extraction from CSV files, including historical, future, and range-based data.
  - `DatabaseMutationHelper.py`: Provides functions for adding and clearing data in the Firestore database.
  - `DatabaseRetrieveHelper.py`: Retrieves specific data points from the Firestore database.
  - `VirtualMeterHelper.py`: Manages the setup, start, and running processes of the virtual meter.
- `data/`: Stores the required files to run the virtual meter.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Firebase Admin SDK credentials file (`.json`) for authenticating with the Firebase project.
- Firebase Firestore configured as the database backend.
- Required Python packages:
  - `firebase-admin`
  - `numpy`

### Installation

1. **Install the required Python packages**:
   ```bash
   pip install firebase_admin numpy
2. **Configure Database**
    - Add Firebase Admin SDK credentials JSON file to the data directory

3. **Add Data**
    - Retrieve data from the following link: [UCI Electric Power Consumption Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set/data)
    - Convert to a csv file (can do be changing the file extension from .txt to .csv)
    - Save csv file to data directory
    - Change dataFilepath in constants.py if required

### Virtual Meter Run Instructions
1. **Set Up the Database**:
   - Use the `SetupWithHistoricalData` or `SetupWithNoData` commands to initialize the database.
   - Change the `meterId` variable in `constants.py` to specify which virtual meter to use.

2. **Run the Virtual Meter**:
   - Use `setupWithNoData`, `StartFromNow`, or `Resume` functions to begin the virtual meter simulation.
   - Refer to the function documentation in `VirtualMeterHelper.py` for detailed descriptions of each function.

3. **Testing/Debugging**:
   - The commands provided in `main.py` can also be used for testing and debugging. Modify `command` in `main.py` to execute different functions.
   - Note: Only the setup and run functions are required for regular usage; other commands are meant for testing and debugging.
