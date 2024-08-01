
# DECO3801 - WinDash

## Innovation Energy Monitoring

For project details see:

https://studio3build.uqcloud.net/project/29S

### Summary

The aim of this project is to deliver a solution to users that translates real-time information from a meter attached to their household to an easy to digest report that enables users to receive interactive alerts for pre-defined usage limits and to be able to utilize AI/ML to make predictions on future usage to help users plan for reducing power consumption. Our app aims to achieve this through delivering a clear and precise application that updates periodically to inform users of their real-time power usage. To simulate the application working we will provide a simulated meter that can be replaced with a real alternative if this was to be launched. As well as an API to calculate the machine learning predictions, that will be connected directly to the application, to help users plan out their energy consumption.

---

## Components

### Simulated Meter

To simulate what an actual implementation of a meter connected to the user's household would do, we will create a mockup representation of how a meter will operate to showcase the functionality of real-time data processing.

An actual meter that we are trying to simulate would be connected to the switchboard of any residential or industrial building and would send the real-time energy consumption data to a database to store the information. This meter will handle all the logic of saving the most recent time-series data to the database where it will be stored in 2 different tables, one table representing the current day's power consumption on a per-minute basis, and the other representing a summary of all previous day records so not all minute-wise data needs to be saved, more information on the structure will be provided in the database component explanation below. This means at the end of the day the meter will calculate the summary of that day to save to the overall database table and reset the current day's database table to start fresh again.

The virtual meter provides this functionality for showcasing the functionality of the other components within the project and a multitude of other debugging functionality to help with the development of the project as a whole.

### Website / Application

The website implementation is the main user interface for all interactions. It handles the retrieval of information from the database, the visualization of the data in a organized format for the user, the updating of information from the real time data being added to the database, the communication with the python API to get results of future consumption predictions, and the alerting of users when the consumption gets close to the limit set by them. 

### Python API

The Python API is ran on a separate machine and provides all the processing for all forecasting predictions. The API incorporates the use of AI/ML models that are pre-trained on previous data to make predictions on what the future energy consumption would be given the current usage rate. This API is needed since the processing of model outputs would be beyond the scope on what the website processing can handle, and so we can save pre-trained models in a separate location away from user access.

### Database

The database is where all the meter data will be stored to provide data to the website. The data will be categorized in the following way; there will be a main collection for `overallData` and `dailyData`, each collection will map a `meterId` to each meter's respective individual data collection. For the `dailyData` collection, the data stored will be an array with each entry representing a map of; date-time, voltage, intensity, and potentially sub-meters. For the `overallData` collection, the overall data will be stored in an array of summarized daily information, that way we don't have to save every time increment of every minute in our database, the summarized maps stored in the array in the database will include fields; date, average intensity, total consumption, minimum intensity, and maximum intensity (subject to change over the course of the project).

To implement this solution we will use a firebase database to handle all storing, reading and writing of data within the project.
