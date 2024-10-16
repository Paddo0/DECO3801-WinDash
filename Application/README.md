
# DECO3801 - WinDash Application

## Summary

This solution includes the implementation of the WinDash - Innovative Energy Monitoring app. The application within this solution is designed to act as an ambient user interface to deliver useful information and graphics about the energy consumption of their household. The user will have the ability to; set energy limits for their daily consumption to be notified when they will exceed this limit, have a AI/ML model show predictions of their expected daily / quarterly consumption, and view their current consumption statistics. To display this information to the user the website will be structured as seen below.

---

## Application Structure

### Ambient Slideshow

Navigating the back arrow or being inactive for long enough (60s) will lead users to the main ambient user interface which displays different mood-based images dependent on the amount of power that is currently being used. The criteria for 'high usage' is based on a multitude of their previous average usage.

### Dashboard / Home

This will be the landing page of the website which will give a very brief overview of the consumption data. This page serves to redirect the user to every other webpage within the application and to display interesting facts about their usage that day that draws comparisons to everyday objects to help users visualize their power usage.

### Daily Statistics

This page will give the user all the information visualization of the current day's consumption statistics including; a graph of todays current power usage history, usage statistics for the day (i.e. average, total, etc.), limit statistics from user-defined daily limit, and predicted usage information (using ML).

### Overall Statistics

This page will be very similar to the daily statistics page where every data visualization used in the daily page will also be used in this page, except with the data covering a summary of all days with data recorded.

### Settings

The settings page will offer a large range of functionality to the user. This is where the user will be able to set daily limits and monthly limits to be shown within the statistics pages. This is also where the user would be able to change the meter that they have selected, in case they have made a mistake or reset to the default limits recommended by the application.

### Info

This is a page to help the user with any questions they might have about the website and a guide on how to use the functionality within it.

---

## External Interactions

### Machine Learning API

To generate the predictions for the user for their daily / overall consumption prediction, the application will be using an API to calculate the results that will be output to the user. Since the processing resources is too great to calculate a model prediction within a website application, the website will get this information through sending a request with the data required for input to a python API that will return the results from the activation of a pre-trained model. More information for the specifics of the ML model can be found in the `PythonAPI` project.

### Database Data

To get the data to display to the user, the website queries a database using the meter id as the key to get the series data to display to the user. The application also periodically queries the database in increments of about 30 seconds to process the real-time data that will be received by the database by the virtual meter. Since react is used, all components that are dependent on the data changing automatically update when the data is refreshed.

---

## Build Instructions

To run the application on your local machine to view the application, ensure that Node.js is installed on your computer and run the following command(s) within your terminal within the file path that this file is in.

### `npm install`

This is required to be run once before initially starting the application to download all package dependencies for the project.

### `npm start`

This is the recommended way to run the application. \
Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

---

## Icons

All icons were sourced from the [Font Awesome](https://fontawesome.com/search?o=r&m=free) website from their range of free .svg icons available for public use. The icons are all stored within the `src/assets` folder to be used throughout the project.