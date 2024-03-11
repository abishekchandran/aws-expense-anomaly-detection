# AWS Expense Anomaly Detector

## Overview
This repository contains a machine learning project for detecting anomalies in daily expenses on the AWS (Amazon Web Services) platform. The project utilizes synthetic data generated for training and employs the Prophet time series model to analyze various periodic, holiday, and weekday variations in expenses. Anomalies are detected by comparing the predicted expenses with the actual expenses.

## Features
- Synthetic data generation for training the anomaly detection model.
- Implementation of Prophet time series model to analyze and predict daily expenses.
- Comparison of predicted expenses with actual expenses to detect anomalies.

## Requirements
- Python 3.11
- Jupyter Notebook (optional for viewing and running the project)
- Required Python libraries: pandas, numpy, matplotlib, seaborn, fbprophet

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/username/aws-expense-anomaly-detector.git
    ```
2. Install the required Python libraries:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Generate synthetic data: 
    - Run the `expense_data_generator.ipynb` notebook to generate synthetic data for training into `aws_expense.csv`.
2. Detect anomalies: 
    - Run the `deployment_script.py` notebook to detect anomalies in the AWS daily expenses using the trained model.
3. Furthermore, execute the `Visualise_Seasonal_Trends.ipynb` notebook to analyze the visual representations generated by the Prophet model for various seasonal datasets.

## Contributors
- [Abishek Chandran](https://github.com/abishekchandran)

## Acknowledgements
- [Facebook Prophet](https://facebook.github.io/prophet/) - For providing the time series forecasting model.

Feel free to contribute, report issues, or suggest improvements by opening an issue or pull request. Happy anomaly detection! 🚀
