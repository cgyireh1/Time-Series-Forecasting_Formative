# Time-Series-Forecasting_Formative

# Air Quality Forecasting with RNN (LSTM)

This project focuses on forecasting PM2.5 air quality levels in Beijing using Recurrent Neural Network [RNN](#RNN), Long Short Term Memory [LSTM](#LSTM). The notebook walks you through data exploration, preprocessing, model building, training, evaluation, and generating submission files. Air quality is a critical environmental and public health issue. 

---

## Overview

In this project, I built a forecasting model to predict PM2.5 concentrations using historical data from Beijing. I employed a multi-layer LSTM network—with a Bidirectional LSTM layer—to capture the temporal dependencies in the data. 

This model was my best pick out of 17 experiments I did with different parameters. The Process:
- **Data Exploration & Cleaning:** Checking for missing values, handling outliers, and visualizing data distributions.
- **Feature Engineering:** Creating additional features (hour, day, month, and weekday) from the datetime column.
- **Data Normalization:** Standardizing the features to improve model performance.
- **Model Building:** Implementing a deep learning model with LSTM layers, including a Bidirectional LSTM.
- **Training & Evaluation:** Splitting the data into training and validation sets, training the model, and evaluating it using MSE and RMSE.
- **Submission:** Generating a predictions file of the outputs.


---

## Dataset

- **Train Data:** [train.csv](#train.csv) contains 30,676 rows and 12 columns, including meteorological variables, datetime, and the target variable pm2.5.
- **Test Data:** [test.csv](#test.csv) contains 13,148 rows and 11 columns. The test set does not include the pm2.5 values.

**Note:** Ensure that these CSV files are located in your Google Drive or your local directory in the correct folder as referenced in the notebook.

---

## Environment and Prerequisites

- **Python 3.7+**
- **Libraries:**
  - [pandas](#pandas)
  - [numpy](#numpy)
  - [seaborn](#seaborn)
  - [matplotlib](#matplotlib)
  - [scikit-learn](#scikit-learn)
  - [tensorflow](#tensorflow)
- **Google Colab:** Recommended for easy GPU support and for mounting Google Drive to access the dataset.

---

## Project Structure

```
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── Air_Quality_Forecasting_Best_model.ipynb
├── outputs/
│   ├── subm_fixed_1.csv
│   ├── subm_fixed_2.csv
│   ├── subm_fixed_3.csv
│   ├── subm_fixed_4.csv
│   ├── subm_fixed_5.csv
│   ├── subm_fixed_6.csv
│   ├── subm_fixed_7.csv
│   ├── subm_fixed_8.csv
│   ├── subm_fixed_9.csv
│   ├── subm_fixed_10.csv
│   ├── subm_fixed_11.csv
│   ├── subm_fixed_12.csv
│   ├── subm_fixed_13.csv
│   ├── subm_fixed_14.csv
│   ├── subm_fixed_15.csv
│   ├── subm_fixed_16.csv
│   └── subm_fixed_17.csv
│
│
└── README.md
```

- **data/**: Contains the training and testing datasets.
- **notebooks/**: Contains the Jupyter Notebook with the complete workflow.
- **outputs/**: Contains the submission files generated after predictions.
- **README.md**: This file.

---

## Instructions To Run The Code

1. **Clone the Repository or Download the Notebook:**

   ```bash
   git clone https://github.com/yourusername/air-quality-forecasting.git
   cd air-quality-forecasting
   ```

2. **Open the Notebook in Google Colab:**

   - Upload or open the notebook file ``Air_Quality_Forecasting_Best_model.ipynb`` in [Google Colab](https://colab.research.google.com/).

3. **Mount Google Drive:**

   In the first cell, the notebook mounts your Google Drive. Ensure you have the dataset files [train.csv](#train.csv) and [test.csv](#test.csv) in the correct folder (as specified in the code):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Update File Paths (if necessary):**

   Check the file paths when loading data. For example:

   ```python
   train = pd.read_csv('/content/drive/MyDrive/Kaggle_competition_ML/air_quality_forcasting/train.csv')
   test = pd.read_csv('/content/drive/MyDrive/Kaggle_competition_ML/air_quality_forcasting/test.csv')
   ```

   Modify these paths based on your Google Drive folder structure.

5. **Run Through the Notebook Cells:**

   - **Data Exploration & Cleaning:** [First_15_cells] load the data, check missing values, and handle them by filling with the mean.
   - **Exploratory Data Analysis (EDA):** [Next_6_cells] create line plots, box plots, histograms, and a correlation heatmap.
   - **Feature Engineering:** [Next_7_cells] generate datetime features and prepare the data for modeling.
   - **Data Scaling and Splitting:** Standardize features using [StandardScaler](#StandardScaler) and reshape the data for LSTM.
   - **Model Building and Training:**[Next_3_cells] build the LSTM model with multiple layers and train it using early stopping and learning rate scheduling.
   - **Evaluation and Visualization:**[Next_2_cells] evaluate the model performance using MSE and RMSE, and visualize predictions.
   - **Submission:** [final_cells] shows how to generate predictions on the test set and save them to a CSV file. Uncomment and adjust the code if you wish to produce the submission file.

6. **Experiment with the Model:**

   Modify parameters such as:
   - The number and size of LSTM layers.
   - Activation functions and dropout rates.
   - Optimizers (SGD, Adam, ...) and learning rates.
   - Batch size and number of epochs.
   
   These changes can be experimented with to improve performance or discover trends or find better fine tuning parameters.

---

## Model Architecture
- A **Bidirectional LSTM** layer (128 units) with ReLU activation and L2 regularization.
- Two additional **LSTM layers** (64 units with 'return_sequences=[True](#True)' and 32 units respectively) with ReLU activation.
- A **Dropout** layer (0.3 dropout rate) to prevent overfitting.
- A **Dense** layer (1 unit) that outputs the predicted PM2.5 value.

The model is compiled using the Adam optimizer with a learning rate of 0.0005 and optimized for the mean squared error (MSE) loss function.

---

## Results and Visualizations

Throughout the notebook, various plots are generated to aid understanding:
- **Line Plot:** Shows PM2.5 concentration over time.
- **Box Plot & Histograms:** Display distribution of PM2.5 and other features.
- **Heatmap:** Visualizes the correlation between features.
- **Training Loss Plot:** Compares training and validation loss over epochs.
- **Prediction vs. Actual:** Plots to compare model predictions with actual PM2.5 values on both training and validation sets.

Performance metrics, MSE and RMSE are printed after training.

---

By following the instructions above and running the notebook cells in Google Colab (or your preferred Python environment), you should be able to reproduce the results.

