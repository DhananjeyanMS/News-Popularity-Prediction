
# Social Media News Popularity Prediction

This project aims to predict the popularity of news articles on social media platforms like Facebook, GooglePlus, and LinkedIn using various machine learning models. The dataset contains news articles with metadata, time-series data, and popularity measures across these platforms. The pipeline involves data preprocessing, feature engineering, outlier treatment, sentiment analysis, and applying several machine learning algorithms to improve the model's predictive performance.

## Table of Contents

1. [Installation](#installation)
2. [Project Workflow](#project-workflow)
3. [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
4. [Machine Learning Models](#machine-learning-models)
5. [Model Evaluation and Hyperparameter Tuning](#model-evaluation-and-hyperparameter-tuning)
6. [Results and Visualization](#results-and-visualization)

## Installation

To run this project, you need the following dependencies installed:

```bash
!pip install -U textblob
!pip install catboost
!pip install -U scikit-learn
!pip install shap
!pip install lightgbm
!pip install xgboost
!pip install tensorflow
```

Alternatively, you can create a virtual environment and install the required packages by running:

```bash
pip install -r requirements.txt
```

Make sure to have the required Python libraries for NLP and machine learning installed.

## Project Workflow

1. **Loading the Dataset**: We start by loading and exploring the dataset, which includes news metadata, such as titles, headlines, and time series data representing popularity on social media platforms.
   
2. **Preprocessing**: The data preprocessing step includes:
   - Handling missing values
   - Treating outliers using the 90th percentile
   - Scaling numerical features
   - Applying cyclic transformations to time variables for better feature representation

3. **Feature Engineering**:
   - We introduce new features like document length (Title and Headline word length).
   - Sentiment categories are derived from sentiment analysis using `SentimentTitle` and `SentimentHeadline` fields.

4. **Data Split**: The dataset is split into train and test sets for model training and evaluation.

## Preprocessing and Feature Engineering

1. **Outlier Treatment**: The outliers are handled by applying a 90th percentile cutoff for the popularity metrics across platforms.

2. **Scaling**: Numerical features like social media popularity metrics are scaled using Standard Scaler.

3. **Feature Transformation**:
   - **Logarithmic Transformation**: For popularity measures.
   - **Cyclic Time Transformation**: We transform time variables using sine and cosine functions to preserve the cyclical nature of the data.

4. **NLP Processing**:
   - **TF-IDF Vectorization**: We apply TF-IDF on the combined 'Title' and 'Headline' columns.
   - **Word2Vec Embeddings**: Word embeddings are generated from titles and headlines using Gensim's Word2Vec model.

5. **Sentiment Classification**: Sentiment scores are categorized into positive, negative, and neutral sentiments for both headlines and titles.

## Machine Learning Models

The following machine learning algorithms have been implemented to predict social media popularity:

1. **Random Forest**
2. **Extra Trees**
3. **Decision Tree**
4. **CatBoost**
5. **LightGBM**
6. **Gradient Boosting**
7. **XGBoost**
8. **Support Vector Machine (SVM)**

Each algorithm is optimized through hyperparameter tuning using:
- **Randomized Search CV**
- **Grid Search CV**
- **Halving Randomized Search CV**

## Model Evaluation and Hyperparameter Tuning

We evaluate the models using the following metrics:

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Adjusted R² Score**

Hyperparameter tuning is performed using the best-performing models to improve their performance. We optimize models like Random Forest, XGBoost, and LightGBM using RandomizedSearchCV and GridSearchCV.

## Results and Visualization

- **Feature Importance**: Feature importance plots are generated to visualize the most significant features influencing the popularity predictions.
- **Correlation Heatmap**: Displays the relationships between features.
- **SHAP (SHapley Additive exPlanations)**: SHAP values are used to interpret the feature importance for models like CatBoost and LightGBM.

### Visualizations:
1. **Outliers Visualization**: Outliers for different platforms are visualized using box plots and distplots.
2. **Sentiment Distribution**: Bar charts to display the distribution of news items across positive, negative, and neutral sentiments.
3. **Feature Importance Plot**: Shows the importance of individual features in predicting news popularity.

---

### Notes:

- Ensure the dataset is available at the specified location, or adjust the paths in the script accordingly.
- If you're using a GPU-enabled environment for TensorFlow or XGBoost, make sure the corresponding libraries are set up properly.

