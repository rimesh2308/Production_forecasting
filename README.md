# Production Forecasting with LSTM Neural Networks

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
## 📋 Project Overview

**PetroForecast AI** is an advanced oil production forecasting system using **LSTM (Long Short-Term Memory) neural networks**. This project provides both univariate and multivariate forecasting approaches with an interactive **Streamlit web interface** for real-time production predictions.

## 🎯 Objectives

- **Forecast** oil production using deep learning LSTM networks
- **Provide** both univariate and multivariate forecasting approaches
- **Deliver** interactive web-based forecasting interface
- **Optimize** model performance with advanced neural network techniques
- **Export** results for business intelligence and planning

## 🛠 Technical Implementation

### Deep Learning Architecture
- **LSTM Networks**: Advanced time series forecasting
- **Univariate Approach**: Production history-based predictions (more stable)
- **Multivariate Approach**: Including operational parameters
- **Real-time Training**: Early stopping and learning rate scheduling

### Neural Network Features
- **Sequential Architecture**: Layered LSTM design
- **Dropout Layers**: Preventing overfitting
- **Adam Optimizer**: Adaptive learning rate optimization
- **Custom Callbacks**: Early stopping and model checkpointing

## 📊 Key Features
- **Interactive Visualizations**: Plotly-based charts and graphs
- **Model Configuration**: Adjustable parameters and settings
- **Export Capabilities**: CSV and JSON output formats

### Forecasting Capabilities
- **Flexible Time Windows**: 7-180 days lookback periods
- **Variable Horizons**: 1-180 days forecast periods
- **History Matching**: Model validation and performance assessment

## 🔧 Technical Specifications

### Data Parameters

#### Production Data
- **Daily Oil Production**: Primary time series target
- **Historical Trends**: Long-term production patterns
- **Data Quality**: Automated preprocessing and validation

### Model Configuration
- **LSTM Units**: 50-200 neurons per layer
- **Sequence Length**: Configurable lookback windows
- **Batch Size**: Optimized for memory and performance
- **Epochs**: Adaptive training with early stopping

## 🚀 Usage Instructions

### Running the Application
```bash
# Navigate to project directory
cd production_forecasting

# Install dependencies
pip install streamlit tensorflow pandas numpy plotly scikit-learn

# Launch the Streamlit app
streamlit run forecasting.py
```

## 📈 Key Features

### Model Performance
- **Multiple Metrics**: Comprehensive evaluation suite
- **Validation**: Train-test split and cross-validation
- **Uncertainty Quantification**: Prediction confidence intervals
- **Model Comparison**: Univariate vs. multivariate performanc

## 🔍 Technical Details

### Data Processing
- **MinMax Scaling**: Feature normalization for neural networks
- **Sequence Generation**: Time window creation for LSTM input
- **Data Validation**: Quality checks and missing value handling
- **Feature Engineering**: Derived metrics and transformations

### Neural Network Architecture
```python
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(lookback, features)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])
```

### Training Process
- **Early Stopping**: Prevent overfitting with patience monitoring
- **Learning Rate Scheduling**: Adaptive rate adjustment
- **Model Checkpointing**: Save best performing models
- **Validation Monitoring**: Real-time performance tracking


---

---

*Part of the PetroAI-Suite - Comprehensive AI Solutions for Oil & Gas Industry*
