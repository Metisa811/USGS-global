# USGS Global Earthquake Forecasting Project

A comprehensive research project for predicting earthquakes worldwide using USGS official data.

## 📋 Project Description

This project focuses on analyzing and forecasting global earthquakes using official data from USGS (United States Geological Survey). It implements advanced machine learning methods including Graph Convolutional Networks (GCN) and other neural network architectures.

## 🎯 Project Objectives

- Analyze patterns of global earthquake occurrences
- Forecast the probability of future earthquakes
- Compare different machine learning models
- Create a reproducible pipeline for researchers
- Develop interpretable predictions

## 📁 Repository Structure

```
USGS-global/
├── README.md                                  # Project overview
├── PROJECT_LOG.md                            # Project progress tracking
├── notebooks/
│   ├── USGS_Global_Earthquake_Project.ipynb # Main project notebook
│   └── USGS_global.ipynb                    # Data analysis and exploration
├── docs/
│   └── METHODOLOGY.md                        # Detailed methodology
├── data/
│   ├── raw/                                 # Original USGS data
│   └── processed/                           # Cleaned and processed data
├── models/
│   ├── gcn_model.pkl                        # Trained GCN model
│   └── model_metrics.json                   # Performance metrics
└── scripts/
    ├── data_processing.py                   # Data preprocessing
    ├── feature_engineering.py               # Feature extraction
    └── model_training.py                    # Model training pipeline
```

## 🔬 Machine Learning Models

**Best Current Model:** Graph Convolutional Network (GCN)
- **ROC-AUC Score:** ≈ 0.83
- **Precision:** 0.81
- **Recall:** 0.85
- **Advantages:** Better understanding of spatial and temporal relationships between earthquakes

## 📊 Dataset Information

- **Source:** USGS Earthquake Hazards Program API
- **Coverage:** Global earthquake events
- **Time Period:** Multiple years of historical data
- **Features:** 
  - Timestamp (date and time)
  - Geographic coordinates (latitude, longitude)
  - Depth (km)
  - Magnitude (Richter scale)
  - Location information

## 🛠️ Technologies & Libraries

- **Language:** Python 3.8+
- **Notebooks:** Jupyter Notebook
- **Data Processing:**
  - Pandas - Data manipulation and analysis
  - NumPy - Numerical computations
  
- **Machine Learning:**
  - Scikit-learn - Classical ML algorithms
  - TensorFlow/PyTorch - Deep learning frameworks
  - NetworkX - Graph operations
  - DGL (Deep Graph Library) - Graph neural networks
  
- **Visualization:**
  - Matplotlib - Static plots
  - Seaborn - Statistical visualizations
  - Plotly - Interactive visualizations
  - Folium - Geographic maps

## 📈 Project Progress

For detailed progress tracking and milestones, see [`PROJECT_LOG.md`](PROJECT_LOG.md).

### Key Milestones
- ✅ Data collection and preprocessing
- ✅ Feature engineering
- ✅ GCN model development
- ✅ Model evaluation and validation
- ⏳ Ensemble methods implementation
- ⏳ Real-time prediction API

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Metisa811/USGS-global.git
cd USGS-global
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
```bash
pip install pandas numpy scikit-learn tensorflow networkx dgl plotly folium seaborn matplotlib jupyter
```

### Running the Project

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `USGS_Global_Earthquake_Project.ipynb`

3. Execute cells sequentially to:
   - Load and explore data
   - Preprocess features
   - Train the GCN model
   - Evaluate performance
   - Generate predictions

## 📊 Model Performance Comparison

| Model | ROC-AUC | Precision | Recall | Status |
|-------|---------|-----------|--------|--------|
| GCN | 0.83 | 0.81 | 0.85 | ✅ Active |
| Random Forest | 0.75 | 0.78 | 0.72 | ✅ Tested |
| Neural Network | 0.78 | 0.79 | 0.77 | ✅ Tested |
| SVM | 0.72 | 0.74 | 0.70 | ✅ Tested |

## 🔍 Key Features

### Data Features
- Magnitude-based classification
- Temporal patterns and seasonality
- Spatial clustering and proximity analysis
- Depth distribution analysis
- Inter-event time statistics

### Model Capabilities
- Binary and multi-class earthquake prediction
- Risk assessment for geographic regions
- Temporal forecasting
- Uncertainty quantification
- Feature importance analysis

## 📝 Research Notes

- Ongoing research to improve model accuracy
- Investigation of ensemble methods combining multiple models
- Analysis of feature importance using SHAP values
- Testing on independent validation datasets

## 🎓 Methodology Highlights

For comprehensive technical details, see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)

### Data Pipeline
```
Raw USGS Data → Validation → Cleaning → Feature Engineering → Model Training → Evaluation
```

### Feature Engineering
- **Temporal Features:** Hour, day of week, seasonal indicators
- **Spatial Features:** Grid encoding, distance metrics, cluster assignments
- **Historical Features:** Magnitude statistics, frequency windows, inter-event times

### Model Architecture
- 2-layer Graph Convolutional Network
- ReLU activation functions
- Global graph pooling
- Dense output layers with sigmoid classification

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Improving model accuracy
- Adding new features
- Optimizing performance
- Expanding dataset coverage
- Creating visualization tools

## 📚 References & Resources

- [USGS Earthquake Hazards Program](https://www.usgs.gov/programs/VHP/earthquake)
- [Kipf & Welling (2017) - Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- Recent papers on earthquake forecasting with machine learning
- Global Seismic Network data

## 📞 Contact & Support

- **Project Owner:** Metisa811
- **Repository:** https://github.com/Metisa811/USGS-global
- **Issues:** Please report bugs and feature requests via GitHub Issues
- **Email:** Available through GitHub profile

## 📄 License

This project is public and available for research and educational purposes.

---

**Last Updated:** June 30, 2026

**Status:** 🚀 Active Development
