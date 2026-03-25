# Smart Property Investment Insights
### AI & Data Analytics Portfolio Project

An end-to-end data science project that demonstrates the full analytics lifecycle — from raw data through
cleaning, exploratory analysis, feature engineering, machine learning, clustering, and investment
insight generation — applied to South African residential and commercial property data.

---

## Table of Contents
1. [Project Objective](#project-objective)
2. [Project Structure](#project-structure)
3. [Data Sources](#data-sources)
4. [Methodology](#methodology)
5. [Key Results](#key-results)
6. [Visualisations](#visualisations)
7. [How to Run](#how-to-run)
8. [Skills Demonstrated](#skills-demonstrated)
9. [Limitations & Future Work](#limitations--future-work)

---

## Project Objective

Design and implement an AI-powered data analysis system that helps property investors make
informed decisions by:

- Identifying investment hotspots across South African regions
- Predicting property sale prices with ensemble machine learning models
- Segmenting properties into investment tiers using unsupervised clustering
- Quantifying ROI metrics including gross rental yield and net cashflow

---

## Project Structure

```
smart_property_insights/
│
├── data/
│   └── raw_property_data.csv          # Synthetic dataset (2,000+ records)
│
├── notebooks/
│   └── smart_property_analysis.ipynb  # Full step-by-step Jupyter notebook
│
├── src/
│   ├── generate_dataset.py            # Synthetic data generator
│   ├── analysis_pipeline.py           # End-to-end analysis & modelling script
│   ├── generate_notebook.py           # Notebook builder utility
│   └── app.py                         # Streamlit prediction app
│
├── outputs/
│   ├── cleaned_property_data.csv      # Cleaned dataset
│   ├── final_enriched_dataset.csv     # Dataset with all engineered features
│   ├── results_summary.json           # Model & cluster metrics
│   ├── fig01_price_distributions.png
│   ├── fig02_price_by_region.png
│   ├── fig03_property_types.png
│   ├── fig04_correlation_heatmap.png
│   ├── fig05_price_per_sqm_boxplot.png
│   ├── fig06_roi_analysis.png
│   ├── fig07_kmeans_elbow.png
│   ├── fig08_clusters_scatter.png
│   ├── fig09_cluster_heatmap.png
│   ├── fig10_predicted_vs_actual.png
│   ├── fig11_residuals.png
│   ├── fig12_feature_importance.png
│   ├── fig13_model_comparison.png
│   └── fig14_investment_dashboard.png
│
├── docs/
│   └── project_report.md              # Extended project report
│
├── requirements.txt
└── README.md
```

---

## Data Sources

The dataset is **synthetically generated** using domain-informed parameters based on:

- South African property market regional price dynamics (StatsSA, Lightstone property reports)
- Typical property size distributions by type (SAPOA guidelines)
- Realistic rental yield ranges for South African metros (PayProp Rental Index)
- Property age distributions reflecting building activity trends

**Dataset features (18 columns):**

| Column | Type | Description |
|---|---|---|
| PropertyID | String | Unique property identifier |
| Region | Categorical | One of 15 SA regions (Sandton, Cape Town CBD, etc.) |
| PropertyType | Categorical | House, Apartment, Townhouse, Commercial, Duplex |
| SizeSqm | Float | Floor area in square metres |
| Bedrooms | Integer | Number of bedrooms |
| Bathrooms | Integer | Number of bathrooms |
| YearBuilt | Integer | Construction year |
| PropertyAge | Integer | Derived: 2024 − YearBuilt |
| HasGarage | Binary | Garage present (1/0) |
| HasPool | Binary | Swimming pool present (1/0) |
| SecurityLevel | Ordinal | 1 = Basic, 2 = Standard, 3 = High |
| ProximityToAmenities_km | Float | Distance to nearest amenity hub (km) |
| SalePrice | Float | Sale price in ZAR |
| MonthlyRental | Float | Monthly rental income in ZAR |
| PricePerSqm | Float | Derived: SalePrice / SizeSqm |
| GrossRentalYield | Float | Derived: (MonthlyRental × 12 / SalePrice) × 100 |
| NetROI | Float | Net return after estimated holding costs |
| SaleDate | Date | Date of sale transaction |

**Intentional data quality issues introduced for realism:**
- ~4% missing values across Bedrooms, SizeSqm, YearBuilt, MonthlyRental
- ~1% duplicate records
- 5 extreme outlier prices (both very high and very low)

---

## Methodology

### Step 1 — Data Cleaning
- Removed 20 duplicate PropertyIDs (keep first occurrence)
- Imputed missing numeric columns using **median by PropertyType**
- Imputed missing MonthlyRental using **median by Region + PropertyType**
- Removed outlier prices using **IQR fence** (1st–99th percentile bounds)

### Step 2 — Exploratory Data Analysis
- Distribution analysis of sale price and rental income
- Regional median price comparison with directional colour-coding
- Property type breakdown (count + median price)
- Feature correlation matrix (Pearson) identifying top price drivers
- Rental yield analysis by region (bar chart + scatter)
- Price-per-sqm regional boxplots

### Step 3 — Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `SqmPerBedroom` | SizeSqm / Bedrooms | Space efficiency signal |
| `AgeCategory` | Binned PropertyAge | Ordinal age grouping for non-linear effects |
| `PremiumLocation` | Binary flag (top-5 regions) | Captures location premium |
| `InvestmentScore` | Weighted composite (yield, location, security, age) | Summary investment metric |
| `LogSalePrice` | log1p(SalePrice) | Normalised target for analysis |

### Step 4 — K-Means Clustering
- Clustering features: PricePerSqm, GrossRentalYield, InvestmentScore, SizeSqm, PropertyAge, SecurityLevel
- Features standardised with **StandardScaler**
- Optimal k selected via **elbow method** → **k = 4**
- Clusters auto-labelled by InvestmentScore rank: Low / Moderate / High / Prime

### Step 5 — Predictive Modelling

**Models trained:** Random Forest Regressor, Gradient Boosting Regressor  
**Target:** SalePrice  
**Train / Test split:** 80% / 20% (stratified random, seed=42)  
**Features (14):** SizeSqm, Bedrooms, Bathrooms, PropertyAge, HasGarage, HasPool, SecurityLevel,
ProximityToAmenities_km, PremiumLocation, SqmPerBedroom, InvestmentScore, Region_enc, PropType_enc, AgeCategory_enc

**Hyperparameters:**

| Parameter | Random Forest | Gradient Boosting |
|---|---|---|
| n_estimators | 200 | 300 |
| max_depth | 20 | 5 |
| learning_rate | — | 0.08 |
| subsample | — | 0.8 |
| min_samples_leaf | 4 | 5 |

---

## Key Results

### Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | R 1,056,106 | R 502,644 | 0.866 |
| **Gradient Boosting** | **R 992,048** | **R 415,206** | **0.882** |

Gradient Boosting is the superior model, explaining **88.2%** of variance in sale price on the held-out test set.

### Top Feature Importances (Gradient Boosting)
1. `SizeSqm` — floor area is the single strongest predictor
2. `InvestmentScore` — composite score captures regional and quality signals
3. `Region_enc` — location remains a dominant value driver
4. `PropertyAge` — newer properties command a clear premium
5. `SqmPerBedroom` — space-per-bedroom captures bedroom density quality

### Investment Clusters

| Cluster | Label | Avg Price | Avg Rental Yield | Count |
|---|---|---|---|---|
| 0 | Prime Investment | R 6,275,335 | 5.98% | 388 |
| 1 | High Potential | R 1,601,121 | 7.68% | 555 |
| 2 | Moderate Potential | R 4,199,000 | 5.73% | 487 |
| 3 | Low Potential | R 2,298,024 | 6.13% | 570 |

### Top Investment Regions (by InvestmentScore)
1. **Rosebank** — Premium location, strong demand, competitive pricing
2. **Sandton** — CBD of Africa, highest capital values
3. **Cape Town CBD** — Tourism + corporate demand supports dual-income strategy
4. **Waterfront** — Highest absolute prices but stable long-term appreciation
5. **Stellenbosch** — Growing tech hub + wine estate lifestyle premium

### Key Insights

- **Apartments 80–150 m²** in mid-premium locations (Umhlanga, Fourways) deliver the best yield/price ratio at 7–9%
- **Security Level 3** estates show a measurable **8–10% price premium** vs Level 1
- Properties **under 15 years old** command a consistent **12–18% premium** over similar older stock
- Properties **over 30 years old** in non-premium regions consistently show **negative NetROI** after maintenance
- **Proximity to amenities** has a clear inverse price relationship — every additional km reduces value

---

## Visualisations

All charts are saved to `outputs/`. Key figures:

| Figure | Description |
|---|---|
| fig01 | Sale price & rental distributions |
| fig02 | Median sale price by region (horizontal bar) |
| fig03 | Property type count + median price |
| fig04 | Full feature correlation heatmap |
| fig05 | Price-per-sqm by region (boxplot) |
| fig06 | Rental yield by region + price vs yield scatter |
| fig07 | K-Means elbow curve |
| fig08 | Cluster scatter (price vs yield) |
| fig09 | Cluster characteristics heatmap |
| fig10 | Predicted vs actual — both models |
| fig11 | Residual plots — both models |
| fig12 | Feature importance — both models |
| fig13 | Model performance comparison |
| fig14 | Full investment hotspot dashboard (2×2) |

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 1. Generate the dataset
```bash
python src/generate_dataset.py
```

### 2. Run the full analysis pipeline
```bash
python src/analysis_pipeline.py
```
Outputs: cleaned CSV, enriched CSV, all 14 figures, results_summary.json

### 3. Open the Jupyter notebook
```bash
jupyter notebook notebooks/smart_property_analysis.ipynb
```

### 4. Launch the Streamlit prediction app
```bash
streamlit run src/app.py
```
Enter property features in the sidebar → receive instant price prediction, rental estimate,
yield calculation, and investment rating.

---

## Skills Demonstrated

| Skill Area | Tools / Techniques |
|---|---|
| Data generation | NumPy, Pandas, domain-informed simulation |
| Data cleaning | Deduplication, IQR outlier removal, grouped median imputation |
| EDA | Matplotlib, Seaborn, distribution analysis, correlation matrices |
| Feature engineering | Domain knowledge, binning, composite scoring, label encoding |
| Unsupervised ML | K-Means clustering, elbow method, StandardScaler |
| Supervised ML | Random Forest, Gradient Boosting, train/test split, cross-validation |
| Model evaluation | RMSE, MAE, R², residual analysis, feature importance |
| Visualisation | 14 publication-quality figures, multi-panel dashboards |
| Application | Streamlit interactive prediction app |
| Documentation | README, inline docstrings, notebook narrative |

---

## Limitations & Future Work

**Current limitations:**
- Dataset is synthetic — real-world data would require integration with Lightstone, Deeds Office, or PropTrack APIs
- No geospatial mapping (requires Plotly or Folium — not available in offline environment)
- Models are not hyperparameter-tuned via GridSearchCV (resource constraint)
- No time-series forecasting component

**Future enhancements:**
- Integrate real Lightstone / PropStats API data
- Add Folium choropleth map of property values by region
- Build a hyperparameter tuning pipeline (Optuna or GridSearchCV)
- Add XGBoost and LightGBM for comparison
- Develop a time-series price forecasting module (ARIMA / Prophet)
- Deploy Streamlit app to Streamlit Cloud or Heroku
- Add SHAP explainability plots for individual property predictions

---

## Author

**[Your Name]**  
Data Scientist | Python · Machine Learning · Analytics  
[LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

---

*This project was built for portfolio purposes. The dataset is synthetic and does not represent
any real property transactions. The models and analysis are for demonstration only and should
not be used as financial or investment advice.*
