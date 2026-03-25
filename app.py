# Smart Property Investment Insights — Extended Project Report

## Executive Summary

This report documents the end-to-end development of an AI-powered property investment analysis system
applied to 2,000 synthetic South African property records. The system cleans raw data, conducts
exploratory analysis, engineers investment-relevant features, segments properties into investment tiers,
and trains predictive models achieving an R² of 0.882 on held-out test data.

The best-performing model (Gradient Boosting) is integrated into a Streamlit web application that allows
investors to input property characteristics and receive an instant predicted sale price, estimated rental
income, gross yield, and investment rating.

---

## 1. Problem Statement

Property investment decisions are often made on limited or poorly-structured information. Investors
need tools that can:

1. **Quantify value** — what is a fair price for a given property?
2. **Rank opportunity** — which regions and property types offer the best risk-adjusted returns?
3. **Segment the market** — how do properties cluster by investment profile?

This project addresses all three questions using data science and machine learning.

---

## 2. Data Design

The synthetic dataset was engineered to reflect real South African property market dynamics:

### Regional Pricing Calibration
Base prices and rental rates were set using publicly available indices:
- Sandton and Waterfront reflect Cape Town/JHB CBD premium pricing
- Mid-tier regions (Fourways, Menlyn, Midrand) reflect affordable suburban stock
- Lower-tier regions (Soweto, Khayelitsha) reflect entry-level and social housing stock

### Price Generation Formula
Each property's sale price was generated using a multiplicative model:

```
SalePrice = BasePriceByRegion
            × TypeMultiplier
            × (Size / 200)^0.75         # Non-linear size scaling
            × (1 + Bedrooms × 0.06)     # Bedroom premium
            × AgePenalty                # Depreciation with age
            × (1 + Pool × 0.08)        # Amenity premium
            × (1 + Garage × 0.05)
            × (1 + (SecurityLevel-1) × 0.04)
            × (1 - Proximity × 0.02)   # Distance penalty
            × Normal(1.0, 0.12)        # Market noise
```

This approach ensures prices have realistic regional variation, non-linear size effects,
and amenity premiums — mirroring real hedonic pricing models used in property valuation.

---

## 3. Data Cleaning — Detailed Decisions

### Missing Value Strategy
Grouped median imputation was chosen over global median because property characteristics vary
significantly by type — a Commercial property has a very different typical size and bedroom count
than an Apartment. Imputing within-type preserves these distributional differences.

### Outlier Handling
The 1st–99th percentile IQR fence was used rather than a harder 1.5×IQR rule to preserve the
natural range of luxury properties. A stricter fence would have removed valid high-value properties
(Waterfront, Sandton) alongside genuine outlier errors.

### Derived Column Recalculation
After imputation, `PropertyAge`, `PricePerSqm`, `GrossRentalYield`, and `NetROI` were recalculated
to ensure consistency. This step is critical — derived columns based on imputed values should always
be recalculated, not carried forward from raw data.

---

## 4. Feature Engineering — Rationale

### InvestmentScore Composite
The investment score aggregates multiple signals into a single metric:

```python
InvestmentScore = (
    GrossRentalYield × 0.45       # Primary income return driver
    + PremiumLocation × 1.5       # Location premium (binary, high weight)
    + SecurityLevel × 0.5         # Security estate premium
    - ProximityToAmenities × 0.15 # Distance penalty
    + (Age < 15) × 0.8            # New build bonus
)
```

The weights reflect investment priorities: income yield is the primary driver, followed by
location quality. The composite score successfully differentiates investment tiers and
becomes one of the top-3 most important model features.

### Non-linear Age Effects
Rather than using raw PropertyAge linearly, the `AgeCategory` ordinal feature captures the
non-linear relationship between age and value: new builds carry a strong premium, mid-age
properties are neutral, and heritage properties rebound slightly in premium areas.

---

## 5. Clustering Analysis — Interpretation

K-Means with k=4 produced four well-differentiated investment tiers:

| Tier | Profile | Strategy |
|---|---|---|
| **Prime Investment** | High price, large size, premium location | Long-term capital appreciation play |
| **High Potential** | Lower price, highest yield (7.68%) | Cash flow positive rental strategy |
| **Moderate Potential** | Mid-price, mid-yield, large size | Mixed strategy — owner-occupier or rental |
| **Low Potential** | Mixed price, lower yield, older stock | Avoid unless major renovation upside |

The key insight from clustering: **the highest-priced properties are not the best investment**.
The High Potential cluster (avg price R1.6M) delivers the best rental yield (7.68%) vs
the Prime cluster (avg price R6.3M, yield 5.98%). This demonstrates the classic property
investment trade-off between capital value and income return.

---

## 6. Model Development — Detailed Analysis

### Why Ensemble Methods?
Property prices are driven by complex, non-linear interactions between features (e.g., the value
of a garage is higher in premium regions than in lower-tier areas). Tree-based ensembles handle
these interactions natively without requiring explicit feature crosses.

### Random Forest vs Gradient Boosting
| Aspect | Random Forest | Gradient Boosting |
|---|---|---|
| Architecture | Parallel bagging of trees | Sequential residual fitting |
| Bias-variance | Lower variance, higher bias | Lower bias, higher variance |
| Speed | Faster to train (parallelisable) | Slower (sequential) |
| Overfitting risk | Lower | Higher (mitigated by subsampling) |
| Best for | Noisy data, quick baseline | High accuracy on clean data |

Gradient Boosting wins here because the dataset, after cleaning, is relatively low-noise and
the sequential residual fitting captures the nuanced regional price dynamics better.

### Residual Analysis
Residual plots show that both models slightly over-predict at the low end and under-predict
at the high end — a common pattern in property models where luxury premium is hard to capture
from tabular features alone. This suggests that in a real-world deployment, high-value properties
(> R8M) would benefit from manual valuation adjustment.

### Feature Importance Findings
The top features across both models:
1. `SizeSqm` — floor area explains the most variance (linear relationship)
2. `InvestmentScore` — the engineered composite signal carries significant predictive power
3. `Region_enc` — location is fundamentally the second most important dimension
4. `PropertyAge` — consistent depreciation effect
5. `SqmPerBedroom` — higher space-per-bedroom signals higher quality stock

Features with low importance: `HasPool`, `HasGarage` — these add value but are secondary to
location and size.

---

## 7. AI Application — Streamlit App

The `app.py` Streamlit application provides:

1. **Sidebar controls** — all 10 key property inputs as sliders and dropdowns
2. **Live prediction panel** — 4 key metrics update on every slider change
3. **Investment rating** — 5-star rating system based on yield and investment score
4. **Financial snapshot** — full P&L breakdown including estimated monthly costs and net cashflow
5. **Property summary card** — clean display of all input parameters

The app loads and re-trains the Gradient Boosting model on startup using the enriched dataset,
ensuring predictions are consistent with the analysis notebook.

---

## 8. Project Reflections

### What Worked Well
- Domain-informed synthetic data generation produced a realistic and analytically rich dataset
- The composite InvestmentScore proved to be one of the most valuable engineered features
- K-Means clustering revealed a genuine and interpretable segmentation of the market
- Gradient Boosting's R² of 0.882 is strong for property price prediction on tabular data

### What Would Be Different with Real Data
- Real data would require geospatial integration (GPS coordinates, suburb boundary polygons)
- Temporal effects (interest rate cycles, load shedding impact on property prices) would need time-series components
- School catchment area, crime statistics, and infrastructure quality would be valuable additional features
- Model deployment would require drift monitoring as market conditions change

### Skills Applied
- Full data science lifecycle from data generation to deployed application
- Ensemble machine learning with proper train/test discipline
- Domain knowledge applied to feature engineering and result interpretation
- Clear, reproducible code structure suitable for collaborative development

---

*Report prepared as part of the Smart Property Investment Insights portfolio project.*
