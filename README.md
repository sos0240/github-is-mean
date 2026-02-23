# Operator 1 -- Point-in-Time Financial Analysis Pipeline

Institutional-grade equity research powered by free government filing APIs. Covers 10 markets, $91+ trillion in market cap, with 25+ mathematical models and zero data subscription costs.

---

## What Is Operator 1?

Operator 1 is an open-source equity research pipeline that performs the same type of fundamental and quantitative analysis that institutional investors pay $20,000+/year per Bloomberg Terminal to access -- except it pulls data directly from free government filing APIs and runs everything locally on your machine.

The core idea: every public company in the world is required by law to file financial statements with a government regulator. Those filings are public, free, and immutable (they have a filing date that never changes). Operator 1 taps into those regulatory APIs across 10 global markets, normalizes the data into a single format, runs 25+ quantitative models, and produces a professional investment report.

No subscriptions. No paid data feeds. Just government data and math.

---

## How It Works

You pick a market, search for a company, and Operator 1 runs an 8-step pipeline:

**Step 1 -- Data Source Selection**
Choose a region and market. The app connects to the corresponding government filing API (SEC EDGAR for the US, EDINET for Japan, DART for Korea, etc.).

**Step 2 -- Company Profile**
Fetches the company's identity, sector, industry, and exchange information from the regulatory database.

**Step 3 -- Point-in-Time Financial Data**
Downloads income statements, balance sheets, and cash flow statements with their original filing dates. Also fetches OHLCV price data. All data is merged using filing dates (not report dates) to prevent look-ahead bias -- the same discipline used by quantitative hedge funds.

**Step 4 -- Daily Cache Construction**
Builds a unified time-indexed table where financial statement data is forward-filled onto daily price data using as-of joins. Macroeconomic indicators (GDP, inflation, interest rates, unemployment, FX) from central bank APIs are merged in. Missing values are filled using accounting identities first, then Bayesian Ridge regression or a Variational Autoencoder.

**Step 5 -- Feature Engineering & Analysis**
Computes 25+ derived financial ratios (margins, coverage ratios, yields, returns). Runs survival mode detection to flag companies in financial distress. Applies fuzzy logic government protection scoring. Calculates financial health scores (0-100) across five tiers: liquidity, solvency, stability, profitability, and growth. Runs ethical filters for purchasing power, leverage, speculation risk, and cash flow quality. Optionally discovers linked entities (competitors, suppliers, customers) via Gemini AI.

**Step 6 -- Temporal Modeling**
Runs 25+ models including:
- Regime detection (Hidden Markov Model, Gaussian Mixture, PELT structural breaks, Bayesian change point)
- Forecasting (Kalman Filter, GARCH, VAR, LSTM, Random Forest, XGBoost, Transformer)
- Monte Carlo simulation (10,000 regime-aware paths with importance sampling)
- Causality analysis (Granger, Transfer Entropy, Copula tail dependency)
- Uncertainty quantification (Conformal Prediction, distribution-free intervals)
- Pattern recognition (candlestick detector, Fourier/wavelet cycles, DTW historical analogs)
- Optimization (Genetic Algorithm for ensemble weights, PID adaptive learning)
- Explainability (SHAP feature attribution, Sobol sensitivity)

Each model is wrapped in a fallback chain -- if a dependency is missing or data is insufficient, the next model in the chain picks up automatically.

**Step 7 -- Profile Assembly**
All results are collected into a single `company_profile.json` containing every metric, score, forecast, and model output.

**Step 8 -- Report Generation**
Produces three tiers of investment reports:
- **Basic** (5 sections) -- quick screening
- **Pro** (13 sections) -- with peers and macro context
- **Premium** (22 sections) -- full institutional-grade analysis

Reports are generated via Google Gemini AI for natural language narrative, with a structured template fallback when Gemini is unavailable. Optional PDF output via pandoc.

---

## Supported Markets ($91T+ Coverage)

| Region | Country | Exchange | Market Cap | Data Source | Macro API | Free? |
|--------|---------|----------|-----------|-------------|-----------|-------|
| North America | **United States** | NYSE / NASDAQ | $50T | SEC EDGAR | FRED | Yes |
| Europe | **United Kingdom** | LSE | $3.18T | Companies House | ONS | Yes |
| Europe | **European Union** | ESEF (pan-EU) | $8-9T | ESEF / XBRL | ECB SDW | Yes |
| Europe | **France** | Paris / Euronext | $3.13T | ESEF | INSEE | Yes |
| Europe | **Germany** | Frankfurt / XETRA | $2.04T | ESEF | Bundesbank | Yes |
| Asia | **Japan** | Tokyo (JPX) | $6.5T | EDINET | e-Stat | Yes |
| Asia | **South Korea** | KOSPI / KOSDAQ | $2.5T | DART | KOSIS | Yes |
| Asia | **Taiwan** | TWSE / TPEX | ~$1.2T | MOPS | DGBAS | Yes |
| South America | **Brazil** | B3 | $2.2T | CVM | BCB | Yes |
| South America | **Chile** | Santiago | $0.4T | CMF | BCCh | Yes |

All data sources are free government APIs. No Bloomberg Terminal or paid data subscriptions required.

---

## What's in the Report?

The premium report covers 22 sections:

| # | Section | What it tells you |
|---|---------|------------------|
| 1 | Executive Summary | Key findings and recommendation at a glance |
| 2 | Company Overview | Identity, sector, exchange, currency |
| 3 | Historical Performance | 2-year return, Sharpe ratio, max drawdown, up/down day statistics |
| 4 | Current Financial Snapshot | Actual ratio values organized by tier (cash ratio, debt/equity, margins, P/E) |
| 5 | Financial Health Scoring | Composite 0-100 score with tier breakdown, Altman Z-Score, Beneish M-Score |
| 6 | Survival Mode Analysis | Distress detection, episode history, macro context from central bank data |
| 7 | Linked Variables | Sector performance, competitor health, supply chain risk |
| 8 | Temporal Analysis | Market regime classification, structural breaks, model insights |
| 9 | Predictions & Forecasts | Day/week/month/year forecasts with uncertainty bands |
| 10 | Technical Patterns | Candlestick patterns detected + predicted OHLC chart series |
| 11 | Ethical Filters | Purchasing power, solvency, speculation risk, cash flow quality |
| 12 | Supply Chain Risk | Network topology, contagion probability, concentration risk |
| 13 | Competitive Landscape | Market structure, leadership position, equilibrium shares |
| 14 | Government Protection | Regulatory protection degree, sector strategicness |
| 15 | Model Calibration | Adaptive learning feedback, recalibration history |
| 16 | Market Sentiment | News flow scoring, sentiment trend |
| 17 | Peer Comparison | Percentile ranking against peers |
| 18 | Macroeconomic Environment | GDP/inflation quadrant classification |
| 19 | Advanced Insights | Cycle analysis, tail risk, causality network, sensitivity, deep learning |
| 20 | Risk Factors & Limitations | Data caveats, model assumptions, failed modules |
| 21 | Investment Recommendation | Multi-factor BUY/HOLD/SELL with detailed rationale |
| 22 | Appendix & Methodology | Technical details, tier definitions, data sources |

---

## The Survival Hierarchy

The app uses a 5-tier survival hierarchy that dynamically reweights what matters based on whether a company (or its country) is in financial distress:

| Tier | Category | Normal Weight | Extreme Survival Weight |
|------|----------|--------------|------------------------|
| 1 | Liquidity & Cash | 20% | 60% |
| 2 | Debt & Solvency | 20% | 30% |
| 3 | Market Stability | 20% | 10% |
| 4 | Profitability | 20% | 0% |
| 5 | Growth & Valuation | 20% | 0% |

When a company is in crisis, growth metrics like P/E ratio become irrelevant -- what matters is whether it can make payroll. The system mirrors how credit analysts actually think during a downturn.

Four regimes are supported:
- **Normal** -- equal weights across all tiers
- **Company Survival** -- company in distress, environment stable
- **Modified Survival** -- healthy company in a toxic macro environment
- **Extreme Survival** -- both company and country in crisis

---

## Ethical Filters

Four investment quality filters inspired by Islamic finance principles and universal risk management:

1. **Purchasing Power** -- Does the investment beat inflation? A stock that rises 5% while inflation is 7% actually lost you money.
2. **Solvency** -- Is the company dangerously leveraged? Debt-to-equity above 3.0 means vulnerability to bankruptcy in a recession.
3. **Speculation (Gharar)** -- Is this investing or gambling? Extreme volatility means predictions are unreliable regardless of the model.
4. **Cash is King** -- Does the company generate real cash? "Profit is an opinion, but cash is a fact."

---

## Mathematical Models (25+)

| Category | Models |
|----------|--------|
| Regime Detection | Hidden Markov Model, Gaussian Mixture, PELT structural breaks, Bayesian change point |
| Forecasting | Adaptive Kalman Filter, GARCH volatility, Vector Autoregression, LSTM neural network |
| Deep Learning | Temporal Fusion Transformer (attention-based, multi-horizon) |
| Tree Ensembles | Random Forest, XGBoost, Gradient Boosting |
| Causality | Granger Causality, Transfer Entropy, Copula tail dependency |
| Uncertainty | Conformal Prediction (distribution-free), MC Dropout, Regime-Aware Monte Carlo |
| Explainability | Feature attribution (per-prediction), Sobol global sensitivity |
| Pattern Recognition | Candlestick detector, Fourier/Wavelet cycle decomposition, DTW historical analogs |
| Optimization | Genetic Algorithm (ensemble weight evolution), PID adaptive learning |
| State Estimation | Particle Filter (Sequential Monte Carlo) for non-linear tracking |

Every model has a fallback -- if a dependency is missing or data is insufficient, the pipeline degrades gracefully and continues with the next available method.

---

## Output Files

After the pipeline finishes, results are in the `cache/` folder:

```
cache/
  company_profile.json          # Complete analysis data (JSON)
  estimation_coverage.json      # Variable coverage statistics
  report/
    basic_report.md             # Quick screening report (5 sections)
    pro_report.md               # Peer + macro report (13 sections)
    premium_report.md           # Full institutional report (22 sections)
    premium_report.pdf          # PDF version (if --pdf flag used)
    charts/
      price_history.png         # 2-year price chart with regime shading
      survival_timeline.png     # Distress period visualization
      hierarchy_weights.png     # Risk allocation over time
      volatility.png            # Realized volatility chart
      financial_health.png      # Composite health score chart
      predicted_ohlc_week.png   # Predicted candlestick (next week)
      predicted_ohlc_month.png  # Predicted candlestick (next month)
```

---

## Architecture

```
main.py                     # CLI entry point (8-step pipeline)
run.py                      # Interactive terminal launcher
operator1/
  clients/
    pit_registry.py         # Market + Macro API registry (10 markets, 10 macro APIs)
    pit_base.py             # PITClient protocol (filing_date + report_date)
    sec_edgar.py            # SEC EDGAR client (US)
    esef.py                 # ESEF/XBRL client (EU/FR/DE)
    companies_house.py      # Companies House client (UK)
    edinet.py               # EDINET client (Japan)
    dart.py                 # DART client (South Korea)
    mops.py                 # MOPS client (Taiwan)
    cvm.py                  # CVM client (Brazil)
    cmf.py                  # CMF client (Chile)
    macro_client.py         # Unified macro data fetcher (10 central bank APIs)
    canonical_translator.py # Normalizes all filing formats into one schema
    gemini.py               # Gemini AI (report generation only)
  estimation/
    estimator.py            # Two-pass missing data inference (accounting + ML)
    vae_imputer.py          # Variational Autoencoder imputation option
  features/
    derived_variables.py    # 25+ financial ratios with rolling 4Q TTM
    linked_aggregates.py    # Peer group aggregation with temporal masking
    macro_alignment.py      # Macro indicator alignment to daily cache
    macro_quadrant.py       # GDP/inflation quadrant classification
  analysis/
    survival_mode.py        # Country + company distress detection
    hierarchy_weights.py    # Dynamic tier weight computation
    fuzzy_protection.py     # Government protection scoring
    ethical_filters.py      # 4 investment quality filters
    economic_planes.py      # 5-plane economic sector classification
  models/
    regime_detector.py      # HMM + GMM + PELT + BCP
    forecasting.py          # Kalman + GARCH + VAR + LSTM + trees + baseline
    monte_carlo.py          # Regime-aware simulations with importance sampling
    prediction_aggregator.py# Inverse-RMSE ensemble weighting
    financial_health.py     # 5-tier scoring + Altman Z + Beneish M
    transformer_forecaster.py # Temporal Fusion Transformer
    conformal.py            # Distribution-free prediction intervals
    copula.py               # Tail dependency analysis
    causality.py            # Transfer entropy
    granger_causality.py    # Granger causality + feature pruning
    cycle_decomposition.py  # Fourier + wavelet cycle analysis
    dtw_analogs.py          # Dynamic Time Warping historical analogs
    explainability.py       # SHAP feature attribution
    sensitivity.py          # Sobol global sensitivity
    genetic_optimizer.py    # GA ensemble weight evolution
    particle_filter.py      # Sequential Monte Carlo state estimation
    pattern_detector.py     # Candlestick pattern recognition
    ohlc_predictor.py       # Predicted OHLC candlestick series
    graph_risk.py           # Supply chain network risk
    game_theory.py          # Competitive dynamics analysis
    model_synergies.py      # Cross-model synergies + plane-aware weighting
    regime_mixer.py         # Dual regime classification
    pid_controller.py       # PID adaptive learning
  quality/
    data_quality.py         # Look-ahead bias detection (hard fail on violation)
    data_reconciliation.py  # Field normalization, filing date validation
  report/
    profile_builder.py      # Company profile JSON assembly
    report_generator.py     # 3-tier Markdown/PDF report generation
config/
  global_config.yml         # HTTP settings, rate limits, estimation method
  survival_hierarchy.yml    # 5-tier weights per regime
  economic_planes.yml       # Sector-to-plane mapping
  canonical_fields.yml      # Cross-market field normalization rules
  country_protection_rules.yml  # Government protection thresholds
  country_survival_rules.yml    # Country distress thresholds
```

---

## API Keys (All Optional)

| Key | Purpose | Required? | Where to get it |
|-----|---------|-----------|-----------------|
| `GEMINI_API_KEY` | AI report generation | Optional | [ai.google.dev](https://ai.google.dev/) |
| `FRED_API_KEY` | US macro data (FRED) | Optional | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `COMPANIES_HOUSE_API_KEY` | UK market data | Optional | [developer.company-information.service.gov.uk](https://developer.company-information.service.gov.uk/) |
| `DART_API_KEY` | South Korea market data | Optional | [opendart.fss.or.kr](https://opendart.fss.or.kr/) |

Without any keys, the pipeline works for most markets (SEC EDGAR, ESEF, EDINET, BCB, etc.).

---

## Running the App

For detailed setup and usage instructions, see [HOW-TO-RUN.md](HOW-TO-RUN.md).

---

## License

This project is provided as-is for educational and research purposes.
