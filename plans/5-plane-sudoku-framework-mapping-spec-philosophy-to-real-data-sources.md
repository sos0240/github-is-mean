# 5-Plane Sudoku Framework: Mapping Spec Philosophy to Real Data Sources

## The Core Question

The PDF describes a "5-plane Sudoku world model" with five economic planes:
1. **Supply** -- raw material providers, commodity producers
2. **Manufacturing** -- companies that transform inputs into products
3. **Consumption** -- retailers, consumer-facing companies
4. **Logistics** -- shipping, distribution, warehousing
5. **Financial Services** -- banks, insurers, investment firms, and general services

The code currently classifies companies by `sector` and `industry` from the equity provider (FMP/Eulerpool/EOD), but does NOT map them to these five economic planes. The user wants the code to "compromise" -- interpret what the framework needs from what the data sources actually provide, and match them correctly.

## What the Framework Actually Demands (vs What We Have)

### What the 5-plane model means operationally

The planes are not just labels. They determine **how linked variables flow between companies**:

- A company in the **Supply** plane (e.g., a steel producer) has its linked variables dominated by commodity prices, raw material availability, and upstream macro factors.
- A company in the **Manufacturing** plane (e.g., an auto manufacturer) has linked variables from both Supply (input costs) and Consumption (demand signals).
- A company in the **Consumption** plane (e.g., a retailer) is driven by consumer sentiment, GDP growth, and downstream logistics capacity.
- A company in the **Logistics** plane (e.g., a shipping company) connects Supply to Manufacturing to Consumption -- it is a bridge.
- A company in the **Financial Services** plane (e.g., a bank) sits across all planes as a funding/credit layer.

### What the data sources provide

| Source | What it gives us | Relevant plane signal |
|--------|-----------------|----------------------|
| **FMP/Eulerpool profile** | `sector`, `industry`, `sub_industry` | These map to GICS sectors (Technology, Healthcare, Financials, etc.) which overlap but do not map 1:1 to the 5 planes |
| **Gemini entity discovery** | `relationship_type`: competitors, suppliers, customers, financial_institutions, logistics | These **directly correspond** to the 5-plane model |
| **World Bank macro** | GDP, inflation, unemployment, FX rates, current account | Country-level economic health that affects all planes differently |
| **Linked entity caches** | Per-entity financials for each relationship group | Already grouped by relationship type |

### The key insight: Gemini's relationship types ARE the plane connections

The Gemini discovery already categorizes linked entities into:
- `competitors` -- same plane (horizontal)
- `suppliers` -- upstream plane (Supply -> Manufacturing)
- `customers` -- downstream plane (Manufacturing -> Consumption)
- `financial_institutions` -- Financial Services plane (cross-cutting)
- `logistics` -- Logistics plane (cross-cutting)

**The code already has the plane structure -- it just doesn't label it as such.**

## Proposed Design: Plane-Aware Variable Classification

### Step 1: Add a plane classification config

Create a mapping from GICS sectors to the 5 economic planes in `config/economic_planes.yml`:

```yaml
# Maps GICS sector/industry to the 5 economic planes from the spec.
# A company can participate in multiple planes (e.g., Amazon is both
# Consumption and Logistics). The primary plane is used for
# hierarchy weighting; secondary planes inform linked variable routing.

planes:
  supply:
    label: "Supply & Raw Materials"
    sectors:
      - "Energy"
      - "Materials"
      - "Mining"
    industries:
      - "Oil & Gas"
      - "Metals & Mining"
      - "Chemicals"
      - "Agricultural Products"
      - "Paper & Forest Products"
    
  manufacturing:
    label: "Manufacturing & Production"
    sectors:
      - "Industrials"
      - "Technology"
      - "Healthcare"
    industries:
      - "Automobiles"
      - "Semiconductor"
      - "Aerospace & Defense"
      - "Pharmaceuticals"
      - "Electronic Equipment"
      - "Machinery"
    
  consumption:
    label: "Consumption & Retail"
    sectors:
      - "Consumer Discretionary"
      - "Consumer Staples"
    industries:
      - "Retail"
      - "Food & Beverage"
      - "Restaurants"
      - "Household Products"
      - "Apparel"
      - "Media & Entertainment"
    
  logistics:
    label: "Logistics & Distribution"
    sectors:
      - "Transportation"
    industries:
      - "Airlines"
      - "Shipping"
      - "Trucking"
      - "Railroads"
      - "Warehousing"
      - "Delivery Services"
    
  financial_services:
    label: "Financial Services"
    sectors:
      - "Financials"
      - "Real Estate"
    industries:
      - "Banks"
      - "Insurance"
      - "Asset Management"
      - "Exchanges"
      - "Fintech"

# Default plane when sector/industry doesn't match any above
default_plane: "manufacturing"

# Relationship types map to inter-plane connections
relationship_to_plane:
  competitors: "same_plane"
  suppliers: "supply"
  customers: "consumption"
  financial_institutions: "financial_services"
  logistics: "logistics"
```

### Step 2: Classify the target company into a plane at verification time

After `verify_identifiers()` resolves the company profile, classify it:

```python
def classify_economic_plane(sector: str, industry: str) -> dict:
    """Classify a company into the 5-plane economic model.
    
    Returns:
        {
            "primary_plane": "manufacturing",
            "secondary_planes": ["consumption"],
            "plane_label": "Manufacturing & Production",
        }
    """
```

This function reads `config/economic_planes.yml` and matches the company's sector/industry. Store the result in the profile metadata.

### Step 3: Use plane classification to weight linked variables

The current linked aggregates module (`linked_aggregates.py`) already computes per-group aggregates (competitors_avg_X, supply_chain_avg_X, etc.). The plane model adds meaning:

- **Same-plane competitors**: These are the most directly relevant linked variables. Their metrics (return, volatility, debt levels) directly compete with and influence the target.
- **Upstream suppliers (Supply plane)**: Their stress signals (cash flow deterioration, supply disruption) propagate forward to the target's input costs.
- **Downstream customers (Consumption plane)**: Their demand signals (revenue growth, consumer sentiment) propagate backward to the target's revenue.
- **Financial services plane**: Banking stress affects the target's cost of capital and refinancing ability.
- **Logistics plane**: Disruptions here affect both supply and demand chains.

The plane classification should be used to:
1. **Weight linked variable importance** in the causal network (Granger/TE pruning should prefer intra-plane and adjacent-plane links)
2. **Inform the report narrative** -- instead of just saying "competitors' average return was X%", the report should contextualize: "Within the Manufacturing plane, peer companies showed..."
3. **Feed into survival analysis** -- a company in the Financial Services plane getting country protection should be weighted differently than one in Consumption

### Step 4: Add plane context to the company profile and report

The profile builder should include:
```json
{
  "economic_plane": {
    "primary": "manufacturing",
    "secondary": ["consumption"],
    "label": "Manufacturing & Production",
    "inter_plane_connections": {
      "supply": {"n_entities": 3, "avg_health": 72},
      "consumption": {"n_entities": 5, "avg_health": 65},
      "financial_services": {"n_entities": 2, "avg_health": 80},
      "logistics": {"n_entities": 1, "avg_health": 55}
    }
  }
}
```

The report should have a "Supply Chain & Economic Position" section that explains where the company sits in the economic model.

## How This Maps to FMP-Only vs Eulerpool/EOD+FMP Modes

### FMP-Only Mode
- **Profile**: FMP provides `sector` and `industry` -- sufficient for plane classification
- **Linked entities**: Still discovered via Gemini (which provides relationship types mapping to planes)
- **Statements**: FMP provides the same financial variables as Eulerpool
- **OHLCV**: FMP provides historical prices with explicit date range (we already fixed this)

**No gap.** FMP provides everything needed for plane classification.

### Eulerpool/EOD + FMP Mode
- **Profile**: Eulerpool/EOD provides `sector`, `industry`, `sub_industry` -- richer classification possible
- **Linked entities**: Same Gemini discovery + Eulerpool peers/supply-chain endpoints provide additional relationship data
- **Statements**: Eulerpool may provide more granular fields (sub-industry level)
- **OHLCV**: FMP as authoritative source

**Slight advantage.** Eulerpool's sub_industry field and peers/supply-chain endpoints can enrich the plane classification and inter-plane connections.

## Implementation Priority

This is a **medium-priority enhancement** that adds analytical depth but doesn't block the pipeline from running. The current code already has the data -- it just needs a classification layer on top.

### Recommended implementation order:
1. Create `config/economic_planes.yml` with the sector-to-plane mapping
2. Add `classify_economic_plane()` function in a new file or in `config_loader.py`
3. Store plane classification in profile metadata (1-line addition to `main.py`)
4. Add "Economic Position" section to the report generator (uses plane data in narrative)
5. Optionally: weight linked variable importance by plane adjacency in `model_synergies.py`

### What NOT to change:
- The 5-tier survival hierarchy stays as-is (it's about financial health, not economic planes)
- The derived variables stay as-is (they're company-internal metrics)
- The temporal models stay as-is (they learn from the data regardless of plane label)
- The Sudoku estimation stays as-is (it infers missing values from available ones)

The planes are a **classification and routing layer**, not a replacement for the existing financial analysis.
