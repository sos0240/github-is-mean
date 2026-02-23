# Temporal Relationships and Portfolio Awareness Design

## Problem 1: Linked Entities Are Temporal, Not Static

### The issue

The current code treats linked entities (suppliers, customers, competitors, etc.) as if they exist for the entire 2-year window. In reality:

- A supplier contract might have started 8 months ago and ended 3 months ago
- A competitor might have merged with another company 1 year into the window
- A customer might have switched to a different vendor mid-period
- A financial institution might have sold its position

When computing linked variable aggregates (e.g., `competitors_avg_return_21d`), including an entity that wasn't actually a competitor during that period pollutes the signal.

### Current state

- Gemini discovers relationships at a single point in time (now)
- All linked entity data is fetched for the full 2-year window
- Linked aggregates treat every discovered entity as present for all 500 trading days
- No temporal validity window per relationship

### Proposed design: Relationship validity windows

#### A. Discovery with temporal awareness

Modify the Gemini discovery prompt to ask for **temporal context**:

```
For each entity, also estimate:
- "relationship_start": approximate start of relationship ("2024-Q1", "ongoing", "unknown")
- "relationship_end": approximate end ("current", "2025-Q3", "unknown")
- "relationship_stability": "stable" / "volatile" / "new"
```

This doesn't require extra API calls -- Gemini already knows industry context. It's a prompt enhancement.

#### B. Eulerpool/FMP data provides natural validity signals

For each linked entity, we already fetch their financial data. We can detect natural relationship boundaries:

- **Supply chain endpoint** (Eulerpool): If it returns a list of current suppliers/customers, those are "as of now" relationships
- **Peers endpoint**: Changes in peer lists over time indicate competitive landscape shifts
- **Revenue correlation**: If a supplier's revenue stops correlating with the target's COGS, the relationship may have ended

#### C. Null-masking linked variables by validity window

For each relationship `(target, linked_entity, relationship_type)`, store:

```python
@dataclass
class TemporalRelationship:
    target_isin: str
    linked_isin: str
    relationship_type: str  # competitors, suppliers, etc.
    valid_from: date | None  # None = beginning of cache
    valid_to: date | None  # None = still active
    confidence: float  # 0-1
    source: str  # "gemini", "eulerpool_peers", "eulerpool_supply_chain"
```

When computing linked aggregates for day `t`:
- Only include entities where `valid_from <= t <= valid_to`
- Entities outside their validity window contribute `null` (not zero)
- The `is_missing` flag correctly reflects temporal gaps

#### D. Efficient API call strategy

To minimize API calls while capturing temporal changes:

1. **Single Gemini call** for relationship discovery (already the case)
2. **Single data fetch per entity** for the full 2-year window (already the case)
3. **Post-fetch temporal masking** -- apply validity windows after data is fetched, not during fetching
4. **Correlation-based validation** (optional): After building caches, compute rolling correlation between target and each linked entity's key variables. If correlation drops below threshold for a sustained period, flag the relationship as potentially ended.

No additional API calls are needed. The temporal awareness comes from:
- Gemini's contextual knowledge (prompt enhancement)
- Statistical analysis of already-fetched data (correlation-based)

---

## Problem 2: Portfolio Relationships (Missing from Code)

### What the PDF says

Section A of the Core Idea PDF lists "portfolio relationships" as one of the linked variable connection types. This is currently not implemented at all.

### Two dimensions of portfolio awareness

#### A. Institutional Ownership Overlap (Systemic Risk)

**What it is**: When large institutional investors (BlackRock, Vanguard, sovereign wealth funds) hold significant positions in both the target company and its linked entities, their rebalancing creates correlated price movements. This is a contagion channel that isn't visible from fundamentals alone.

**Data source**: FMP provides institutional ownership data:
- `GET /institutional-holder/{symbol}` -- top institutional holders
- `GET /mutual-fund-holder/{symbol}` -- mutual fund holders
- These are available in FMP's API (documented at financialmodelingprep.com)

**Linked variables to derive**:
```
portfolio_overlap_score: float  # 0-1, how many shared large holders
top_shared_holders: list  # names + % held in both target and linked entity
portfolio_concentration_risk: float  # Herfindahl of shared institutional ownership
portfolio_rebalancing_pressure: float  # derived from recent 13F changes
```

**When it matters**:
- In market stress, institutional holders sell across their entire portfolio
- A company with high overlap with a distressed entity faces contagion selling
- This directly feeds the Graph Risk module (contagion probability)

**Implementation**:
1. Add `get_institutional_holders(symbol)` to `FMPClient` and `FMPFullClient`
2. Compute ownership overlap matrix between target and all linked entities
3. Inject `portfolio_overlap_score` as a linked variable in the cache
4. Feed into Monte Carlo as a correlation factor (copula tail dependency)

#### B. User's Own Portfolio Context

**What it is**: The user tells Operator 1 what other stocks they hold. The analysis then factors in:
- **Concentration risk**: If the user already holds 30% in the same sector, adding this company increases risk
- **Cross-correlation**: How this company's returns correlate with the user's existing holdings
- **Diversification benefit**: Whether this company adds diversification or piles on the same risk

**Data source**: User input (a list of symbols and weights)

**Implementation**:
1. Add optional `--portfolio` flag to `main.py` and `run.py`:
   ```
   --portfolio "AAPL:30,MSFT:25,GOOGL:20,AMZN:15,TSLA:10"
   ```
   Or load from a file: `--portfolio-file portfolio.csv`

2. For each holding in the user's portfolio:
   - Fetch 2-year OHLCV from FMP (batch endpoint: `GET /batch-quote`)
   - Compute correlation matrix with the target company
   - Compute portfolio beta and marginal risk contribution

3. Add to the company profile:
   ```json
   {
     "portfolio_context": {
       "user_holdings": 5,
       "correlation_with_portfolio": 0.72,
       "marginal_var_contribution": 0.08,
       "diversification_benefit": "low",
       "sector_concentration": 0.45,
       "recommendation_adjustment": "CAUTION - high sector overlap"
     }
   }
   ```

4. The report generator adds a "Portfolio Fit" section:
   - "Adding this position would increase your sector concentration to 45%"
   - "The correlation with your existing portfolio is 0.72 -- limited diversification benefit"
   - "Marginal Value-at-Risk contribution: 8% of portfolio risk"

**API call budget**: Each portfolio holding needs 1 FMP OHLCV call (~5 calls for a 5-stock portfolio). This is efficient.

---

## Combined Architecture

```
User Input
    |
    v
[Mode Selection: FMP-only / Eulerpool+FMP / EOD+FMP]
    |
    v
[Company Resolution: symbol/ISIN -> profile -> sector -> economic plane]
    |
    +---> [Optional: User portfolio input]
    |
    v
[Gemini Discovery: relationships WITH temporal context]
    |
    v
[Data Extraction: target + linked entities + macro + institutional ownership]
    |
    v
[Cache Building: as-of alignment + temporal relationship masking]
    |
    v
[Derived Variables: decision vars + linked aggregates + portfolio overlap]
    |
    v
[Temporal Models: plane-aware weighting + portfolio correlation]
    |
    v
[Report: economic position + portfolio fit + temporal relationship narrative]
```

## Implementation Priority

| Feature | Priority | API Calls Needed | Lines Est. |
|---------|----------|-----------------|-----------|
| Temporal relationship masking (post-fetch) | High | 0 extra | ~150 |
| Gemini prompt enhancement for temporal context | High | 0 extra (same call) | ~30 |
| Institutional ownership overlap (FMP) | Medium | 1-2 per entity | ~250 |
| User portfolio input (CLI + analysis) | Medium | 1 per holding | ~300 |
| Portfolio Fit report section | Medium | 0 | ~100 |
| Correlation-based relationship validation | Low | 0 | ~100 |

### Recommended execution order:
1. Gemini prompt enhancement (temporal context) -- zero cost, immediate improvement
2. Temporal relationship masking in linked_aggregates.py -- data quality improvement
3. Institutional ownership data fetch from FMP -- new linked variable
4. User portfolio CLI input and correlation analysis
5. Portfolio Fit report section
