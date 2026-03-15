# Task 5.1: Dashboard Design Specification

**Task Period:** Feb 24 – Feb 28  
**Status:** Design Phase Complete  
**Date:** February 2026

---

## Executive Summary

This document defines the complete design specifications for the TradingPro Dashboard, a comprehensive real-time trading and analytics platform. The dashboard integrates predictions, trade execution history, portfolio metrics, and risk monitoring into a unified, intuitive interface.

---

## 1. DESIGN REQUIREMENTS

### 1.1 Core Information Architecture

The dashboard must display the following information categories:

#### **1. Stock Prediction Panel**
- Current stock price (AAPL)
- Forecast price for next trading day
- Forecast direction (bullish/bearish)
- Confidence level (0-100%)
- Confidence interval (min/max range)
- Expected movement ($)

#### **2. Technical Analysis Chart**
- Historical price action (daily bars)
- Technical indicators overlaid:
  - SMA (20-day, 50-day moving averages)
  - EMA (12-day exponential moving average)
  - Predicted price overlay
- Time periods: 1M, 3M, 6M, 1Y views
- Interactive zoom/pan capabilities

#### **3. Trading Signal Display**
- Signal type: BULLISH, BEARISH, NEUTRAL, MIXED
- Signal strength: Visual color coding
- Latest signal generation time
- Signal reason/explanation

#### **4. Trade History Log**
- Date of trade execution
- Action type: BUY / SELL
- Execution price
- Stop-loss level
- Take-profit target
- Trade status: OPEN / CLOSED
- P&L if closed
- Position duration

#### **5. Portfolio Metrics Dashboard**
- Current total balance ($)
- Total unrealized gains/losses ($)
- ROI percentage
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Annual volatility
- Win rate
- Profit factor

#### **6. Risk Indicators Panel**
- Portfolio heat score (0-100, color-coded)
- Position concentration (pie chart)
- Risk level gauge (LOW/MODERATE/HIGH/CRITICAL)
- Asset exposure breakdown
- Sector exposure breakdown
- VaR (Value at Risk) estimate

#### **7. Alerts and Notifications**
- Real-time trading signals
- Risk threshold breaches
- Stop-loss/take-profit triggered
- Portfolio milestone alerts
- System status messages

---

## 2. INFORMATION HIERARCHY

### Primary Content Areas (Visible at Load)

```
┌─────────────────────────────────────────────────────────────┐
│  NAVIGATION BAR: Dashboard | Simulator | Reports | Settings  │
└─────────────────────────────────────────────────────────────┘
┌──────────────────────┬──────────────────────────────────────┐
│                      │                                       │
│  PRICE CHART         │  PREDICTION PANEL                    │
│  (Large, central)    │  • Forecast Price: $184.15           │
│                      │  • Confidence: 76%                   │
│  AAPL: $179.66       │  • Signal: BULLISH                   │
│  +1.32%              │  • Movement: +$4.49 (2.50%)          │
│                      │                                       │
│  SMA(20), EMA(12)    │  SIGNAL DISPLAY                      │
│  Predicted line      │  • Type: RSI Oversold                │
│                      │  • Strength: 78%                     │
└──────────────────────┴──────────────────────────────────────┘
┌──────────────────────┬──────────────────────────────────────┐
│  PORTFOLIO METRICS   │                                       │
│  • Balance: $125.5K  │  TRADE HISTORY (Scrollable Table)    │
│  • Gains: +$18.4K    │  ┌─────────────────────────────────┐ │
│  • ROI: 17.21%       │  │ Date  | Action | Price | SL | TP│ │
│  • Sharpe: 1.85      │  ├─────────────────────────────────┤ │
│  • Max DD: 8.32%     │  │ 2/18  | Buy    | $188 | ... | ..│ │
│                      │  │ 2/17  | Sell   | $186 | --- | -- │
│  RISK INDICATORS     │  │ 2/15  | Buy    | $183 | ... | ..│ │
│  • Heat: 35/100      │  └─────────────────────────────────┘ │
│  • Level: MODERATE   │                                       │
└──────────────────────┴──────────────────────────────────────┘
```

---

## 3. VISUALIZATION SPECIFICATIONS

### 3.1 Chart Types

| Component | Chart Type | Library | Notes |
|-----------|-----------|---------|-------|
| Price History | Line/Candlestick | Plotly.js / Chart.js | Interactive, zoom-enabled |
| Predictions Overlay | Line with Confidence Bands | Plotly.js | Shaded confidence interval |
| Portfolio P&L | Area Chart | Chart.js | Cumulative gain/loss over time |
| Asset Allocation | Pie Chart | Chart.js | Percentage breakdown by stock |
| Performance Comparison | Bar Chart | Plotly.js | Strategy comparison |
| Risk Indicators | Gauge / Speedometer | Custom SVG | Color-coded (Green→Red) |
| Trade Distribution | Histogram | Chart.js | Win/loss distribution |

### 3.2 Color Coding System

**Signal Direction:**
- 🟢 **BULLISH**: #00D084 (Green)
- 🔴 **BEARISH**: #FF3B30 (Red)
- 🟡 **NEUTRAL**: #FFCC33 (Yellow)
- 🟣 **MIXED**: #9C27B0 (Purple)

**Risk Levels:**
- 🟢 LOW: #00D084 (Heat Score 0-25)
- 🟡 MODERATE: #FFCC33 (Heat Score 25-50)
- 🟠 HIGH: #FF9500 (Heat Score 50-75)
- 🔴 CRITICAL: #FF3B30 (Heat Score 75-100)

**Performance:**
- 🟢 Profit: #00D084
- 🔴 Loss: #FF3B30
- 🔵 Neutral: #007AFF

---

## 4. LAYOUT ARCHITECTURE

### 4.1 Desktop Layout (1920x1080)

**Grid System:** 12-column, responsive flexbox

```
Header (Full Width)
├── Logo/Brand (2 cols)
├── Navigation Menu (7 cols)
└── User Menu (3 cols)

Main Content (Full Width)
├── Sidebar (2 cols, collapsible)
│   ├── Dashboard
│   ├── Simulator
│   ├── Reports
│   ├── Alerts
│   └── Settings
│
└── Content Area (10 cols)
    ├── Top Section (Price & Prediction) [12 cols]
    │   ├── Chart Panel [7 cols]
    │   └── Metrics Panel [5 cols]
    │
    ├── Middle Section (Portfolio & Risk) [12 cols]
    │   ├── Portfolio Metrics [4 cols]
    │   ├── Trade History [8 cols]
    │
    └── Bottom Section (Alerts) [12 cols]
        └── Alert Log [12 cols]
```

### 4.2 Tablet Layout (768x1024)

- Stacked layout for prediction panel
- Full-width charts
- Trade history scrollable
- Risk metrics compressed

### 4.3 Mobile Layout (375x667)

- Single column
- Collapsible sections
- Horizontal scroll for tables
- All metrics on dedicated screens/tabs

---

## 5. COMPONENT SPECIFICATIONS

### 5.1 Prediction Panel Component

**Dimensions:** 400px width (desktop)
**Data Points:**
- Forecast price (large, 32pt font)
- Confidence level (0-100%)
- Confidence interval (range)
- Expected movement ($, %)
- Direction indicator (arrow icon)
- Signal type
- Update timestamp

**Visual Elements:**
- Confidence bar (0-100 with color gradient)
- Directional arrow (↑ green for bullish, ↓ red for bearish)
- Color-coded background based on signal strength

### 5.2 Price Chart Component

**Dimensions:** Responsive (600-1000px width)
**Features:**
- Candlestick/Line toggle
- Time period selector (1M, 3M, 6M, 1Y)
- Technical indicators toggle:
  - SMA 20
  - SMA 50
  - EMA 12
  - Historical volatility
- Predicted price overlay (different color line)
- Interactive legend
- Hover tooltips with detailed price info
- Zoom and pan enabled

### 5.3 Portfolio Metrics Component

**Layout:** 4-column grid (desktop), responsive
**Metrics Displayed:**
- Current Balance (large, primary)
- Total Gains/Losses (with color)
- ROI (with trend icon)
- Sharpe Ratio
- Sortino Ratio
- Annual Volatility
- Max Drawdown
- Win Rate
- Profit Factor
- Trade Count
- Average Trade Duration

**Update Frequency:** Real-time (every minute) or on-demand

### 5.4 Trade History Table Component

**Columns:**
- Date (MM/DD/YYYY format)
- Action (BUY / SELL icons)
- Price ($, 2 decimals)
- Stop-Loss ($, 2 decimals)
- Take-Profit ($, 2 decimals)
- Status (OPEN / CLOSED / PENDING)
- P&L ($ and %, conditional color)
- Duration (days held)

**Features:**
- Sortable columns (date, price, P&L)
- Filterable by:
  - Date range picker
  - Action type (BUY/SELL)
  - Status
  - Symbol
- Pagination (10, 25, 50 rows per page)
- Export to CSV
- Hover row highlighting
- Expandable row details

### 5.5 Risk Indicators Component

**Elements:**
- **Heat Score Gauge** (0-100, color-coded)
  - Visual speedometer dial
  - Numeric display
  - Risk level label
- **Position Concentration Pie Chart**
  - Stock/ETF breakdown
  - Click-to-focus
- **Risk Level Indicator**
  - Color-coded label
  - Threshold status
- **Exposure Breakdown Bar Chart**
  - Single stock exposure (%)
  - Sector exposure (%)
  - Max allowed overlay

---

## 6. NAVIGATION STRUCTURE

### 6.1 Top Navigation

```
[TradingPro Logo]  [Dashboard]  [Simulator]  [Reports]  [Alerts]  [Settings]  [User ▼]
```

- Primary focus: Dashboard (active by default)
- Secondary views: Simulator (backtesting), Reports (analytics)
- Settings: User preferences, notification settings

### 6.2 Sidebar Navigation (Collapsible)

```
═ Dashboard
  ├─ Overview (default)
  ├─ Predictions
  ├─ Trade History
  └─ Alerts

═ Simulator
  ├─ Run Backtest
  ├─ Strategy Config
  └─ Results

═ Reports
  ├─ Performance
  ├─ Risk Analysis
  └─ Trade Analysis

═ Settings
  ├─ Preferences
  ├─ Notifications
  └─ Data Export
```

---

## 7. USABILITY PRINCIPLES

### 7.1 Design Goals

✓ **Clarity:** Information hierarchy is obvious  
✓ **Scannability:** Key metrics visible at a glance  
✓ **Interactivity:** Hover tooltips and drill-down capabilities  
✓ **Responsiveness:** Works on all screen sizes  
✓ **Performance:** Charts load in < 2 seconds  
✓ **Accessibility:** WCAG AA compliance, keyboard navigation  
✓ **Visual Consistency:** Unified design language, 8px grid system

### 7.2 Color Palette

| Element | Primary | Secondary | Tertiary |
|---------|---------|-----------|----------|
| Background | #0F1419 (Dark) | #1A1F26 | #2A3039 |
| Text | #FFFFFF (White) | #E4E6EB | #A0A9B8 |
| Accent | #007AFF (Blue) | #00D084 (Green) | #FF3B30 (Red) |
| Borders | #3A444F | #4A5568 | - |

### 7.3 Typography

- **Brand:** "SF Pro Display" (Apple design system)
- **UI:** "SF Pro Display" for headings, "SF Mono" for data
- **Size Scale:** 10px, 12px, 14px, 16px, 20px, 24px, 32px, 48px

### 7.4 Spacing System

- **Grid Unit:** 8px
- **Common Spacing:** 8px, 16px, 24px, 32px, 48px

---

## 8. RESPONSIVENESS BREAKPOINTS

| Breakpoint | Width | Layout |
|-----------|-------|--------|
| Mobile | < 600px | Single column, stacked |
| Tablet | 600px - 1024px | 2-column, collapsed nav |
| Desktop | 1024px - 1920px | Full layout, sidebar |
| Ultra-wide | > 1920px | 3-column layout optional |

---

## 9. DATA INTEGRATION POINTS

### 9.1 Data Sources

1. **Prediction Engine**
   - Source: `prediction_models/` (from Task 3)
   - Data: Next-day price forecast, confidence
   - Update: Daily or on-demand

2. **Backtest Results**
   - Source: `backtesting_engine.py` (Task 4.4)
   - Data: ROI, Sharpe, drawdown, trade history
   - Update: After each backtest run

3. **Real-time Market Data**
   - Source: External API (Alpha Vantage, IEX)
   - Data: Current price, volume, indicators
   - Update: Real-time (or 15-min delayed)

4. **Risk Management**
   - Source: `risk_management_enhanced.py` (Task 4.3)
   - Data: Heat scores, position sizing, alerts
   - Update: With each trade

### 9.2 API Endpoints Required

```
GET /api/predictions/current
  Response: {
    symbol: "AAPL",
    forecast_price: 184.15,
    confidence: 0.76,
    confidence_interval: [182.30, 186.00],
    direction: "BULLISH",
    expected_movement: 4.49
  }

GET /api/trades/history
  Response: {
    trades: [
      {
        date: "2026-02-18",
        action: "BUY",
        price: 188.45,
        stop_loss: 185.20,
        take_profit: 192.00,
        status: "OPEN"
      }, ...
    ]
  }

GET /api/portfolio/metrics
  Response: {
    balance: 125480,
    gains_losses: 18420,
    roi: 0.1721,
    sharpe_ratio: 1.85,
    max_drawdown: 0.0832
  }

GET /api/risk/indicators
  Response: {
    heat_score: 35,
    heat_level: "MODERATE",
    position_concentration: {...},
    exposure: {...}
  }
```

---

## 10. PERFORMANCE REQUIREMENTS

### 10.1 Load Times

- Initial Dashboard Load: < 2 seconds
- Chart Rendering: < 1 second
- Table Population: < 0.5 seconds (first 50 rows)
- Trade History Pagination: < 0.3 seconds
- Real-time Updates: < 100ms latency

### 10.2 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (Safari iOS 14+, Chrome Mobile)

### 10.3 Optimization Strategies

- Lazy loading for off-screen components
- Virtual scrolling for large trade lists
- Canvas rendering for charts
- Debounced updates during interactions
- Service worker caching for static assets

---

## 11. ACCESSIBILITY REQUIREMENTS (WCAG 2.1 AA)

- **Color Contrast:** 4.5:1 for text (Small), 3:1 for large text
- **Keyboard Navigation:** All functionality accessible via keyboard
- **ARIA Labels:** Proper labels for icons and interactive elements
- **Focus Indicators:** Visible focus states for all buttons
- **Error Messages:** Clear, descriptive error feedback
- **Responsive Text:** Min 16px on mobile, scalable to 200%

---

## 12. SECURITY & DATA CONSIDERATIONS

- SSL/TLS for all connections
- User authentication (optional for demo)
- Local data caching (IndexedDB) for performance
- No sensitive data in localStorage
- CORS headers properly configured
- Rate limiting on API calls

---

## 13. DESIGN ARTIFACTS

### 13.1 Wireframes Created

1. Desktop View - Full Dashboard
2. Desktop View - Prediction Detail
3. Tablet View - Stacked Layout
4. Mobile View - Single Column
5. Mobile View - Trade History Detail

### 13.2 Figma Mockup

- High-fidelity mockup with all components
- Interactive prototype for navigation
- Design system with reusable components
- Style guide with colors, typography, spacing

---

## 14. NEXT STEPS (IMPLEMENTATION)

### Task 5.2: Prediction Display Implementation
- Create React/Vue components for chart rendering
- Integrate Plotly.js for interactive charts
- Implement prediction overlay logic
- Add real-time update mechanism

### Task 5.3: Trade History & Portfolio Views
- Build trade history table component
- Implement sorting/filtering/pagination
- Create portfolio metrics dashboard
- Add risk indicator visualizations

### Task 5.4: Testing & Deployment
- Unit tests for each component
- Integration tests for data flow
- Usability testing with sample users
- Performance profiling and optimization
- Cross-browser and responsive testing

---

## Summary Checklist

- ✅ Information architecture defined
- ✅ Visualization types selected
- ✅ Color palette specified
- ✅ Layout blueprints created
- ✅ Navigation structure designed
- ✅ Responsiveness breakpoints defined
- ✅ Component specifications detailed
- ✅ API contracts defined
- ✅ Performance targets set
- ✅ Accessibility requirements listed
- ✅ Security considerations documented
- ✅ Design artifacts ready for implementation

**Next Task:** 5.2 Implement Prediction Display
