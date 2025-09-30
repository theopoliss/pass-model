# Alternative Data Sources for Pass Prediction Model

Since FBRef is blocking automated scraping (403 errors), here are alternative data sources:

## 1. **Free APIs That Work**

### **StatsBomb Open Data** (Already in our pipeline!)
- Free data for certain competitions
- High quality event data
- Includes passes, positions, formations
- No rate limiting
```bash
# This already works:
python train.py --collect-data
```

### **Football-Data.org API** (Free tier)
```python
# Free API with 10 requests/minute
# Includes match results, not detailed player stats
API_KEY = "get_from_football-data.org"
```

### **OpenLigaDB** (German leagues)
- Completely free, no authentication
- Good for Bundesliga data
- REST API with match and player data

## 2. **Paid APIs (Better for Production)**

### **API-Football** ($0-$600/month)
- 100 requests/day free tier
- Detailed player statistics
- Pass completion rates available
- Coverage: All major leagues

### **SportMonks** ($0-$150/month)
- Free tier: 3000 requests/month
- Player match statistics
- Pass data included
- Good documentation

### **Opta** (Enterprise)
- Industry standard
- Most detailed data
- Used by major broadcasters
- Contact for pricing

## 3. **Web Scraping Alternatives**

### **Understat.com**
- Less aggressive blocking than FBRef
- xG and shot data
- Player statistics

### **WhoScored.com**
- Uses Opta data
- Detailed match stats
- Requires Selenium (JavaScript heavy)

### **Transfermarkt.com**
- Player profiles and stats
- Match lineups
- More tolerant of scraping

## 4. **Manual Data Collection**

### **Browser Extensions**
- Table Capture (Chrome)
- Web Scraper (Chrome)
- Data Miner (Chrome)

### **Copy & Paste Method**
1. Open match pages in browser
2. Select tables
3. Paste into Excel/CSV
4. Save in our data folder

## 5. **Hybrid Approach (Recommended)**

1. **Use StatsBomb for historical data** (working)
2. **Use API-Football free tier for recent matches**
3. **Manual collection for specific matches**
4. **Generate synthetic data for testing** (working)

## Implementation Priority

1. ✅ **StatsBomb** - Already working
2. ✅ **Synthetic data** - Already working
3. 🔄 **API-Football integration** - Can implement
4. 🔄 **Understat scraper** - Less blocking
5. 🔄 **Manual CSV import** - Simple to add

## Quick Start with What Works Now

```bash
# Option 1: Use StatsBomb data (historical)
python train.py --collect-data

# Option 2: Use synthetic data (testing)
python generate_fbref_data.py
python train_fbref.py --use-cached

# Option 3: Manual data
# 1. Export tables from FBRef in browser
# 2. Save as CSV in data/raw/
# 3. Process with our pipeline
```

## Next Steps

1. **For immediate results**: Use the synthetic data generator
2. **For real data**: Set up API-Football free account
3. **For production**: Consider SportMonks or paid API
4. **For research**: StatsBomb open data is sufficient