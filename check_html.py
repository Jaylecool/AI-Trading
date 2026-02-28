import requests

# Get the dashboard HTML
resp = requests.get('http://localhost:5000/')
html = resp.text

# Check if our new code is there
if 'Drawing' in html and 'data points on chart' in html:
    print("✓ Updated updateChart code is present")
else:
    print("✗ updateChart code not found or outdated")

if 'style="height: 600px' in html:
    print("✓ Inline height style is present")
else:
    print("✗ Inline height style not found")

if '#chartPlot' in html:
    print("✓ chartPlot element reference found")
else:
    print("✗ chartPlot element reference missing")

# Check for Plotly script
if 'plotly-latest.min.js' in html:
    print("✓ Plotly CDN script present")
else:
    print("✗ Plotly CDN script missing")

# Count occurrences of updateChart
count = html.count('updateChart')
print(f"updateChart mentioned {count} times")

# Find the section with updateChart
if 'console.log' in html and 'updateDashboard' in html:
    print("✓ Console logging code present")
else:
    print("✗ Console logging code not found")
