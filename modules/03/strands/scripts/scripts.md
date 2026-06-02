<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
Sample code, non-production. See README.md for full disclaimer.
-->

# Module 03 - Strands Version Test Commands

## Launch the Strands DataViz Agent

```bash
uv run modules/03/strands/dataviz.py -m "global.anthropic.claude-sonnet-4-6"
```

## Test Histogram (paste into agent, then type END)

```
"CustomerID,Age,PurchaseAmount
1,25,150
2,32,200
3,28,175
4,45,300
5,38,250
6,52,400
7,29,180
8,41,320
9,35,220
10,48,380
11,26,160
12,55,420
13,31,190
14,44,310
15,37,240
16,50,390
17,27,170
18,42,330
19,33,210
20,49,370"

Create a histogram showing the distribution of customer ages with 5 bins
END
```

## Test Bar Chart (paste into agent, then type END)

```
"Month,Sales
Jan,1000
Feb,1200
Mar,800
Apr,1500
May,1100
Jun,1300"

Create a bar chart showing sales by month
END
```

## Test Scatter Plot (paste into agent, then type END)

```
"Product,Price,Category,Rating
Laptop,999,Electronics,4.5
Phone,699,Electronics,4.2
Desk,299,Furniture,4.0
Chair,199,Furniture,4.3
Tablet,399,Electronics,4.1"

Create a scatter plot showing the relationship between price and rating, colored by category
END
```

## Test Data Analysis (paste into agent, then type END)

```
"Region,Q1,Q2,Q3,Q4
North,1000,1200,1100,1400
South,800,900,1000,1100
East,1100,1300,1200,1500
West,900,1000,1100,1200"

Analyze this data and recommend the best visualization
END
```

## Exit the Agent

```
quit
```