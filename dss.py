import pandas as pd
import numpy as np

# Simulated dataset
data = {
    'School': ['A', 'B', 'C', 'D', 'E'],
    'Test Scores': [65, 70, 80, 90, 85],
    'Graduation Rate': [0.8, 0.85, 0.9, 0.95, 0.92],
    'Attendance Rate': [0.9, 0.95, 0.85, 0.8, 0.88],
    'Funding': [1_000_000, 800_000, 1_200_000, 1_500_000, 1_100_000],
    'Staffing': [30, 25, 35, 40, 28],
    'Materials': [0.8, 0.7, 0.9, 1.0, 0.85],
    'Low-income': [0.5, 0.4, 0.6, 0.7, 0.45],
    'Special Needs': [0.1, 0.15, 0.05, 0.2, 0.12],
    'Enrollment': [500, 450, 600, 700, 480],
    'Community Events': [10, 8, 12, 15, 9],
    'Fundraising': [50_000, 40_000, 60_000, 70_000, 55_000],
    'Volunteer Hours': [500, 400, 600, 700, 550],
    'Building Age': [20, 25, 15, 10, 18],
    'Condition': [0.8, 0.7, 0.9, 1.0, 0.85],
    'Tech Resources': [0.8, 0.7, 0.9, 1.0, 0.85]
}

df = pd.DataFrame(data)

# Criteria weights
weights = {
    'Performance Levels': 0.3,
    'Resource Deficits': 0.2,
    'Student Demographics': 0.2,
    'Community Engagement': 0.15,
    'School Infrastructure Quality': 0.15
}

# Normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Performance Levels Score
df['Test Scores Norm'] = normalize(df['Test Scores'])
df['Performance Score'] = 0.6 * df['Test Scores Norm'] + 0.3 * df['Graduation Rate'] + 0.1 * df['Attendance Rate']

# Resource Deficits Score
df['Funding Norm'] = normalize(df['Funding'])
df['Staffing Norm'] = normalize(df['Staffing'])
df['Materials Norm'] = df['Materials']
df['Resource Score'] = 0.5 * df['Funding Norm'] + 0.3 * df['Staffing Norm'] + 0.2 * df['Materials Norm']

# Student Demographics Score
df['Demographics Score'] = 0.5 * df['Low-income'] + 0.3 * df['Special Needs'] + 0.2 * normalize(df['Enrollment'])

# Community Engagement Score
df['Community Score'] = 0.4 * normalize(df['Community Events']) + 0.3 * normalize(df['Fundraising']) + 0.3 * normalize(df['Volunteer Hours'])

# School Infrastructure Quality Score
df['Infrastructure Score'] = 0.4 * normalize(df['Building Age']) + 0.3 * df['Condition'] + 0.3 * df['Tech Resources']

# Overall Score
df['Overall Score'] = (weights['Performance Levels'] * df['Performance Score'] +
                       weights['Resource Deficits'] * df['Resource Score'] +
                       weights['Student Demographics'] * df['Demographics Score'] +
                       weights['Community Engagement'] * df['Community Score'] +
                       weights['School Infrastructure Quality'] * df['Infrastructure Score'])

# Rank schools by overall score
df['Rank'] = df['Overall Score'].rank(ascending=False)

# Display the ranked list
ranked_schools = df.sort_values(by='Rank')
print(ranked_schools[['School', 'Overall Score', 'Rank']])
