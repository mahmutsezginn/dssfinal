import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define weights for each metric
weights = {
    'Performance Levels': 0.3,
    'Resource Levels': 0.2,
    'Student Demographics': 0.2,
    'Community Engagement': 0.2,
    'School Infrastructure Quality': 0.1
}

def normalize(series):
    """Normalize a pandas Series using MinMaxScaler."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

def calculate_scores(df):
    """Calculate and normalize scores, and rank schools."""
    # Ensure columns are in correct format (lists as strings)
    for col in ['Test_Scores', 'Graduation_Rates', 'Attendance_Records']:
        # Remove extra characters and then convert to lists
        df[col] = df[col].astype(str).str.replace(r'[\[\]"“]','', regex=True) 
        df[col] = df[col].apply(lambda x: [int(i) for i in x.split(',') if i])

    # Normalize Performance Levels
    df['Normalized_Test_Scores'] = df['Test_Scores'].apply(lambda x: normalize(pd.Series(x)).mean())
    df['Normalized_Graduation_Rates'] = df['Graduation_Rates'].apply(lambda x: normalize(pd.Series(x)).mean())
    df['Normalized_Attendance_Records'] = df['Attendance_Records'].apply(lambda x: normalize(pd.Series(x)).mean())

    # Normalize other relevant fields
    for col in ['Funding', 'Staffing', 'Materials', 'Low_Income_Students', 
                'Special_Needs_Students', 'Enrollment_Numbers', 'Community_Participation',
                'Fundraising', 'Volunteer_Hours', 'Building_Age', 'Condition_of_Facilities',
                'Technological_Resources', 'Safety']:
        df[f'Normalized_{col}'] = normalize(df[col])
    
    # Calculate composite scores for each category
    df['Performance_Score'] = (0.5 * df['Normalized_Test_Scores'] + 
                               0.3 * df['Normalized_Graduation_Rates'] + 
                               0.2 * df['Normalized_Attendance_Records'])

    df['Resource_Score'] = (0.6 * df['Normalized_Funding'] + 
                            0.4 * df['Normalized_Staffing'])

    df['Demographic_Score'] = (0.6 * df['Normalized_Low_Income_Students'] + 
                               0.4 * df['Normalized_Special_Needs_Students'])

    df['Community_Score'] = (0.5 * df['Normalized_Community_Participation'] + 
                             0.3 * df['Normalized_Fundraising'] + 
                             0.2 * df['Normalized_Volunteer_Hours'])

    df['Infrastructure_Score'] = (0.3 * df['Normalized_Building_Age'] + 
                                  0.3 * df['Normalized_Condition_of_Facilities'] + 
                                  0.2 * df['Normalized_Technological_Resources'] + 
                                  0.2 * df['Normalized_Safety'])

    # Calculate the final composite score
    df['Composite_Score'] = (weights['Performance Levels'] * df['Performance_Score'] + 
                             weights['Resource Levels'] * df['Resource_Score'] + 
                             weights['Student Demographics'] * df['Demographic_Score'] + 
                             weights['Community Engagement'] * df['Community_Score'] + 
                             weights['School Infrastructure Quality'] * df['Infrastructure_Score'])

    # Rank schools based on composite score
    df['Rank'] = df['Composite_Score'].rank(ascending=False).astype(int)
    
    return df

def main():
    # Check the current working directory and list files
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    
    # Read the dataset
    file_path = 'school_data.csv'
    df = pd.read_csv(file_path)
    
    # Calculate scores and rank schools
    df = calculate_scores(df)
    
    # Output the final DataFrame with scores and ranks in a clear format
    result = df[['School', 'Composite_Score', 'Rank']]
    result = result.sort_values(by='Rank')
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
