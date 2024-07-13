import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define weights for each metric
weights = {
    'Performance Levels': 0.3,
    'Resource Deficits': 0.25,
    'Student Demographics': 0.15,
    'Community Engagement': 0.1,
    'School Infrastructure Quality': 0.2
}

# Define sub-weights for each category
sub_weights = {
    'Performance Levels': {
        'Test Scores': 0.4,
        'Graduation Rates': 0.4,
        'Attendance Records': 0.2
    },
    'Resource Deficits': {
        'Funding': 0.6,
        'Staffing': 0.3,
        'Materials': 0.1
    },
    'Student Demographics': {
        'Low-Income Students': 0.3,
        'Special Needs Students': 0.3,
        'Enrollment Numbers': 0.4
    },
    'Community Engagement': {
        'Community Participation': 0.3,
        'Fundraising': 0.5,
        'Volunteer Hours': 0.2
    },
    'School Infrastructure Quality': {
        'Building Age': 0.2,
        'Condition of Facilities': 0.5,
        'Technological Resources': 0.3
    }
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
        df[col] = df[col].astype(str).str.replace(r'[\[\]"“]', '', regex=True)
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
    df['Performance_Score'] = (sub_weights['Performance Levels']['Test Scores'] * df['Normalized_Test_Scores'] + 
                               sub_weights['Performance Levels']['Graduation Rates'] * df['Normalized_Graduation_Rates'] + 
                               sub_weights['Performance Levels']['Attendance Records'] * df['Normalized_Attendance_Records'])

    df['Resource_Deficit_Score'] = (sub_weights['Resource Deficits']['Funding'] * df['Normalized_Funding'] + 
                                    sub_weights['Resource Deficits']['Staffing'] * df['Normalized_Staffing'] + 
                                    sub_weights['Resource Deficits']['Materials'] * df['Normalized_Materials'])

    df['Demographic_Score'] = (sub_weights['Student Demographics']['Low-Income Students'] * df['Normalized_Low_Income_Students'] + 
                               sub_weights['Student Demographics']['Special Needs Students'] * df['Normalized_Special_Needs_Students'] + 
                               sub_weights['Student Demographics']['Enrollment Numbers'] * df['Normalized_Enrollment_Numbers'])

    df['Community_Score'] = (sub_weights['Community Engagement']['Community Participation'] * df['Normalized_Community_Participation'] + 
                             sub_weights['Community Engagement']['Fundraising'] * df['Normalized_Fundraising'] + 
                             sub_weights['Community Engagement']['Volunteer Hours'] * df['Normalized_Volunteer_Hours'])

    df['Infrastructure_Score'] = (sub_weights['School Infrastructure Quality']['Building Age'] * df['Normalized_Building_Age'] + 
                                  sub_weights['School Infrastructure Quality']['Condition of Facilities'] * df['Normalized_Condition_of_Facilities'] + 
                                  sub_weights['School Infrastructure Quality']['Technological Resources'] * df['Normalized_Technological_Resources'])

    # Calculate the final composite score
    df['Composite_Score'] = (weights['Performance Levels'] * (1 - df['Performance_Score']) + # Subtracting as high performance reduces need for resources
                             weights['Resource Deficits'] * df['Resource_Deficit_Score'] + 
                             weights['Student Demographics'] * df['Demographic_Score'] - # Subtracting as high community engagement reduces need for resources
                             weights['Community Engagement'] * (1 - df['Community_Score']) + 
                             weights['School Infrastructure Quality'] * df['Infrastructure_Score'])

    # Rank schools based on composite score
    df['Rank'] = df['Composite_Score'].rank(ascending=False).astype(int)
    
    return df

def categorize_schools(df):
    """Categorize schools based on composite score thresholds."""
    conditions = [
        (df['Composite_Score'] < 0.4),
        (df['Composite_Score'] >= 0.4) & (df['Composite_Score'] < 0.7),
        (df['Composite_Score'] >= 0.7) & (df['Composite_Score'] < 0.85),
        (df['Composite_Score'] >= 0.85)
    ]
    categories = ['High Need', 'Moderate Need', 'Low Need', 'No Need']
    
    df['Need_Category'] = np.select(conditions, categories)
    
    return df

def allocate_resources(df, total_resources):
    """Allocate resources based on need categories and composite scores."""
    # Categorize schools based on composite score thresholds
    df = categorize_schools(df)
    
    # Calculate the base allocation for High Need schools
    base_allocation_high_need = 0.5 * total_resources
    
    # Count the number of schools in each category
    category_counts = df['Need_Category'].value_counts(normalize=True)
    
    # Calculate remaining resources
    remaining_resources = total_resources - base_allocation_high_need
    
    # Allocate remaining resources proportionally
    category_allocation = {
        'High Need': base_allocation_high_need,
        'Moderate Need': category_counts.get('Moderate Need', 0) * remaining_resources,
        'Low Need': category_counts.get('Low Need', 0) * remaining_resources,
        'No Need': 0
    }
    
    # Allocate resources within each category
    allocations = []
    for category, allocation in category_allocation.items():
        category_df = df[df['Need_Category'] == category]
        total_score = category_df['Composite_Score'].sum()
        category_df['Allocated_Resources'] = category_df['Composite_Score'].apply(lambda x: (x / total_score) * allocation)
        allocations.append(category_df)
    
    # Combine all allocations into one DataFrame
    df = pd.concat(allocations)
    
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
    
    # Define total resources available for allocation
    total_resources = 100000000  # $100,000,000
    
    # Allocate resources based on need categories and composite scores
    df = allocate_resources(df, total_resources)
    
    # Format the allocated resources to the correct format
    df['Allocated_Resources'] = df['Allocated_Resources'].apply(lambda x: f"${x / 1_000_000:.1f}M")
    
    # Output the final DataFrame with scores, ranks, allocation groups, and allocated resources in a clear format
    result = df[['School', 'Composite_Score', 'Need_Category', 'Rank', 'Allocated_Resources']]
    result = result.sort_values(by='Rank')
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
