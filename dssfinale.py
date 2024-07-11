import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# Define weights for each metric (with justification in comments)
weights = {
    'Performance Levels': 0.3,  # Prioritize academic achievement
    'Resource Levels': 0.2,      # Address resource disparities
    'Student Demographics': 0.2,  # Focus on equity and inclusion
    'Community Engagement': 0.2,  # Value community involvement
    'School Infrastructure Quality': 0.1  # Ensure safe learning environments
}

def normalize(series):
    scaler = MinMaxScaler()
    return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

def calculate_scores(df):
    # Normalizing all relevant fields
    for col in ['Test_Scores', 'Graduation_Rates', 'Attendance_Records']:
        df[f'Normalized_{col}'] = df[col].apply(lambda x: normalize(pd.Series(eval(x))).mean())

    for col in ['Funding', 'Staffing', 'Materials', 'Low_Income_Students', 'Special_Needs_Students', 
                'Enrollment_Numbers', 'Community_Participation', 'Fundraising', 'Volunteer_Hours', 
                'Building_Age', 'Condition_of_Facilities', 'Technological_Resources', 'Safety']:
        df[f'Normalized_{col}'] = normalize(df[col])

    # Calculate composite scores for each category (with internal weights)
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
    df['Rank'] = df['Composite_Score'].rank(ascending=False)

    return df

def load_data_from_csv():
    file_path = 'school_data.csv'
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def display_results(df):
    # Create main window
    window = tk.Tk()
    window.title("School Ranking Results")

    # Create treeview for results
    tree = ttk.Treeview(window)
    tree["columns"] = ("School", "Composite Score", "Rank")
    tree.heading("School", text="School")
    tree.heading("Composite Score", text="Composite Score")
    tree.heading("Rank", text="Rank")

    # Insert data into treeview
    for index, row in df.iterrows():
        tree.insert("", "end", text=index, values=(row['School'], row['Composite_Score'], row['Rank']))

    tree.pack()

    # Create button to show bar chart
    show_chart_button = tk.Button(window, text="Show Bar Chart", command=lambda: show_bar_chart(df))
    show_chart_button.pack()

    window.mainloop()

def show_bar_chart(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df['School'], df['Composite_Score'])
    plt.xlabel("School")
    plt.ylabel("Composite Score")
    plt.title("School Ranking by Composite Score")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    df = load_data_from_csv()
    if df is not None:
        df = calculate_scores(df)
        display_results(df)
