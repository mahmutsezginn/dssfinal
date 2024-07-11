import os

def print_csv_lines(file_path, num_lines=5):
    try:
        with open(file_path, 'r') as file:
            for _ in range(num_lines):
                print(file.readline())
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    # Check the current working directory and list files
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    
    # Path to the CSV file
    file_path = 'school_datamake.csv'
    
    # Print the first few lines of the CSV file
    print_csv_lines(file_path)

if __name__ == "__main__":
    main()