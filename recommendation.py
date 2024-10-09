import pandas as pd
import numpy as np
import csv

def remove_empty_cells(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            cleaned_row = [cell for cell in row if cell.strip()]  # Remove empty cells
            writer.writerow(cleaned_row)

input_file = 'data.csv'   # Replace with your input CSV file path
output_file = 'output.csv'  # Replace with your desired output CSV file path

remove_empty_cells(input_file, output_file)

# Step 1: Read the CSV file
file_path = 'output.csv'  # Update the path as necessary
df = pd.read_csv(file_path)

# Step 2: Extract the data
emails = df.iloc[:, 0]  # First column for email addresses
impressions = df.iloc[:, 1:]  # Remaining columns for impressions

# Step 3: Create an empty adjacency matrix
num_students = len(emails)
adj_matrix = np.zeros((num_students, num_students), dtype=int)

# Step 4: Fill the adjacency matrix
for i in range(num_students):
    for j in range(impressions.shape[1]):
        if impressions.iloc[i, j] in emails.values:
            target_index = emails[emails == impressions.iloc[i, j]].index[0]
            adj_matrix[i, target_index] = 1

# Step 5: Adjust values to -1 where necessary
for i in range(num_students):
    for j in range(num_students):
        if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
            adj_matrix[j, i] = -1

# Convert the adjacency matrix to a DataFrame for better readability
p_df = pd.DataFrame(adj_matrix, index=emails, columns=emails)

print(p_df)

# Initialize the latent feature matrices
num_features = 3
u_df = pd.DataFrame(np.random.rand(num_students, num_features), index=p_df.index)
v_df = pd.DataFrame(np.random.rand(num_features, num_students), columns=p_df.index)

# Gradient Descent Parameters
alpha = 0.001  # Learning rate
beta = 0.01  # Regularization term
num_iterations = 5

error = []
# Gradient Descent for Matrix Factorization
for iteration in range(num_iterations):
    P = u_df.values.dot(v_df.values)
    s = 0
    for i in range(num_students):
        for j in range(num_students):
            if p_df.iloc[i, j] != 0:  # Only update for non-zero entries in p_df
                e = p_df.iloc[i, j] - P[i, j]
                for k in range(num_features):
                    u_df.iloc[i, k] += alpha * (2 * e * v_df.iloc[k, j] - beta * u_df.iloc[i, k])
                    v_df.iloc[k, j] += alpha * (2 * e * u_df.iloc[i, k] - beta * v_df.iloc[k, j])
                s += abs(e)
    print("error : ", s)
    error.append(s)

# Calculate the final P matrix
P = u_df.values.dot(v_df.values)

# Convert P to DataFrame for better readability
P_df = pd.DataFrame(P, index=p_df.index, columns=p_df.columns)
# print(P_df)  # Print a portion of the matrix for inspection

# Convert the predicted matrix P into an adjacency matrix
threshold = 0.6
predicted_adj_matrix = (P_df > threshold).astype(int)

# Replace all cells with the data from the original adjacency matrix where values are 1 or -1
for i in range(num_students):
    for j in range(num_students):
        if adj_matrix[i, j] in [1, -1]:
            predicted_adj_matrix.iloc[i, j] = adj_matrix[i, j]

# Print the final adjusted adjacency matrix
print(predicted_adj_matrix)

