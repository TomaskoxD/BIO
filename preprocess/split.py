import argparse
import pandas as pd
import numpy as np

def split_data(input_path, train_output_path, test_output_path, threshold):
    df = pd.read_csv(input_path)

    msk = np.random.rand(len(df)) < threshold
    train = df[msk]
    test = df[~msk]

    train.to_csv(train_output_path, index=False)
    test.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV data into training and testing sets.")

    parser.add_argument("input_path", help="Path to the input CSV file containing the data.")
    parser.add_argument("train_output_path", help="Path to the output CSV file for the training data.")
    parser.add_argument("test_output_path", help="Path to the output CSV file for the testing data.")
    parser.add_argument("threshold", type=float, help="Threshold for the data split (a number between 0 and 1).")

    args = parser.parse_args()

    split_data(args.input_path, args.train_output_path, args.test_output_path, args.threshold)
    print("Data split successfully...")
