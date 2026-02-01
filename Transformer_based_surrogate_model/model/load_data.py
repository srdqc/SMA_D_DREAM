import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_csv(file_path, data_line):
    """
    Load a single CSV file and remove the first and last columns.

    Parameters:
    file_path (str): Path to the CSV file
    data_line (int): Number of rows to retain (if greater than available rows, all rows are used)
    """
    df = pd.read_csv(file_path)

    # Filter out rows where the first column contains abnormal values
    df = df[(df.iloc[:, 0] != -10000000000.0) & (df.iloc[:, 0] != -1.4308224e17)]

    # Remove the first column and the last column
    df = df.iloc[:, 1:-1]

    # If data_line exceeds the number of available rows, adjust automatically
    sample_n = min(data_line, len(df))

    # Random sampling (will not raise an error if data_line > data size)
    df = df.sample(n=sample_n, random_state=42)

    return df


def load_data_from_directory(directory_path, data_line):
    """
    Load all CSV files from a given directory, remove the first and last columns
    of each file, and merge them into a single DataFrame.

    Parameters:
    directory_path (str): Directory path
    data_line (int): Number of rows to load from each CSV file

    Returns:
    pd.DataFrame: Merged DataFrame
    """
    file_paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.endswith(".csv")
    ]

    # Convert data_line into an iterable for parallel processing
    data_lines = [data_line] * len(file_paths)

    with ThreadPoolExecutor() as executor:
        data_frames = list(executor.map(load_csv, file_paths, data_lines))

    return pd.concat(data_frames, ignore_index=True)


def extract_param_data(data):
    """
    Extract the first 99 columns as the parameter dataset.

    Parameters:
    data (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Parameter dataset
    """
    param_data = data.iloc[:, :99]
    return param_data


def extract_result_data(data):
    """
    Extract all columns from the 100th column onward as the result dataset.

    Parameters:
    data (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Result dataset
    """
    result_data = data.iloc[:, 99:]
    return result_data


def split_result_data_to_dict(result_data, chunk_size=71):
    """
    Split the result dataset into multiple sub-datasets, each containing
    a fixed number of columns, and store them in a dictionary.

    Parameters:
    result_data (pd.DataFrame): Result dataset
    chunk_size (int): Number of columns per sub-dataset

    Returns:
    dict: Dictionary containing chunked result datasets
    """
    result_dict = {}
    num_chunks = result_data.shape[1] // chunk_size

    for i in range(num_chunks):
        chunk = result_data.iloc[:, i * chunk_size : (i + 1) * chunk_size]
        result_dict[f"result_chunk_{i + 1}"] = chunk

    return result_dict


def split_data_from_dict(param_data, result_dict, test_size=0.3):
    """
    Split the parameter dataset and each result sub-dataset into
    training and validation sets.

    Parameters:
    param_data (pd.DataFrame): Parameter dataset
    result_dict (dict): Dictionary of result datasets
    test_size (float): Proportion of the validation set

    Returns:
    tuple: Training and validation parameter datasets, and
           dictionaries of training and validation result datasets
    """
    # Split parameter dataset
    param_train = param_data.sample(frac=1 - test_size, random_state=42)
    param_val = param_data.drop(param_train.index)

    # Split each result dataset in the dictionary
    result_train_dict = {}
    result_val_dict = {}

    for key, value in result_dict.items():
        train = value.sample(frac=1 - test_size, random_state=42)
        val = value.drop(train.index)
        result_train_dict[key] = train
        result_val_dict[key] = val

    return param_train, param_val, result_train_dict, result_val_dict


def get_dict_data(directory_path, data_line, test_size=0.3):
    """
    Main function to load data and extract parameter and result datasets
    in dictionary form.

    Parameters:
    directory_path (str): Path to the data directory
    data_line (int): Number of rows to sample from each CSV file
    test_size (float): Proportion of the validation set

    Returns:
    tuple: Training and validation parameter datasets, and
           dictionaries of training and validation result datasets
    """
    # Load data
    data = load_data_from_directory(directory_path, data_line)

    # Extract parameter dataset
    param_data = extract_param_data(data)

    # Extract result dataset
    result_data = extract_result_data(data)

    # Split result dataset into chunks stored in a dictionary
    result_dict = split_result_data_to_dict(result_data)

    # Split datasets into training and validation sets
    param_train, param_val, result_train_dict, result_val_dict = split_data_from_dict(
        param_data, result_dict, test_size
    )

    return param_train, param_val, result_train_dict, result_val_dict


def split_data(param_data, result_data, test_size=0.3):
    """
    Split parameter and result datasets into training and validation sets.

    Parameters:
    param_data (pd.DataFrame): Parameter dataset
    result_data (pd.DataFrame): Result dataset
    test_size (float): Proportion of the validation set

    Returns:
    tuple: Training and validation parameter datasets and result datasets
    """
    # Split parameter dataset
    param_train = param_data.sample(frac=1 - test_size, random_state=42)
    param_val = param_data.drop(param_train.index)

    # Split result dataset
    result_train = result_data.sample(frac=1 - test_size, random_state=42)
    result_val = result_data.drop(result_train.index)

    return param_train, param_val, result_train, result_val


def get_data(directory_path, data_line, test_size=0.3):
    """
    Main function to load data and extract parameter and result datasets.

    Parameters:
    directory_path (str): Path to the data directory
    data_line (int): Number of rows to sample from each CSV file
    test_size (float): Proportion of the validation set

    Returns:
    tuple: Training and validation parameter datasets and result datasets
    """
    # Load data
    data = load_data_from_directory(directory_path, data_line)

    # Extract parameter dataset
    param_data = extract_param_data(data)

    # Extract result dataset
    result_data = extract_result_data(data)

    # Split datasets into training and validation sets
    param_train, param_val, result_train, result_val = split_data(
        param_data, result_data, test_size
    )

    return param_train, param_val, result_train, result_val
