import psycopg2
import json
import torch
import os
import sys
import random
import math
from bmtree.bmtree_env import BMTEnv
from int2binary import Int2BinaryTransformer
import argparse

import struct
import configs
import csv
from utils.query import Query
import numpy as np
import time
from utils.curves import ZorderModule, DesignQuiltsModule, HighQuiltsModule,  Z_Curve_Module

import configs

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))  # Adjust path to the root
sys.path.append(project_root)

from constants import TEMP_OUTPUT_FILE_PATH, TEMP_OUTPUT_FILE_NAME

args = configs.set_config()

def load_data(file_path):
    # with open(file_path, 'r') as f:
    #     return json.load(f)
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        data_reader = csv.reader(csvfile)
        data = [[float(item) for item in row] for row in data_reader if row]
    return data

# def float_to_int_bits(value, shift_length):
#     if value < 0:
#         value += 180  # dealing with negative numbers of locations
#     binary_value = struct.pack('>f', value)
#     int_value = int.from_bytes(binary_value, 'big')
#     sign = (int_value >> 31) & 0x1
#     high_bit = 1 if sign == 0 else 0
#     int_temp = int_value >> (33 - shift_length) # here use 33 instead of 32
#     int_32_bits = (high_bit << (shift_length - 1)) | int_temp
#     return int(int_32_bits)

def float_to_int_bits(value, shift_length, min_val, max_val):
    """
    Converts a floating-point value to integer bit representation, handling negative numbers, 
    and applying the specified shift length.

    :param value: The floating-point value to convert.
    :param shift_length: The bit shift length, used to control the final number of bits.
    :param min_val: The minimum value for scaling.
    :param max_val: The maximum value for scaling.
    :return: The integer bit representation after applying the shift length.
    """
    # Scale the float value to the range [0, (1 << shift_length) - 1]
    scale = (1 << shift_length) - 1
    range_val = max_val - min_val
    
    # Avoid division by zero in case max_val == min_val
    if range_val == 0:
        range_val = 1e-8
    
    # Ensure the value is within the specified range (handle out-of-bounds values)
    if value < min_val:
        value = min_val
    elif value > max_val:
        value = max_val
    
    # Map the float value to the integer range
    mapped_value = ((value - min_val) / range_val * scale)
    
    # Convert the mapped float value to an integer
    mapped_int = int(mapped_value)
    
    return np.uint64(mapped_int)


def compute_sfc_values(dataset, env, bit_length):
    """
    Compute SFC (Space Filling Curve) values for a given dataset.

    :param dataset: List of data points where each point is a list of float values.
    :param env: Environment with a tree structure that has an 'output' method for SFC calculation.
    :param bit_length: List of bit lengths for each dimension.
    :return: List of tuples where each tuple contains (original data point, SFC value).
    """
    # Initialize lists to store min and max values for each dimension
    n_dims = len(dataset[0])
    min_vals = [float('inf')] * n_dims
    max_vals = [float('-inf')] * n_dims

    # Find the min and max values for each dimension
    for data in dataset:
        for i in range(n_dims):
            min_vals[i] = min(min_vals[i], data[i])
            max_vals[i] = max(max_vals[i], data[i])

    # Now compute SFC values using the identified min and max values for scaling
    sfc_values = []
    for data in dataset:
        mapped_data = [float_to_int_bits(data[i], bit_length[i], min_vals[i], max_vals[i]) for i in range(n_dims)]
        sfc_value = env.tree.output(mapped_data)
        sfc_values.append((data, sfc_value)) 

    return sfc_values


def sort_by_sfc(sfc_values):
    sorted_sfc_values = sorted(sfc_values, key=lambda x: x[1])
    return sorted_sfc_values

def save_to_csv(sorted_sfc_values_with_data, output_file_path):
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for data, sfc_value in sorted_sfc_values_with_data:
            # writer.writerow(list(data) + [sfc_value])
            writer.writerow(list(data))

def main():

    bit_length = args.bit_length
    data_space = [2**(bit_length[i]) - 1 for i in range(len(bit_length))]
    
    binary_transfer = Int2BinaryTransformer(data_space)
    smallest_split_card = args.smallest_split_card
    max_depth = args.max_depth

    
    os.makedirs(TEMP_OUTPUT_FILE_PATH, exist_ok=True)
    bmtree_file = f"../../{TEMP_OUTPUT_FILE_PATH}/learned_bmtree.txt"
    dataset = load_data(args.data_file)
    
    env = BMTEnv(list(dataset), None, bit_length, bmtree_file, binary_transfer, smallest_split_card, max_depth)

    sfc_values_with_data = compute_sfc_values(dataset, env, bit_length)
    sorted_sfc_values_with_data = sort_by_sfc(sfc_values_with_data)

    output_file_path = f'../../{TEMP_OUTPUT_FILE_PATH}'
    os.makedirs(output_file_path, exist_ok=True)
    output_file_path = os.path.join(output_file_path, TEMP_OUTPUT_FILE_NAME.format(order_type='bmtree'))
    save_to_csv(sorted_sfc_values_with_data, output_file_path)

    # print("Sorted data and SFC values saved to:", output_file_path)

if __name__ == '__main__':
    main()
