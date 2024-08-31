import psycopg2
import json
import torch
import os
import sys
import random
import math
from bmtree.bmtree_env import BMTEnv
from int2binary import Int2BinaryTransformer

import struct
import configs
args = configs.set_config()
import csv
from utils.query import Query
import numpy as np
import time
from utils.curves import ZorderModule, DesignQuiltsModule, HighQuiltsModule,  Z_Curve_Module




'''Set the state_dim and action_dim, for now'''
bit_length = args.bit_length
data_dim = len(bit_length)
bit_length = [int(i) for i in bit_length]
data_space = [2**(bit_length[i]) - 1 for i in range(len(bit_length))]
bmtree_file = 'learned_bmtree.txt'
binary_transfer = Int2BinaryTransformer(data_space)
smallest_split_card = args.smallest_split_card
max_depth = args.max_depth


def load_data(file_path):
    # with open(file_path, 'r') as f:
    #     return json.load(f)
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        data_reader = csv.reader(csvfile)
        data = [[float(item) for item in row] for row in data_reader if row]
    return data


# def float_to_int_bits(value, shift_length):
#     binary_value = struct.pack('>f', value) 
#     int_value = int.from_bytes(binary_value, 'big')
#     int_20_bits = int_value >> (32 - shift_length) 
#     return int(int_20_bits)


def float_to_int_bits(value, shift_length):
    if value < 0:
        value += 180  # dealing with negative numbers of locations
    binary_value = struct.pack('>f', value)
    int_value = int.from_bytes(binary_value, 'big')
    sign = (int_value >> 31) & 0x1
    high_bit = 1 if sign == 0 else 0
    int_temp = int_value >> (33 - shift_length) # here use 33 instead of 32
    int_32_bits = (high_bit << (shift_length - 1)) | int_temp
    return int(int_32_bits)


def compute_sfc_values(dataset, env):
    sfc_values = []
    for data in dataset:
        mapped_data = [float_to_int_bits(data[i], bit_length[i]) for i in range(len(data))]
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
            writer.writerow(list(data) + [sfc_value])

def main():
    # data_path = 'data/{}.json'.format(args.data)
    data_path = '../../{}'.format(args.data)
    dataset = load_data(data_path)
    
    env = BMTEnv(list(dataset), None, bit_length, bmtree_file, binary_transfer, smallest_split_card, max_depth)

    sfc_values_with_data = compute_sfc_values(dataset, env)
    sorted_sfc_values_with_data = sort_by_sfc(sfc_values_with_data)

    output_file_path = 'sorted_data_with_sfc.csv'
    save_to_csv(sorted_sfc_values_with_data, output_file_path)

    print("Sorted data and SFC values saved to:", output_file_path)


    # todo deal with queries.

if __name__ == '__main__':
    main()
