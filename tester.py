# Define a simplified test input
test_input = [
    [1, 2],
    [3, 4]
]

# Expected output based on a specific pattern (adjust as per your actual pattern)
expected_output = [
    [1, 2, 1, 2, 1, 2],
    [3, 4, 3, 4, 3, 4],
    [1, 2, 1, 2, 1, 2],
    [3, 4, 3, 4, 3, 4],
    [1, 2, 1, 2, 1, 2],
    [3, 4, 3, 4, 3, 4],
]

# Define necessary tools (ensure no conflicting names)
def get_size(input_grid):
    return len(input_grid), len(input_grid[0])

def empty_grid(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def fill_grid(grid, row, col, value, height, width):
    for i in range(row, row + height):
        for j in range(col, col + width):
            grid[i][j] = value
    return grid

def horizontal_flip(grid, row, col, value, size):
    subgrid = [grid[row + i][col:col + size] for i in range(size)]
    flipped_subgrid = [list(reversed(r)) for r in subgrid]
    for i in range(size):
        grid[row + i][col:col + size] = flipped_subgrid[i]
    return grid

def vertical_flip(grid, row, col, value, size):
    subgrid = [grid[row + i][col:col + size] for i in range(size)]
    flipped_subgrid = list(reversed(subgrid))
    for i in range(size):
        grid[row + i][col:col + size] = flipped_subgrid[i]
    return grid

def rotate_clockwise(grid, row, col, size):
    subgrid = [grid[row + i][col:col + size] for i in range(size)]
    rotated_subgrid = [list(r) for r in zip(*subgrid[::-1])]
    for i in range(size):
        grid[row + i][col:col + size] = rotated_subgrid[i]
    return grid

# Update all_tools without conflicting helper functions
all_tools = {
    "get_size": get_size,
    "empty_grid": empty_grid,
    "fill_grid": fill_grid,
    "horizontal_flip": horizontal_flip,
    "vertical_flip": vertical_flip,
    "rotate_clockwise": rotate_clockwise,
    # Add other tools if necessary
}

# Prepare the execution environment
import numpy as np
import pandas as pd
import json
import copy
import math

# Inject necessary modules into global_vars
global_vars = {name: func for name, func in all_tools.items()}
global_vars.update({
    "np": np,
    "pd": pd,
    "json": json,
    "copy": copy,
    "math": math,
})

# Define a simplified generated code (simulate what the LLM would generate)
generated_code = """
def solve_task(input_grid):
    rows, cols = get_size(input_grid)
    output_grid = empty_grid(rows * 3, cols * 3)

    for row in range(rows):
        for col in range(cols):
            value = input_grid[row][col]
            fill_grid(output_grid, row * 3, col * 3, value, 3, 3)
            horizontal_flip(output_grid, row * 3 + 1, col * 3, value, 3)
            vertical_flip(output_grid, row * 3 + 2, col * 3, value, 3)
            rotate_clockwise(output_grid, row * 3, col * 3 + 1, 3)
            rotate_clockwise(horizontal_flip(output_grid, row * 3 + 1, col * 3 + 1, 3), 3)
            rotate_clockwise(vertical_flip(output_grid, row * 3 + 2, col * 3 + 1, 3), 3)

    return output_grid
"""

# Execute the generated code
exec(generated_code, global_vars)

# Retrieve the solve_task function
solve_task = global_vars.get('solve_task')

# Execute solve_task with test_input
if solve_task:
    prediction = solve_task(test_input)
    print("Prediction:", prediction)
    print("Expected:", expected_output)
    print("Match:", prediction == expected_output)
else:
    print("solve_task not defined.")
