from datasets import load_dataset
import numpy as np
from tqdm import tqdm
# augment method
def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

# The data are sourced from https://huggingface.co/datasets/sapientinc/sudoku-extreme-1k
data_files = {
    'train': './train.csv',
    'test': './test.csv',
}

# augment the original dataset 160x
dataset = load_dataset('csv', data_files=data_files,split='test').repeat(1)

# len_log = []
from data.generator.util import generate_levels
xs, ys = [], []
puzzles = {}
leave_blank = 1
import random
nums = random.choices(range(81), k=len(dataset))
for idx,item in enumerate(tqdm(dataset)):
    board = np.frombuffer(item["question"].replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
    solution = np.frombuffer(item["answer"].encode(), dtype=np.uint8).reshape(9, 9) - ord('0')

    # transform puzzle
    # such as rotation and exchange columns or remapping the number
    # board, solution = shuffle_sudoku(board, solution)

    # len_log.append((board==0).sum())
    # Convert and flatten
    # board = (board + +ord('0')).flatten().astype(np.int8).tobytes().decode()

    solution = (solution.flatten().astype(np.int8) + ord('0')).tobytes().decode()
    # puzzles = generate_levels(solution)
    # puzzles['leave1'] = solution[nums[idx]]
    board = list(solution)
    board[nums[idx]] = "0"
    board = "".join(board)
    # Pad a BOS token
    xs.append(board)
    ys.append(solution)

import pandas as pd

# 创建 DataFrame
data = {'question': xs, 'answer': ys}
df = pd.DataFrame(data)
# len_log = np.array(len_log)
# print(len_log.mean())

df.to_csv(f'./test_leave1.csv', index=False)