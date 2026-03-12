import numpy as np
import torch
from typing import Tuple, Dict, Any


# [Dataloader and training loop]
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

def collate_fn(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for item in batch:
        board = np.frombuffer(item["question"].replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        solution = np.frombuffer(item["answer"].encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        # Convert and flatten
        board = board.flatten().astype(np.int32)
        solution = solution.flatten().astype(np.int32)
        # Pad a BOS token
        xs.append(board)
        ys.append(solution)

    return torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0))