def count_solutions(board, limit=2):
    count = 0

    def valid(r, c, n):
        for i in range(9):
            if board[r][i] == n or board[i][c] == n:
                return False
        br, bc = (r//3)*3, (c//3)*3
        for i in range(3):
            for j in range(3):
                if board[br+i][bc+j] == n:
                    return False
        return True

    def solve():
        nonlocal count
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    for n in range(1,10):
                        if valid(r,c,n):
                            board[r][c] = n
                            solve()
                            board[r][c] = 0
                            if count >= limit:
                                return
                    return
        count += 1

    solve()
    return count

import random

def generate_puzzle(solution_str, clues=40):

    board = [[int(solution_str[r*9+c]) for c in range(9)] for r in range(9)]

    cells = list(range(81))
    random.shuffle(cells)

    removed = 0
    target_remove = 81 - clues

    for cell in cells:

        if removed >= target_remove:
            break

        r = cell // 9
        c = cell % 9

        backup = board[r][c]
        board[r][c] = 0

        copy_board = [row[:] for row in board]

        if count_solutions(copy_board) != 1:
            board[r][c] = backup
        else:
            removed += 1

    return ''.join(str(board[r][c]) for r in range(9) for c in range(9))

def generate_levels(solution):

    levels = {
        "easy":40,
        "medium":32,
        "hard":26,
        "expert":22
    }

    puzzles = {}

    for k,v in levels.items():
        puzzles[k] = generate_puzzle(solution, v)

    return puzzles