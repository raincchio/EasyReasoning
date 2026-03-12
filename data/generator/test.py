from  util import generate_levels

solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"

puzzles = generate_levels(solution)

for k,v in puzzles.items():
    print(k, v)