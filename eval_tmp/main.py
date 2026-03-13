import yaml, json
import numpy as np
import torch

from tqdm import trange

from hrm import HierarchicalRecurrentModelV1

def trunc_normal_init_(tensor, std=1.0, lower=-2.0, upper=2.0):
    """Truncated normal initialization"""
    lower = torch.tensor(lower, device=tensor.device)
    upper = torch.tensor(upper, device=tensor.device)
    with torch.no_grad():
        if std == 0: return tensor.zero_()
        sqrt2 = torch.sqrt(torch.tensor(2.0,device=tensor.device))
        a, b = torch.erf(lower/sqrt2), torch.erf(upper/sqrt2)
        z, c = (b-a)/2, (2*torch.pi)**-0.5
        pdf_u, pdf_l = c*torch.exp(-0.5*lower**2), c*torch.exp(-0.5*upper**2)
        comp_std = std / torch.sqrt(1 - (upper*pdf_u - lower*pdf_l)/z - ((pdf_u-pdf_l)/z)**2)
        tensor.uniform_(a, b).erfinv_().mul_(sqrt2*comp_std).clip_(lower*comp_std, upper*comp_std)
    return tensor


def run_validation(model, puzzles, solutions, batch_size):

    puzzle_num = len(puzzles)
    global_right_cell_num = 0.0
    global_right_num = 0

    fail_ = np.array([])
    for idx in trange(0,puzzle_num, batch_size):
        batch_data = puzzles[idx:idx + batch_size]
        inputs = torch.tensor(batch_data).to('cuda')
        puzzle_ids = torch.zeros((inputs.shape[0],1), dtype=torch.long).to('cuda')
        answers = []
        init_hidden=True
        for cycle in range(16):
            logits = model(inputs=inputs, puzzle_identifiers=puzzle_ids,init_hidden=init_hidden)
            init_hidden=True
            answer = (torch.argmax(logits, dim=-1) - 1)
            answers.append(answer)
        batch_solutions = solutions[idx:idx + batch_size]
        batch_solutions_tensor = torch.tensor(batch_solutions).to('cuda')
        right_cell_num = (batch_solutions_tensor==answers[-1]).float().sum().item()
        right_num = ((batch_solutions_tensor==answers[-1]).sum(-1)==81).sum().item()

        current_fail_np  = ((batch_solutions_tensor == answers[-1]).sum(-1) != 81).nonzero().squeeze().cpu().numpy() + idx
        fail_ = np.append(fail_, current_fail_np)
        global_right_num += right_num

        global_right_cell_num += right_cell_num
        # print('right cell num',right_cell_num)
        # print(idx)
        # exit()
    print('global right cell num',global_right_num/puzzle_num)
    return fail_.tolist()

def main():
    # import pandas as pd
    # df = pd.read_csv('../test_convert.csv')
    # df.to_json('../test_convert.json', orient='records', force_ascii=False, indent=4)
    # exit()
    with open('../test_convert.json', 'r') as f:
        data = json.load(f)

    # Convert to original format: (puzzle, solution, name)
    puzzles = []
    solutions = []
    rbb = []
    for item in data:  # Take first 10 cases for testing
        puzzles.append([1 if c=='.' else int(c)+1 for c in item['question']])
        rbb.append(item['question'])
        solutions.append([int(c) for c in item['answer']])

    model_path, config_path = "../step_166300", "opti.yaml"

    device = torch.device("cuda")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = 1024
    cfg = dict(**config['arch'], batch_size=batch_size, vocab_size=11, seq_len=81,
               num_puzzle_identifiers=1, causal=False)
    torch.manual_seed(42)
    with torch.device(device):
        model = HierarchicalRecurrentModelV1(cfg)
        state_dict = torch.load(model_path, map_location=device)
        if any(k.startswith('_orig_mod.model.') for k in state_dict.keys()):
            state_dict = {k[len('_orig_mod.model.'):] if k.startswith('_orig_mod.model.') else k: v
                          for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    # model._init_hidden()

    fail_ = run_validation(model, puzzles, solutions, batch_size)

    with open('fail_index', 'w') as f:
        f.write(str(fail_))



if __name__ == "__main__":
    main()