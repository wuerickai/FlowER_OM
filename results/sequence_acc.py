groupings = []
import numpy as np
import ast

with open("result.txt") as t:
    curr = None
    tot = 0
    for line in t:
        id = line.strip().split("|")[-1]
        if curr == None:
            curr = id
            tot = 1
            continue
        if id != curr:
            curr = id
            groupings.append(tot)
            tot = 1
        else:
            tot += 1
    if tot > 0:
        groupings.append(tot)

print("Groupings Determined: ", groupings[:10]) 

def get_top_k(filepath, samples=64, start_line=1, end_line=None):
    topk = []

    with open(filepath, 'r') as f:
        for idx, line in enumerate(f, start=1):  # 1-indexed line numbers
            if idx < start_line:
                continue
            if end_line is not None and idx > end_line:
                break

            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split('|')
                first_list_str = parts[0]
                valid_smiles_str = parts[2]

                first_list = ast.literal_eval(first_list_str)
                valid_smiles = ast.literal_eval(valid_smiles_str)
                output_counts = [int(i[1]) for i in valid_smiles]
                sorted_counts = sorted(output_counts, reverse=True)

                correct_count = first_list[0]
                if correct_count:
                    topk.append(sorted_counts.index(correct_count) + 1)
                else:
                    topk.append(2000)

            except Exception as e:
                print(f"Skipping line {idx} due to parse error: {e}")

    return topk

   
def sequence_accuracy_at_k(groupings, topk, ks=[1,2,3,4,5]):
    """
    groupings : list[int]
        Length of each sequence
    topk : list[int]
        Flattened per-step ranks (1 = top-1 correct, large = incorrect)
    ks : iterable[int]
        Values of k to evaluate

    Returns
    -------
    results : dict
        k -> {
            "per_sequence": list[bool],
            "accuracy": float
        }
    """
    print(len(topk))
    print(sum(groupings))
    assert sum(groupings) == len(topk), "groupings must sum to len(topk)"

    results = {}
    start = 0

    # Pre-split sequences once
    sequences = []
    for g in groupings:
        sequences.append(topk[start:start + g])
        start += g

    for k in ks:
        per_sequence = [
            all(step <= k for step in seq)
            for seq in sequences
        ]
        accuracy = sum(per_sequence) / len(per_sequence)

        results[k] = {
            "accuracy": accuracy
        }

    return results

topk = get_top_k('result.txt', samples=16, start_line=1, end_line=None)
results  = sequence_accuracy_at_k(groupings,topk)
