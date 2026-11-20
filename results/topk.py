import numpy as np
import ast

def get_topk_from_file(filepath, samples=16, start_line=1, end_line=None):
    topk = []
    wrong_smi_conserved = []
    wrong_smi_nonconserved = []
    error_smi_conserved = []
    error_smi_nonconserved = []

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

                wrong_smi_conserved.append(first_list[1])
                wrong_smi_nonconserved.append(first_list[2])
                error_smi_conserved.append(first_list[3])
                error_smi_nonconserved.append(first_list[4])

            except Exception as e:
                print(f"Skipping line {idx} due to parse error: {e}")

    # Convert to numpy arrays
    topk = np.array(topk)
    wrong_smi_conserved = np.array(wrong_smi_conserved)
    wrong_smi_nonconserved = np.array(wrong_smi_nonconserved)
    error_smi_conserved = np.array(error_smi_conserved)
    error_smi_nonconserved = np.array(error_smi_nonconserved)

    # Print error metrics
    print("Wrong SMILES (electrons conserved): ",
          round(np.mean(wrong_smi_conserved > 0) / samples, 4))
    print("Wrong SMILES (electrons non-conserved): ",
          round(np.mean(wrong_smi_nonconserved > 0) / samples, 4))
    print("Error SMILES (electrons conserved): ",
          round(np.mean(error_smi_conserved > 0) / samples, 4))
    print("Error SMILES (electrons non-conserved): ",
          round(np.mean(error_smi_nonconserved > 0) / samples, 4))

    # Top-k accuracy
    for i in range(1, 6):
        print(f"Top {i} step accuracy (conservative): {round(np.mean(topk <= i), 4)}")

    return [round(np.mean(topk <= i), 4) for i in range(1, 6)]