import math
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import a_en, transition_metals, outer_electron_lookup

max_key = max(a_en)
en_lookup_tensor = torch.zeros(max_key + 1)

for k, v in a_en.items():
    try:
        en_lookup_tensor[k] = float(v)
    except ValueError:
        en_lookup_tensor[k] = 0.0

transition_metals_tensor = torch.from_numpy(transition_metals.astype(np.int16))

outer_electron_tensor = torch.zeros(120, dtype=torch.int16)

for i, val in outer_electron_lookup.items():
    outer_electron_tensor[i] = val
outer_electron_tensor[0] = 0  # Padding for atomic number 0

class OxdStatesCalculator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.register_buffer('outer_electron_tensor', outer_electron_tensor.to(self.device))
        self.register_buffer('en_lookup_tensor', en_lookup_tensor.to(self.device))
        self.register_buffer('transition_metals_tensor', transition_metals_tensor.to(self.device))
        #print(self.transition_metals_tensor)


    def get_mat_formal_charge(self, matrix, a_nums) -> torch.Tensor:
        """
        Calculates the formal charge of each non-masked atom in the matrix, 
        matrix: (batch, n, n) tensor of bonding electrons
        a_nums: (batch, n) or tensor of atomic numbers (0 indicates padding)
        
        Returns:
        formal_charge: tensor with shape matching a_num
        """
        valence = self.outer_electron_tensor[a_nums]  # (batch, n)
        #print("Valence: ", valence)
        return valence - matrix.sum(dim=1)  

    def calculate_oxd_states(self, matrix, a_nums, cutoff=0.05) -> torch.Tensor:
        """
        Calculate oxidation states using PyTorch operations.
        matrix: (batch, n, n) tensor of bonding electrons
        a_nums: (batch, n) tensor of atomic numbers (0 indicates padding)

        Returns:
        oxd_states: tensor with shape matching a_nums
        
        """
        # Single gather for atomic-number-indexed lookups.
        valence = self.outer_electron_tensor[a_nums]                # (b, n)
        e_negs = self.en_lookup_tensor[a_nums]                      # (b, n)

        # Symmetric bond-order matrix (matrix[i,j] + matrix[j,i]).
        bond = matrix + matrix.transpose(-1, -2)                    # (b, n, n)

        # Antisymmetric electronegativity-difference matrix.
        diff = e_negs.unsqueeze(2) - e_negs.unsqueeze(1)            # (b, n, n)

        # Per-atom oxidation contribution:
        #     oxd[i] = -sum_j sign(diff[i,j]) * bond[i,j] / 2
        # where sign is +1 when diff > cutoff, -1 when diff < -cutoff, 0
        # otherwise. The (vals != 0) mask used in the original is dropped
        # because the result is already zero wherever bond == 0.
        # Equivalent factored form: bond * (neg - pos) * 0.5.
        sign_neg_minus_pos = (diff < -cutoff).to(bond.dtype) - (diff > cutoff).to(bond.dtype)
        oxd_states = (bond * sign_neg_minus_pos).sum(dim=2) * 0.5   # (b, n)

        # Formal charge = valence - col-sum (preserves original semantics
        # for both symmetric and asymmetric input matrices).
        formal_charges = valence - matrix.sum(dim=1)

        return oxd_states + formal_charges

    def forward(self, matrix, a_nums):
        """
        Calculate the oxidation states of atoms in the matrix.
        matrix: (batch, n, n) tensor of bonding electrons
        a_nums: (batch, n) tensor of atomic numbers (0 indicates padding)
        
        Returns:
        oxd_states: tensor with shape matching a_nums
        """
        with torch.no_grad():
            return self.calculate_oxd_states(matrix, a_nums)