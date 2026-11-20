import os
import glob
import datetime
import torch
import numpy as np
from rdkit import Chem
import torchdiffeq
from utils.data_utils import ReactionDataset, BEmatrix_to_mol, ps, fullSanitize
from utils.rounding import saferound_tensor
import torch.distributed as dist
from train import init_model, init_loader
from model.oxd_states import OxdStatesCalculator
from utils.train_utils import log_rank_0, setup_logger, log_args
from other.settings import Args
from collections import defaultdict
from multiprocessing import Pool
import time
import math
from rdkit.Chem import AllChem, rdchem
import iteround

def append_tensor(big, new):
    if big is None:
        return new  # first batch

    b1, n1, d = big.shape
    b2, n2, _ = new.shape
    N = max(n1, n2)

    if n1 < N:
        pad_big = (0, 0, 0, N - n1)
        big = torch.nn.functional.pad(big, pad_big)

    if n2 < N:
        pad_new = (0, 0, 0, N - n2)
        new = torch.nn.functional.pad(new, pad_new)

    return torch.cat([big, new], dim=0)


def get_embeds(args, model, flow, data_loader, iter_count=np.inf):
    model.eval()
    
    big_embeds = None

    with torch.no_grad():
        log_rank_0('Start ODE Prediction...')
        t = 0
        t = torch.tensor(t, dtype=torch.float32).to(args.device)
            
        for batch_idx, data_batch in enumerate(data_loader):
            data_batch.to(args.device)
            src_data_indices = data_batch.src_data_indices
            y = data_batch.src_token_ids
            y_len = data_batch.src_lens
            y_emb = model.id2emb(y)
            x0 = data_batch.src_matrices
            calc = OxdStatesCalculator(args)
            if args.oxd_states:
                a_nums = data_batch.src_a_nums
                a_masks = data_batch.src_a_masks

            # x1 = data_batch.tgt_matrices
            matrix_masks = data_batch.matrix_masks
            #print(matrix_masks)
            
            batch_size, n, n = x0.shape
            if args.pred_del_BE:
                x0_sample = flow.sample_noised_zero_like(x0)
                x0_sample = x0_sample.masked_fill(~(matrix_masks.bool()), 0) # ode initial step has RMS norm thus padding nan has to be swap to 0
                x0_0 = x0.masked_fill(~(matrix_masks.bool()), 0) 
                
                if args.oxd_states:
                    a_nums = a_nums.masked_fill(~(a_masks.bool()), 0)

                torch.cuda.empty_cache()

                if args.oxd_states:
                    oxd_states = calc.forward(x0_sample + x0_0, a_nums)
                    embeds = model.get_embeds(y_emb, y_len, x0_0, t, oxd_states=oxd_states, del_BE=x0_sample)

                else:
                    embeds = model.get_embeds(y_emb, y_len, x0_0, t, del_BE=x0_sample)
                    
            else:
                x0_sample = flow.sample_be_matrix(x0)
                x0_sample = x0_sample.masked_fill(~(matrix_masks.bool()), 0) # ode initial step has RMS norm thus padding nan has to be swap to 0
                
                if args.oxd_states:
                    a_nums = a_nums.masked_fill(~(a_masks.bool()), 0)

                if args.oxd_states:
                    oxd_states = calc.forward(x0_sample, a_nums)
                    embeds = model.get_embeds(y_emb, y_len, x0_sample, t, oxd_states=oxd_states)
                    
                else: 
                    embeds = model.get_embeds(y_emb, y_len, x0_sample, t)
                    
                    
            for i in range(len(embeds)):
                if big_embeds is None:
                    big_embeds = [None] * len(embeds)
                
                big_embeds[i] = append_tensor(big_embeds[i], embeds[i])
                    
        return big_embeds
            
            


def umap_embed_all(embed_tensor):
    b, n, dim = embed_tensor.shape
    embed_tensor_flat = embed_tensor.reshape(b*n, dim)
    mask = (embed_tensor_flat.abs().sum(dim=1) != 0)
    embed_tensor_flat = embed_tensor_flat[mask]
    embed_tensor_np = embed_tensor_flat.cpu().numpy() 
    import umap
    
    reducer = umap.UMAP(n_components=2, 
        n_neighbors=15,       # Number of neighbors for local structure (default 15)
        min_dist=0.1,         # Minimum distance between points in embedding
        metric='euclidean',   # Distance metric
        random_state=42,      # For reproducibility
        spread=1.0)
        
    return(reducer.fit_transform(embed_tensor_np))

def umap_embed_one_only(embed_tensor):
    b, n, dim = embed_tensor.shape
    embed_tensor_flat = embed_tensor[:, 0, :]
    mask = (embed_tensor_flat.abs().sum(dim=1) != 0)
    embed_tensor_flat = embed_tensor_flat[mask]
    embed_tensor_np = embed_tensor_flat.cpu().numpy() 
    import umap
    
    reducer = umap.UMAP(n_components=2, 
        n_neighbors=50,       # Number of neighbors for local structure (default 15)
        min_dist=0.2,         # Minimum distance between points in embedding
        metric='euclidean',   # Distance metric
        random_state=42,      # For reproducibility
        spread=1.0)
        
    return(reducer.fit_transform(embed_tensor_np))

def main(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device

    print("Local rank:", args.local_rank)
    print("Available devices:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(f"[rank{os.environ.get('LOCAL_RANK', '?')}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[rank{os.environ.get('LOCAL_RANK', '?')}] torch sees {torch.cuda.device_count()} devices")

    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend, init_method='env://', timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    phase = "test"
    checkpoints = [os.path.join(args.model_path, args.model_name)]


    for ckpt_i, checkpoint in enumerate(checkpoints):
        state = torch.load(checkpoint, weights_only=False, map_location=device)
        pretrain_args = state["args"]
        pretrain_args.load_from = None
        pretrain_args.device = device
        
        pretrain_state_dict = state["state_dict"]
        pretrain_args.local_rank = args.local_rank

        attn_model, flow, state = init_model(pretrain_args)
        if hasattr(attn_model, "module"):
            attn_model = attn_model.module        # unwrap DDP attn_model to enable accessing attn_model func directly

        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        attn_model.load_state_dict(pretrain_state_dict)
        log_rank_0(f"Loaded pretrained state_dict from {checkpoint}")

        os.makedirs(args.result_path, exist_ok=True)
        results_path = os.path.join(args.result_path, f'{phase}-{args.sample_size}-{checkpoint.split(".")[-2]}.txt') # CHANGE
    
        log_rank_0("Starting new run...")
        file_mod = 'w'
        start = 0

        with open(args.test_path, 'r') as test_o:
            test_smiles_list = test_o.readlines()[start:]
        
        assert len(test_smiles_list) > 0, "Nothing to do inference"
        
        test_dataset = ReactionDataset(args, test_smiles_list)
        test_loader = init_loader(args, test_dataset,
                                batch_size=args.test_batch_size,
                                shuffle=False, epoch=None, use_sort=False,
                                drop_last=False)
        
        embeds = get_embeds(args, attn_model, flow, test_loader)
        umapped = [umap_embed_one_only(i) for i in embeds]


        np.savez("umap_embeddings.npz", *umapped)


        
if __name__ == "__main__":
    args = Args
    args.local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else -1
    logger = setup_logger(args, "eval")
    log_args(args, 'evaluation') 
    main(args)
