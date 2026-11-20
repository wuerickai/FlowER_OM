import os
import time
import pickle
import signal
import traceback
from collections import defaultdict
from functools import lru_cache

import numpy as np
import networkx as nx
import torch
import torch.multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem  # kept for downstream compatibility

from utils.data_utils import (
    ReactionDataset, BEmatrix_to_mol, fullSanitize, standardize_smiles,
)
from utils.train_utils import log_rank_0, setup_logger, log_args
from train import init_model, init_loader
from eval_multiGPU import predict_batch_new
from other.settings import Args

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --- shared parser params (singletons) ---------------------------------------
_PS = Chem.SmilesParserParams()
_PS.removeHs = True
_PS.sanitize = True

_PS_KEEP_H = Chem.SmilesParserParams()
_PS_KEEP_H.removeHs = False
_PS_KEEP_H.sanitize = True


# --- small helpers -----------------------------------------------------------
@lru_cache(maxsize=200_000)
def canonicalize_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi, _PS)
    smi = standardize_smiles(mol)
    smi = fullSanitize(smi)
    return Chem.MolToSmiles(
        Chem.MolFromSmiles(smi, _PS),
        isomericSmiles=False, canonical=True, allHsExplicit=True,
    )


def standardize_smiles_mapped(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)


def is_valid_mol(mol) -> bool:
    for bond in mol.GetBonds():
        c1 = bond.GetBeginAtom().GetFormalCharge()
        c2 = bond.GetEndAtom().GetFormalCharge()
        if (c1 > 0 and c2 > 0) or (c1 < 0 and c2 < 0):
            return False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() > 1 or atom.GetFormalCharge() < -1:
            return False
        a = atom.GetAtomicNum()
        if a == 6:                                  # C
            if atom.GetFormalCharge() != 0 and atom.GetTotalValence() >= 4:
                return False
        elif a in (1, 9, 17, 35, 53):                  # F/Cl/Br/I
            if atom.GetDegree() > 1:
                return False
        elif a == 46:                               # Pd
            if atom.GetFormalCharge() == 1 or atom.GetFormalCharge() == -1:
                return False
    return True


def reactant_process(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol, explicitOnly=False)
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(idx + 1)
        return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)
    except Exception:
        print(f"reactant_process failed on: {smi}")
        raise


# --- selection ---------------------------------------------------------------
def select_efficient(args, frontiers_dict, graph_list):
    """Top beam_size frontiers per graph by cumulative probability; prune dead leaves."""
    filtered_frontiers_dict = {}
    for g_idx, frontiers in frontiers_dict.items():
        graph, _root, _ = graph_list[g_idx]
        scored = [(f, -graph.nodes[f]['cum_prob']) for f in frontiers]
        scored.sort(key=lambda x: x[1])
        keep = [f for f, _ in scored[:args.beam_size]]
        prune = [f for f, _ in scored[args.beam_size:]]
        for node in prune:
            if graph.out_degree(node) == 0:
                graph.remove_node(node)
        filtered_frontiers_dict[g_idx] = keep
    return filtered_frontiers_dict


# --- model inference ---------------------------------------------------------
def _run_predict(args, batch_idx, data_batch, model, flow, chunks):
    """Forward (and optional reverse) prediction, interleaved per-item.

    Returns (all_matrices, per_item). reverse_aug keeps per_item equal to
    sample_size by allocating half the budget to each direction.
    """
    if args.reverse_aug and args.sample_size % 2 != 0:
        raise ValueError("reverse_aug requires an even sample_size")

    fwd = predict_batch_new(args, batch_idx, data_batch, model, flow, chunks)
    if not args.reverse_aug:
        return fwd, args.sample_size

    rev = predict_batch_new(args, batch_idx, data_batch, model, flow, chunks, reverse=True)
    per_direction = args.sample_size // 2
    batch_size = data_batch.src_matrices.shape[0]
    n = fwd.shape[-1]

    expected = batch_size * per_direction
    if fwd.shape[0] != expected or rev.shape[0] != expected:
        raise RuntimeError(
            "reverse_aug prediction count mismatch: "
            f"expected {expected} per direction, got fwd={fwd.shape[0]} rev={rev.shape[0]}"
        )

    fwd = fwd.view(batch_size, per_direction, n, n)
    rev = rev.view(batch_size, per_direction, n, n)
    # Interleave: per-item block is [fwd_i ; rev_i] of total size sample_size.
    combined = torch.cat([fwd, rev], dim=1).reshape(-1, n, n)
    return combined, args.sample_size


def expand(args, model, flow, data_loader):
    """Run the model over `data_loader`, return ({reac_smi: {pred_smi: count}}, per_item)."""
    overall_dict = {}
    per_item = args.sample_size

    with torch.inference_mode():
        for batch_idx, data_batch in enumerate(data_loader):
            data_batch.to(args.device)
            y_len = data_batch.src_lens
            x0 = data_batch.src_matrices
            src_smiles_list = data_batch.src_smiles_list
            src_dative_donors = data_batch.src_d_donors
            src_not_donors = data_batch.src_not_donors
            batch_size, n, _ = x0.shape

            chunks = 1 if (batch_size * n * n) <= 5 * 360 * 360 else 2
            product_BE_matrices, per_item = _run_predict(
                args, batch_idx, data_batch, model, flow, chunks
            )
            expected_total = batch_size * per_item
            if product_BE_matrices.shape[0] != expected_total:
                raise RuntimeError(
                    "prediction batch layout mismatch: "
                    f"expected {expected_total}, got {product_BE_matrices.shape[0]}"
                )
            product_BE_matrices_batch = torch.split(product_BE_matrices, per_item)

            for idx in range(batch_size):
                reac_smi = src_smiles_list[idx]
                matrices_item = product_BE_matrices_batch[idx]
                donors = src_dative_donors[idx]
                not_donors = src_not_donors[idx]
                reac_mol = Chem.MolFromSmiles(reac_smi, _PS_KEEP_H)

                matrices, counts = torch.unique(matrices_item, dim=0, return_counts=True)
                matrices = matrices.cpu().numpy()
                counts = counts.cpu().numpy()

                num_nodes = y_len[idx]
                reac_be_matrix = x0[idx][:num_nodes, :num_nodes].detach().cpu().numpy()

                pred_smis_dict = defaultdict(int)
                for i in range(matrices.shape[0]):
                    pred_prod_be_matrix = matrices[i][:num_nodes, :num_nodes]
                    count = counts[i]
                    assert pred_prod_be_matrix.shape == reac_be_matrix.shape, \
                        "pred and reac shape mismatch"
                    try:
                        pred_mol = BEmatrix_to_mol(
                            reac_mol, pred_prod_be_matrix, donors, not_donors=not_donors
                        )
                        if not is_valid_mol(pred_mol):
                            raise ValueError("Invalid mol")
                        pred_smi = standardize_smiles_mapped(pred_mol)
                        pred_smi = fullSanitize(pred_smi)
                        if pred_smi is None:
                            raise ValueError("Empty pred smi")
                        pred_smi = Chem.MolToSmiles(
                            Chem.MolFromSmiles(pred_smi, _PS_KEEP_H),
                            isomericSmiles=False, canonical=True, allHsExplicit=True,
                        )
                        pred_smis_dict[pred_smi] += int(count)
                    except Exception:
                        pass

                top = sorted(pred_smis_dict.items(), key=lambda x: x[1], reverse=True)
                overall_dict[reac_smi] = dict(top[:args.nbest])

    return overall_dict, per_item


# --- beam search (iterative) -------------------------------------------------
def beam_search(args, model, flow, frontiers_dict, graph_list, rank):
    """Iterative breadth-first beam search. Mutates graphs in graph_list."""
    iteration = 0
    while frontiers_dict:
        iteration += 1

        # Filter graphs whose frontiers have already hit max_depth.
        active = {}
        for g_idx, fs in frontiers_dict.items():
            if not fs:
                continue
            graph, root, _ = graph_list[g_idx]
            depth = graph.nodes[fs[0]].get('depth', graph.nodes[root]['depth'])
            if depth < args.max_depth:
                active[g_idx] = fs
            else:
                log_rank_0(f"[GPU {rank}] graph {g_idx} hit max_depth={args.max_depth}, excluded")

        if not active:
            log_rank_0(f"[GPU {rank}] beam search complete after {iteration - 1} iters")
            return

        smiles_list = [
            graph_list[g_idx][0].nodes[f]['smiles']
            for g_idx, fs in active.items() for f in fs
        ]
        log_rank_0(
            f"[GPU {rank}] iter={iteration} expanding {len(smiles_list)} "
            f"frontiers across {len(active)} graphs"
        )

        test_dataset = ReactionDataset(args, smiles_list, reactant_only=True)
        test_loader = init_loader(
            args, test_dataset, batch_size=args.test_batch_size,
            shuffle=False, epoch=None, use_sort=False,
            drop_last=False,
        )

        overall_dict, per_item = expand(args, model, flow, test_loader)

        new_frontiers_dict = defaultdict(list)
        for g_idx, frontiers in active.items():
            graph, root, _ = graph_list[g_idx]
            for frontier in frontiers:
                original_frontier = graph.nodes[frontier]['smiles']
                if original_frontier not in overall_dict:
                    log_rank_0(f"[GPU {rank}] WARN: frontier missing from preds: "
                               f"{original_frontier[:40]}...")
                    continue

                current_depth = graph.nodes[frontier].get(
                    'depth', graph.nodes[root]['depth']
                )
                frontier_cum_prob = graph.nodes[frontier].get('cum_prob', 1.0)
                product_info_dict = overall_dict[original_frontier]

                for rank_val, (product, count) in enumerate(product_info_dict.items()):
                    try:
                        clean_product = canonicalize_smiles(product)
                    except Exception:
                        continue

                    edge_prob = count / per_item
                    product_cum_prob = frontier_cum_prob * edge_prob

                    if graph.has_node(clean_product):
                        old = graph.nodes[clean_product].get('cum_prob', 0.0)
                        if product_cum_prob > old:
                            graph.nodes[clean_product]['cum_prob'] = product_cum_prob
                            graph.nodes[clean_product]['best_parent'] = frontier
                            graph.nodes[clean_product]['smiles'] = product
                        if not graph.has_edge(frontier, clean_product):
                            graph.add_edge(
                                frontier, clean_product,
                                rank=rank_val, count=int(count), prob=edge_prob,
                            )
                        continue

                    graph.add_node(
                        clean_product, smiles=product,
                        depth=current_depth + 1,
                        cum_prob=product_cum_prob,
                        best_parent=frontier,
                    )
                    graph.add_edge(
                        frontier, clean_product,
                        rank=rank_val, count=int(count), prob=edge_prob,
                    )
                    new_frontiers_dict[g_idx].append(clean_product)

        frontiers_dict = select_efficient(args, new_frontiers_dict, graph_list)


# --- per-worker setup --------------------------------------------------------
def _load_model(args, device):
    checkpoint = os.path.join(args.model_path, args.model_name)
    state = torch.load(checkpoint, weights_only=False, map_location=device)
    pretrain_args = state["args"]
    pretrain_args.load_from = None
    pretrain_args.device = device
    pretrain_args.local_rank = -1
    attn_model, flow, _ = init_model(pretrain_args)
    sd = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    attn_model.load_state_dict(sd)
    attn_model.eval()
    return attn_model, flow


def _build_initial_graphs(chunk):
    graph_list = []
    frontiers_dict = defaultdict(list)
    for idx, line in enumerate(chunk):
        if ">>" in line:
            ori_reactant = line.strip().split(">>")[0]
            products = line.strip().split(">>")[1].split("|")
            products = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in products]
        else:
            ori_reactant = line.strip()
            products = []
        reactant = reactant_process(ori_reactant)
        graph = nx.DiGraph()
        can_reactant = canonicalize_smiles(reactant)
        graph.add_node(can_reactant, smiles=reactant, depth=1, cum_prob=1.0)
        graph_list.append((graph, can_reactant, (ori_reactant, products)))
        frontiers_dict[idx].append(can_reactant)
    return graph_list, frontiers_dict


# --- persistent worker loop --------------------------------------------------
def worker_loop(rank, args, task_q, result_q):
    """One persistent process per GPU. Loads the model once, then drains task_q."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        args.device = device
        attn_model, flow = _load_model(args, device)
        log_rank_0(f"[GPU {rank}] worker ready")
    except Exception:
        result_q.put(("startup", rank, -1, traceback.format_exc()))
        return

    while True:
        item = task_q.get()
        if item is None:  # shutdown sentinel
            log_rank_0(f"[GPU {rank}] worker shutting down")
            return

        chunk_idx, chunk = item
        try:
            graph_list, frontiers_dict = _build_initial_graphs(chunk)
            beam_search(args, attn_model, flow, frontiers_dict, graph_list, rank)
            result_q.put(("ok", rank, chunk_idx, graph_list))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result_q.put(("err", rank, chunk_idx, "CUDA OOM"))
        except Exception:
            result_q.put(("err", rank, chunk_idx, traceback.format_exc()))


# --- success check / output --------------------------------------------------
def check_if_successful(graph, products):
    """
    Identify graph nodes that achieved a target product.

    If multiple graph nodes contain the same target as a component, the
    node with the highest `cum_prob` wins.
    """
    nodes_with_loops = set(nx.nodes_with_selfloops(graph))
    canon_products = {canonicalize_smiles(p) for p in products}
    matches = {}
    for node in graph.nodes():
        if node not in nodes_with_loops:
            continue
        node_in_products = set(canonicalize_smiles(node).split('.')) & canon_products
        for target in node_in_products:
            existing = matches.get(target)
            if existing is None or \
               graph.nodes[node].get('cum_prob', 0) > \
               graph.nodes[existing].get('cum_prob', 0):
                matches[target] = node
    return matches


def _result_path_for(args, chunk_idx):
    return os.path.join(args.result_path, f'beam_{chunk_idx}.pickle')


def _save_chunk(args, chunk_idx, graph_list):
    results = []
    for beam_idx, (graph, root, (reactant, products)) in enumerate(graph_list):
        check = check_if_successful(graph, products)
        results.append((graph, root, (reactant, products), check))
        log_rank_0(
            f"Chunk {chunk_idx} ; mol {beam_idx}: "
            f"{len(check)}/{len(products)} - {check}"
        )
    with open(_result_path_for(args, chunk_idx), "wb") as f_out:
        pickle.dump(results, f_out)


# --- main --------------------------------------------------------------------
def main_multi_gpu(args):
    start = time.time()
    world_size = torch.cuda.device_count()
    log_rank_0(f"Found {world_size} GPUs")

    with open(args.test_path, 'r') as f:
        test_smiles_list = f.readlines()

    chunk_size = max(1, args.chunk_size // world_size)
    log_rank_0(f"Chunk size: {chunk_size}")
    chunks = [
        test_smiles_list[i:i + chunk_size]
        for i in range(0, len(test_smiles_list), chunk_size)
    ]
    os.makedirs(args.result_path, exist_ok=True)

    # Resume: skip chunks that already have an output pickle.
    todo = []
    for chunk_idx, chunk in enumerate(chunks):
        if os.path.exists(_result_path_for(args, chunk_idx)):
            log_rank_0(f"Skipping already-completed chunk {chunk_idx}")
            continue
        todo.append((chunk_idx, chunk))
    log_rank_0(f"Total chunks to process: {len(todo)} / {len(chunks)}")

    if not todo:
        log_rank_0("Nothing to do.")
        return

    task_q = mp.Queue()
    result_q = mp.Queue()

    # Enqueue all work up front, plus one shutdown sentinel per worker.
    for item in todo:
        task_q.put(item)
    for _ in range(world_size):
        task_q.put(None)

    # Spawn one persistent worker per GPU.
    workers = []
    for rank in range(world_size):
        p = mp.Process(target=worker_loop, args=(rank, args, task_q, result_q))
        p.start()
        workers.append(p)

    # SIGINT handler with the live workers list (was broken in v1).
    def _sigint(_sig, _frame):
        log_rank_0("SIGINT received — terminating workers")
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=5)
        os._exit(130)
    signal.signal(signal.SIGINT, _sigint)

    # Drain results as they arrive — no per-group barrier; GPUs never wait on each other.
    remaining = len(todo)
    while remaining > 0:
        msg = result_q.get()
        tag = msg[0]
        if tag == "startup":
            _, rank, _, tb = msg
            log_rank_0(f"[GPU {rank}] STARTUP FAILED:\n{tb}")
            for p in workers:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"GPU {rank} failed to start")
        elif tag == "ok":
            _, rank, chunk_idx, graph_list = msg
            _save_chunk(args, chunk_idx, graph_list)
            remaining -= 1
            log_rank_0(
                f"[GPU {rank}] chunk {chunk_idx} done "
                f"({len(todo) - remaining}/{len(todo)}, "
                f"elapsed {time.time() - start:.1f}s)"
            )
        elif tag == "err":
            _, rank, chunk_idx, err = msg
            log_rank_0(f"[GPU {rank}] chunk {chunk_idx} FAILED:\n{err}")
            remaining -= 1
        else:
            log_rank_0(f"unknown result tag: {tag!r}")

    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            log_rank_0(f"Worker pid={p.pid} still alive after join, terminating")
            p.terminate()
            p.join(timeout=5)

    log_rank_0(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = Args
    logger = setup_logger(args, "beam")
    log_args(args, 'evaluation')
    main_multi_gpu(args)
