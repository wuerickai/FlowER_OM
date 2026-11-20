"""Exact overlap-limited oracle top-k ceilings for contiguous test pathways.

Canonicalize molecules without maps/stereochemistry, retaining explicit H and
bond/charge information. Parse without sanitization to support metal complexes.
For each input allow at most k distinct products, shared across all pathways.
Maximize the number of fully covered pathways using a binary linear program.
These are oracle ceilings, independent of model sample counts and tie ranking.
"""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from rdkit import Chem
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def canonical(smiles):
    # Drop mapping before caching chemical equivalence calculations.
    return canonical_unmapped(re.sub(r":\d+(?=\])", "", smiles))


@lru_cache(maxsize=None)
def canonical_unmapped(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"Cannot parse {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False,
                            allHsExplicit=True)


def oracle(paths, k):
    outcomes = defaultdict(set)
    frequencies = defaultdict(Counter)
    for path in paths:
        for source, target in path:
            outcomes[source].add(target)
            frequencies[source][target] += 1
    step_correct = sum(sum(sorted(c.values(), reverse=True)[:k])
                       for c in frequencies.values())
    ambiguous = {s for s, products in outcomes.items() if len(products) > k}
    requirements = Counter(frozenset((s, t) for s, t in path if s in ambiguous)
                           for path in paths)
    always = requirements.pop(frozenset(), 0)
    if not requirements:
        return step_correct, always
    # Decompose into independent components linked by ambiguous inputs.
    groups = list(requirements)
    parent = list(range(len(groups)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, group in enumerate(groups):
        for source, _ in group:
            if source in owners:
                parent[find(i)] = find(owners[source])
            owners[source] = i
    components = defaultdict(list)
    for i in range(len(groups)):
        components[find(i)].append(groups[i])
    covered = always
    for component in components.values():
        products = sorted(set().union(*component))
        pair_index = {p: i + len(component) for i, p in enumerate(products)}
        rr, cc, vv, upper = [], [], [], []
        def constraint(entries, bound):
            row = len(upper)
            for col, value in entries:
                rr.append(row); cc.append(col); vv.append(value)
            upper.append(bound)
        for i, group in enumerate(component):
            for pair in group:
                constraint([(i, 1), (pair_index[pair], -1)], 0)
        by_source = defaultdict(list)
        for pair, index in pair_index.items():
            by_source[pair[0]].append(index)
        for indices in by_source.values():
            constraint([(i, 1) for i in indices], k)
        n = len(component) + len(products)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(upper), n)).tocsc()
        objective = np.zeros(n)
        objective[:len(component)] = [-requirements[g] for g in component]
        result = milp(objective, integrality=np.ones(n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, -np.inf, upper),
                      options={"mip_rel_gap": 0})
        if not result.success:
            raise RuntimeError(result.message)
        covered += round(-result.fun)
    return step_correct, covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data/test.txt")
    args = parser.parse_args()
    paths, previous = [], None
    for i, line in enumerate(args.test.open(), 1):
        if not line.strip():
            continue
        reaction, pid = line.rstrip().rsplit("|", 1)
        source, target = reaction.split(">>")
        if pid != previous:
            paths.append([])
            previous = pid
        paths[-1].append((canonical(source), canonical(target)))
        if i % 50000 == 0:
            print(f"Canonicalized {i} test rows", flush=True)
    total_steps = sum(map(len, paths))
    rows = []
    for k in range(1, 6):
        step, pathway = oracle(paths, k)
        row = dict(k=k, correct_steps=step, total_steps=total_steps,
                   step_ceiling=step / total_steps, correct_pathways=pathway,
                   total_pathways=len(paths), pathway_ceiling=pathway / len(paths))
        rows.append(row)
        print(row, flush=True)
    output = ROOT / "output.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
