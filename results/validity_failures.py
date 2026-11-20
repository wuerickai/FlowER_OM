import ast, csv, json, re
from pathlib import Path
from collections import Counter
from functools import lru_cache
from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'results.txt'
LABELS=['bonded_positive_charges','bonded_negative_charges','atom_charge_gt_plus_one','atom_charge_lt_minus_one','charged_carbon_valence_ge_four','hydrogen_or_halogen_degree_gt_one','pd_charge_plus_one','pd_charge_minus_one','rdkit_parse_or_sanitize_failure']
ps=Chem.SmilesParserParams(); ps.removeHs=False
@lru_cache(maxsize=20000)
def check(s):
    mol=Chem.MolFromSmiles(s,ps)
    labels=set(); details=[]
    if mol is None:
        labels.add('rdkit_parse_or_sanitize_failure')
        fallback=Chem.SmilesParserParams(); fallback.removeHs=False; fallback.sanitize=False
        mol=Chem.MolFromSmiles(s,fallback)
        if mol is None: return tuple(labels), []
        mol.UpdatePropertyCache(strict=False)
    def hit(label,kind,index):
        labels.add(label); details.append({'label':label,kind:index})
    for b in mol.GetBonds():
        c1=b.GetBeginAtom().GetFormalCharge(); c2=b.GetEndAtom().GetFormalCharge()
        if c1>0 and c2>0: hit(LABELS[0],'bond_index',b.GetIdx())
        if c1<0 and c2<0: hit(LABELS[1],'bond_index',b.GetIdx())
    for a in mol.GetAtoms():
        z=a.GetAtomicNum(); q=a.GetFormalCharge(); i=a.GetIdx()
        if q>1: hit(LABELS[2],'atom_index',i)
        if q< -1: hit(LABELS[3],'atom_index',i)
        if z==6 and q!=0 and a.GetTotalValence()>=4: hit(LABELS[4],'atom_index',i)
        elif z in (1,9,17,35,53) and a.GetDegree()>1: hit(LABELS[5],'atom_index',i)
        elif z==46:
            if q==1: hit(LABELS[6],'atom_index',i)
            if q== -1: hit(LABELS[7],'atom_index',i)
    return tuple(l for l in LABELS if l in labels),details

tot=Counter(); failures=Counter(); wrong_lo=Counter(); wrong_hi=Counter(); unique_fail=Counter(); ranks=Counter(); rowpresence=Counter(); combos=Counter(); n=0; ambiguous=0; samples=Counter()
with (ROOT/'labeled_candidates.csv').open('w',newline='') as out:
    writer=csv.writer(out); writer.writerow(['line','candidate','count','electron_conserved','correctness','labels','details_zero_based','smiles'])
    for line_no,line in enumerate(SOURCE.open(),1):
        if not line.strip(): continue
        if line_no % 25000 == 0: print(f"Processed {line_no:,} rows", flush=True)
        a,b,c=re.sub(r'np\.int(?:32|64)\((-?\d+)\)', lambda match: match.group(1), line.strip()).split('|',2); metrics=ast.literal_eval(a); preds=ast.literal_eval(c); n+=1
        assert len(metrics)==5 and all(x>=0 for x in metrics)
        samples[sum(metrics)]+=1
        for key,v in zip(['correct','wrong_conserved','wrong_nonconserved','no_smiles_conserved','no_smiles_nonconserved'],metrics): tot[key]+=v
        tot['nonsymmetric_unique_matrices']+=int(b)
        assert sum(p[1] for p in preds)==sum(metrics[:2])
        assert all(p[2] for p in preds)
        correct_options=[i for i,p in enumerate(preds) if metrics[0]>0 and p[1]==metrics[0]]
        assert not metrics[0] or correct_options
        ambiguous+=len(correct_options)>1
        rank=sorted([p[1] for p in preds],reverse=True).index(metrics[0])+1 if metrics[0] else 10**9
        ranks[rank]+=1
        checked=[check(p[0]) for p in preds]
        present=set()
        for i,(s,count,conserved) in enumerate(preds):
            labels,details=checked[i]; keys=set(labels)|({'any_failure'} if labels else {'passes_all_rules'})
            present.update(keys); combos[';'.join(labels) or 'passes_all_rules']+=count
            for key in keys: failures[key]+=count; unique_fail[key]+=1
            status='incorrect' if i not in correct_options else ('correct' if len(correct_options)==1 else 'ambiguous_correct_or_incorrect')
            writer.writerow([line_no,i+1,count,conserved,status,';'.join(labels) or 'passes_all_rules',json.dumps(details),s])
        for key in LABELS+['any_failure','passes_all_rules']:
            base=sum(p[1] for p,ch in zip(preds,checked) if (bool(ch[0]) if key=='any_failure' else not ch[0] if key=='passes_all_rules' else key in ch[0]))
            sub=[metrics[0]*int(bool(checked[i][0]) if key=='any_failure' else not checked[i][0] if key=='passes_all_rules' else key in checked[i][0]) for i in correct_options] or [0]
            wrong_lo[key]+=base-max(sub); wrong_hi[key]+=base-min(sub)
        for key in present: rowpresence[key]+=1
summary={'source':str(SOURCE),'rdkit_version':rdBase.rdkitVersion,'steps':n,'samples_per_step_histogram':dict(samples),'total_samples':sum(k*v for k,v in samples.items()),'standard_counts':dict(tot),'top_k_steps_optimistic_ties':{k:sum(v for r,v in ranks.items() if r<=k) for k in range(1,6)},'steps_any_correct':n-ranks[10**9],'ambiguous_correct_identity_steps':ambiguous,'stored_candidates':sum(unique_fail[k] for k in ['any_failure','passes_all_rules']),'stored_weighted_labels':dict(failures),'stored_candidate_labels':dict(unique_fail),'incorrect_conserved_labels_lower':dict(wrong_lo),'incorrect_conserved_labels_upper':dict(wrong_hi),'steps_with_label':dict(rowpresence),'weighted_label_combinations':dict(combos)}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
