"""Train only the intended all-layer zigzag model; preserve top-only variant."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import subprocess
import platform
import torch
import transformers
from run_recurrent import initialize_worker,train_one,check_pairs

ROOT=Path(__file__).parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'results-zigzag');args=parser.parse_args()
    cfg=json.loads((ROOT/'zigzag_config.json').read_text());out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    path=out/'runs.json';source={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('model.py','zigzag.py','two_sweep.py','diagonal_recurrent.py','run.py','chunk_control.py','run_recurrent.py','recurrent_evaluation.py','run_zigzag.py','zigzag_config.json')}
    data={'config':cfg,'source_sha256':source,'results':[],'wall_seconds':0.,
          'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version()}
    if path.exists():
        data=json.loads(path.read_text());assert data['config']==cfg and data['source_sha256']==source
    prior=data['wall_seconds'];start=time.perf_counter()
    jobs=[('probe',s,c) for s in cfg['seeds'] for c in cfg['conditions']]
    done={(r['seed'],r['condition']) for r in data['results']}
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(cfg,str(out))) as pool:
        for job in jobs:
            if (job[1],job[2]['name']) in done:continue
            row=pool.submit(train_one,job).result();data['results'].append(row)
            check_pairs(data['results'],cfg)
            data['wall_seconds']=prior+time.perf_counter()-start
            tmp=out/'runs.tmp.json';tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
            print(f'{len(data["results"])}/1 {row["condition"]}: {row["rolling_metrics"]["280"]}',flush=True)
    assert len(data['results'])==1
    check_pairs(data['results'],cfg)
    previous=json.loads((ROOT/'results-lateral-slots/runs.json').read_text())['results']
    for old in previous:
        for field in ('initialization_sha256','training_sequences_sha256','training_labels_sha256','parameters','processed_tokens','supervised_targets','optimizer_updates'):
            assert old[field]==data['results'][0][field],field
    data['wall_seconds']=prior+time.perf_counter()-start;path.write_text(json.dumps(data,indent=2))
    print('PASS: matched initialization, inputs, labels, parameters, tokens and updates; zigzag run complete; no control models retrained.',flush=True)

if __name__=='__main__':main()
