"""Reproduce twelve V1 text runs, instrumenting the unchanged trainer with loss checkpoints."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import inspect
import json
import multiprocessing
from pathlib import Path
import time
import torch
import run_v1 as trainer

ROOT=Path(__file__).parent
OUT=ROOT/'results-curves'


def record_curve(model, step, batch_loss=None):
    if batch_loss is not None:
        trainer.WORKER['batch_losses'].append({'step':step,'nll':batch_loss})
    if step % 20: return
    heldout=trainer.evaluate(model,'text',trainer.WORKER['validations']['text'])['heldout_nll']
    train=trainer.evaluate(model,'text',trainer.WORKER['train_eval'])['heldout_nll']
    trainer.WORKER['curve'].append({'step':step,'processed_tokens':step*16*128,
                                  'train_eval_nll':train,'validation_nll':heldout})
    model.train()


def initialize(config):
    trainer.initialize_worker(config,str(OUT/'checkpoints'))
    train=trainer.WORKER['train']; g=torch.Generator().manual_seed(20261012)
    starts=torch.randint(0,len(train)-128,(32,16),generator=g)
    trainer.WORKER['train_eval']=[torch.stack([train[s:s+128] for s in row]) for row in starts]
    # Use the V1 trainer verbatim except for read-only evaluation and logging hooks.
    source=inspect.getsource(trainer.train_one)
    source=source.replace('history=[]; targets=0; start=time.perf_counter()',
                          'record_curve(model,0)\n    history=[]; targets=0; start=time.perf_counter()')
    source=source.replace('optimizer.step()',
                          'optimizer.step()\n        record_curve(model,step+1,float(loss.detach()))')
    namespace={**vars(trainer),'record_curve':record_curve}
    exec(compile(source,'instrumented_v1_trainer','exec'),namespace)
    trainer.WORKER['instrumented_train']=namespace['train_one']


def run(job):
    trainer.WORKER['curve']=[]; trainer.WORKER['batch_losses']=[]
    row=trainer.WORKER['instrumented_train'](job)
    reference=json.loads((ROOT/'results-v1/runs.json').read_text())
    ref=next(r for r in reference['results'] if r['task']==row['task'] and r['seed']==row['seed'] and r['condition']==row['condition'])
    for field in ('initialization_sha256','training_sequences_sha256','training_labels_sha256','history'):
        assert row[field]==ref[field],field
    assert row['rolling_metrics']==ref['rolling_metrics'],'Final metrics changed'
    original=trainer.WindowLM.from_pretrained(ROOT/'results-v1'/ref['checkpoint'])
    reproduced=trainer.WindowLM.from_pretrained(OUT/'checkpoints'/row['checkpoint'])
    assert trainer.digest(original.state_dict())==trainer.digest(reproduced.state_dict()),'Weights changed'
    return {'seed':row['seed'],'condition':row['condition'],'curve':trainer.WORKER['curve'],
            'batch_losses':trainer.WORKER['batch_losses'],'exact_v1_reproduction':True}


def main():
    OUT.mkdir(exist_ok=True)
    config=json.loads((ROOT/'v1_config.json').read_text())
    jobs=[('text',s,c) for s in config['seeds'] for c in config['conditions'] if c['name'] in ('full-128','sliding-32')]
    data={'config':config,'checkpoint_interval':20,'train_eval_seed':20261012,
          'description':'Fixed 512 training-set and 512 held-out sequences scored at updates 0,20,...,200. All final weights exactly match V1.',
          'results':[]}
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(config,)) as pool:
        for f in as_completed([pool.submit(run,job) for job in jobs]):
            row=f.result(); data['results'].append(row)
            (OUT/'curves.json').write_text(json.dumps(data,indent=2))
            print(f'{len(data["results"])}/12 {row["condition"]} seed={row["seed"]}: exact weights and metrics match V1',flush=True)
    data['wall_seconds']=time.perf_counter()-start
    (OUT/'curves.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__': main()
