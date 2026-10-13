"""Small, reproducible thread-count benchmark; not a hardware performance study."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from model import WindowConfig, WindowLM
from run import probe_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('diagonal_benchmark_reproduction.json'))
    args = parser.parse_args()
    cfg = json.loads(Path('lateral_slots_config.json').read_text())
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    measurements = []
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        torch.manual_seed(cfg['seeds'][0])
        model = WindowLM(WindowConfig(vocab_size=32, width=cfg['width'],
            layers=cfg['layers'], heads=cfg['heads'], window=32,
            max_positions=cfg['max_positions'], recurrent_attention_slot=True,
            recurrent_attention_slots=2, recurrent_execution='diagonal'))
        generator = torch.Generator().manual_seed(cfg['seeds'][0]+1000)
        x, labels, _ = probe_batch(cfg['batch_size'], generator,
            sequence_length=cfg['sequence_length'], distances=cfg['distances'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, betas=(.9,.95))
        timings = []
        for step in range(4):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            model(x, labels=labels).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            elapsed = time.perf_counter()-start
            if step:
                timings.append(elapsed)
        measurements.append({'threads':threads, 'median_update_seconds':statistics.median(timings), 'update_seconds':timings})
        print(measurements[-1], flush=True)
    args.output.write_text(json.dumps({'method':'One warmup, three timed updates on one fixed retrieval batch; fresh matched initialization per thread count.',
        'batch':cfg['batch_size'], 'sequence_length':cfg['sequence_length'], 'slots':2,
        'layers':cfg['layers'], 'dtype':'float32', 'torch':torch.__version__, 'results':measurements}, indent=2)+'\n')


if __name__ == '__main__':
    main()
