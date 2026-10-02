#!/usr/bin/env python3
"""Prepare bounded Halo case configurations; never run or download a model."""
import argparse
import json
from pathlib import Path
from halo import config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', required=True, type=Path)
    p.add_argument('--destination', required=True, type=Path)
    args = p.parse_args()
    base = config(args.template)
    if args.destination.exists():
        p.error('--destination must be a new directory')
    cases = [(f'fresh-{length}-c{chunk}', length, length, length+129, chunk, chunk)
             for length in (32768, 65536, 131072) for chunk in (2048, 4096)]
    cases += [('short-2048',2048,2048,32897,2048,2048),
              ('short-4096',4096,4096,32897,4096,4096),
              ('incremental-65536-c2048',2048,65536,65665,2048,2048),
              ('incremental-65536-c4096',4096,65536,65665,4096,4096)]
    args.destination.mkdir(parents=True)
    for name, start, end, capacity, chunk, step in cases:
        cfg = dict(base, context_start=start, context_max=end, context_alloc=capacity,
                   prefill_chunk=chunk, step_incr=step, gen_tokens=16)
        env = dict(base.get('environment', {}))
        # Bind native cache warmup geometry and require actual serialized restore.
        env['DS4_METAL_PREFILL_CHUNK'] = str(chunk)
        env['DS4_BENCH_SNAPSHOT_MAX_BYTES'] = 'unlimited'
        lines = [f'{k} = {json.dumps(v, ensure_ascii=False)}' for k,v in cfg.items() if k != 'environment']
        lines += ['', '[environment]'] + [f'{k} = {json.dumps(v)}' for k,v in sorted(env.items())]
        path = args.destination / (name+'.toml')
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        config(path)
        print(path.name)


if __name__ == '__main__':
    main()
