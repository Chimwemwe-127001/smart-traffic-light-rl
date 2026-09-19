"""Train every learner on every scenario, in several independent runs, in parallel.

    python experiments/train_all.py                  # 10 learners x 5 runs = 50 jobs
    python experiments/train_all.py --runs 2 --episodes 20 --jobs 4     # quick smoke test

Each job is one call of experiments/train.py in its own process (libsumo runs
SUMO inside the process, so jobs cannot collide). One line is printed per
finished job; a failed job prints the end of its output.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNERS = [('q_learning', 'single'), ('dqn', 'single'),
            ('q_learning', 'corridor'), ('coordinated', 'corridor'),
            ('q_learning', 'corridor_heavy'), ('coordinated', 'corridor_heavy'),
            ('q_learning', 'four_way'), ('dqn', 'four_way'),
            ('q_learning', 'lusaka'), ('dqn', 'lusaka')]


def train(agent, scenario, run, episodes):
    cmd = [sys.executable, os.path.join(REPO, 'experiments', 'train.py'), '--agent', agent,
           '--scenario', scenario, '--run', str(run), '--episodes', str(episodes)]
    start = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return agent, scenario, run, p.returncode, p.stdout + p.stderr, time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--episodes', type=int, default=900)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    # the slow scenarios first, so the pool is not left waiting on them at the end
    order = {'lusaka': 0, 'corridor_heavy': 1, 'corridor': 2, 'four_way': 3, 'single': 4}
    jobs = sorted([(a, s, r) for r in range(args.runs) for a, s in LEARNERS], key=lambda j: order[j[1]])
    print(f'{len(jobs)} jobs on {args.jobs} workers', flush=True)
    failed, start = 0, time.time()
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(train, a, s, r, args.episodes) for a, s, r in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            agent, scenario, run, code, out, secs = fut.result()
            last = out.strip().splitlines()[-1] if out.strip() else ''
            print(f'[{done}/{len(jobs)}] {agent} {scenario} run {run}: '
                  f'{"ok" if code == 0 else "FAILED"} in {secs / 60:.1f} min  {last}', flush=True)
            if code != 0:
                failed += 1
                print('\n'.join(out.strip().splitlines()[-15:]), flush=True)
    print(f'finished in {(time.time() - start) / 60:.0f} min, {failed} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
