#!/usr/bin/env python3
# ============================================================================
# runLoop.py
# ============================================================================
# 手順 2 -> 3 -> 4 のループ: subtractFD.py --yields -> omnifold_unfold.py -> truthFit.py
# 1 周目の yields は今の ../output/truthFit/yields_reco.txt (前回の truthFit の結果)。
# N_coh^reco, N_incoh^reco の前の周からの変化が全クラスで tol 未満になったら止める。
# 各周の yields は ../output/truthFit/history/yields_iter<i>.txt に残す。
# 実行: python3 runLoop.py [--maxIter 6] [--tol 0.02]   (alienv の O2Physics 環境で)
# ============================================================================
import argparse
import os
import shutil
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
yieldsFile = os.path.join(here, "..", "output", "truthFit", "yields_reco.txt")
historyDir = os.path.join(here, "..", "output", "truthFit", "history")


def readYields(path):
  return {s[0]: (float(s[1]), float(s[2])) for s in (l.split() for l in open(path)) if s and not s[0].startswith("#")}


def run(script, *args):
  print(f"--- {script} {' '.join(args)}", flush=True)
  subprocess.run([sys.executable, os.path.join(here, script), *args], check=True, cwd=here,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--maxIter", type=int, default=6)
  ap.add_argument("--tol", type=float, default=0.02)
  args = ap.parse_args()
  os.makedirs(historyDir, exist_ok=True)
  shutil.copy(yieldsFile, f"{historyDir}/yields_iter0.txt")
  prev = readYields(yieldsFile)
  for it in range(1, args.maxIter + 1):
    print(f"=== iteration {it}", flush=True)
    run("subtractFD.py", "--yields", yieldsFile)
    run("omnifold_unfold.py")
    run("truthFit.py")
    shutil.copy(yieldsFile, f"{historyDir}/yields_iter{it}.txt")
    cur = readYields(yieldsFile)
    maxChange = 0.0
    for c in cur:
      ch = [(cur[c][k] - prev[c][k]) / max(prev[c][k], 1.0) for k in (0, 1)]
      maxChange = max(maxChange, *map(abs, ch))
      print(f"  {c:5s} coh {prev[c][0]:8.0f} -> {cur[c][0]:8.0f} ({ch[0]:+.3f})   incoh {prev[c][1]:8.0f} -> {cur[c][1]:8.0f} ({ch[1]:+.3f})")
    prev = cur
    if maxChange < args.tol:
      print(f"converged after {it} iterations (max change {maxChange:.3f} < {args.tol})")
      return
  print(f"not converged after {args.maxIter} iterations (last max change {maxChange:.3f})")


if __name__ == "__main__":
  main()
