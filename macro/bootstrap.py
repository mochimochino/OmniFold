#!/usr/bin/env python3
# ============================================================================
# bootstrap.py
# ============================================================================
# unfold した gen pT 分布の誤差を Poisson ブートストラップで出す。
#   各レプリカ: データの事象 (source 0) の重みに Poisson(1) の乱数を掛けて OmniFold をやり直す
#   (FD の MC の負の重みはそのまま; 学習の乱数の種も毎回変える -> OmniFold の実行ごとのばらつきも含む)。
#   誤差 = レプリカの標準偏差。中心値は omnifold/<class>/unfolded.root (ループの最後の結果)。
#   sWeight は固定 (sPlot のフィットはやり直さない)。
# 出力 ../output/bootstrap/<class>/: replicas.npz, pt_unfolded.png, pt_unfolded.txt
# 実行: python3 bootstrap.py [--classes ...] [--nBoot 20] [--reuse]   (--reuse: replicas.npz を読み, 中心値と図だけ作り直す)   (alienv の O2Physics 環境で)
# ============================================================================
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import tensorflow as tf

import omnifold_unfold as of
import sPlot_massfit as sp

here = os.path.dirname(os.path.abspath(__file__))
outBase = os.path.join(here, "..", "output", "bootstrap")
bins = np.linspace(0.0, sp.ptEdges[-1], 61)   # 20 MeV/c (truthFit.py と同じ)


def replicas(name, mc, gen, reco, inRange, args):
  """Poisson(1) でデータの重みを振り, OmniFold をやり直したレプリカの gen pT ヒストグラムと coh/incoh の合計。"""
  d = ROOT.RDataFrame("data", f"{of.dataBase}/{name}/dataForOmniFold.root").AsNumpy(["pairPt", "rapidity", "weight", "source"])
  xData, wData, isData = np.stack([d["pairPt"], d["rapidity"]], axis=1), d["weight"], d["source"] == 0
  nJ = of.readNJpsi(name)
  w0 = np.ones(len(mc["ptG"]))
  for comp, k in enumerate(of.mcFiles):
    sel = mc["component"] == comp
    w0[sel] = nJ[k] / (reco & sel).sum()

  hists, totals = [], []
  for r in range(args.nBoot):
    rng = np.random.default_rng(1000 + r)
    np.random.seed(1000 + r)
    tf.random.set_seed(1000 + r)
    wR = wData.copy()
    wR[isData] *= rng.poisson(1.0, isData.sum())
    print(f"[{name}] replica {r + 1}/{args.nBoot}", flush=True)
    w = w0 * of.omnifold(mc, w0, xData, wR, args.iterations)[-1]
    hists.append(np.histogram(mc["ptG"][inRange], bins, weights=w[inRange])[0])
    totals.append([w[inRange & (mc["component"] == c)].sum() for c in (0, 1)])
  hists, totals = np.array(hists), np.array(totals)
  return hists, totals


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--classes", nargs="+", choices=list(sp.classes), default=list(sp.classes))
  ap.add_argument("--nBoot", type=int, default=20)
  ap.add_argument("--iterations", type=int, default=4)
  ap.add_argument("--reuse", action="store_true", help="レプリカは replicas.npz から読む")
  args = ap.parse_args()
  if not args.reuse:
    mc = of.loadMC()
    gen, reco = mc["genOK"].astype(bool), mc["recoOK"].astype(bool)
    inRange = gen & (mc["ptG"] < bins[-1])

  for name in args.classes:
    if args.reuse:
      z = np.load(f"{outBase}/{name}/replicas.npz")
      hists, totals = z["hists"], z["totals"]
    else:
      hists, totals = replicas(name, mc, gen, reco, inRange, args)

    a = ROOT.RDataFrame("gen", f"{of.outBase}/{name}/unfolded.root").AsNumpy(["pt", "w_unfolded"])
    m = a["pt"] < bins[-1]
    nominal = np.histogram(a["pt"][m], bins, weights=a["w_unfolded"][m])[0]
    err = hists.std(axis=0, ddof=1)

    outDir = os.path.join(outBase, name)
    os.makedirs(outDir, exist_ok=True)
    np.savez(f"{outDir}/replicas.npz", bins=bins, hists=hists, totals=totals, nominal=nominal)
    with open(f"{outDir}/pt_unfolded.txt", "w") as fh:
      fh.write("ptLo ptHi nominal err replicaMean\n")
      for i in range(len(nominal)):
        fh.write(f"{bins[i]:.3f} {bins[i + 1]:.3f} {nominal[i]:.2f} {err[i]:.2f} {hists[:, i].mean():.2f}\n")

    c = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(c, bins=bins, weights=nominal, histtype="step", color="black", lw=2, label="OmniFolded")
    ax.bar(c, 2 * err, bottom=nominal - err, width=np.diff(bins), color="gray", alpha=0.4, label=f"bootstrap ({len(hists)} replicas)")
    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(r"$p_\mathrm{T}^\mathrm{gen}$ (GeV/$c$)")
    ax.set_ylabel("events / 20 MeV/$c$")
    ax.set_title(name)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{outDir}/pt_unfolded.png")
    plt.close(fig)
    tot = totals.sum(axis=1)
    print(f"[{name}] total (gen pT < {bins[-1]}) = {nominal.sum():.0f} +- {tot.std(ddof=1):.0f} "
          f"(replica mean {tot.mean():.0f}); rel. error per bin: median {np.median(err / np.maximum(nominal, 1)):.3f}", flush=True)


if __name__ == "__main__":
  main()
