#!/usr/bin/env python3
# ============================================================================
# compareUnfold.py
# ============================================================================
# OmniFold の結果を従来の Bayes / SVD のアンフォールド (AnalysisNote) の N_unfold と比べる。
#   どちらも gen fiducial (truth のミューオンがアクセプタンス内) の incoherent + ND の数 (A で割る前)。
#   OmniFold: unfolded.root の w_unfolded から, 真値レベルのフィットの coherent 成分 (N_coh x coherent の gen の形) を引く。
#             誤差はブートストラップのレプリカの標準偏差 (coherent の引き算は中心値で固定)。
#   Bayes / SVD: ../Unfold_win2p0-3p3_Expo3_coh0p1/Simple{Bayes,SVD}/*_accCorr.txt の N_unfold。
#   ビン: pT 0.3-1.2 GeV/c, 0.1 GeV/c 幅。
# 出力 ../output/compareUnfold/<class>.png, <class>.txt
# 実行: python3 compareUnfold.py   (alienv の O2Physics 環境で)
# ============================================================================
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT

here = os.path.dirname(os.path.abspath(__file__))
outputDir = os.path.join(here, "..", "output")
anDir = "/home/takuma/alice/upc-analysis-run3/output/AnalysisNote/Unfold_win2p0-3p3_Expo3_coh0p1"
anClass = {"All": "all", "0n0n": "0n0n", "Xn0n": "Xn0n", "0nXn": "0nXn", "XnXn": "XnXn"}
edges = np.round(np.arange(0.3, 1.2001, 0.1), 2)


def readAN(method, name):
  """(N_unfold, 誤差, ラベル) を pT 0.3-1.2 の 0.1 GeV/c ビンで。"""
  path = glob.glob(f"{anDir}/Simple{method}/Simple{method}Data_{anClass[name]}_*_accCorr.txt")[0]
  lines = open(path).read().splitlines()
  label = method + " " + lines[0].split(":")[1].split(",")[0].replace(method, "").strip()
  n, e = {}, {}
  for l in lines[2:]:
    s = l.replace("(", " ").replace(")", " ").split()
    lo = float(s[0].split("-")[0])
    n[round(lo, 2)], e[round(lo, 2)] = float(s[1]), float(s[2])
  return np.array([n[lo] for lo in edges[:-1]]), np.array([e[lo] for lo in edges[:-1]]), label


def omnifold(name):
  a = ROOT.RDataFrame("gen", f"{outputDir}/omnifold/{name}/unfolded.root").AsNumpy(["pt", "component", "w_prior", "w_unfolded"])
  total = np.histogram(a["pt"], edges, weights=a["w_unfolded"])[0]
  # coherent 成分: truthFit.py と同じ形 (coherent MC の gen, pT < 1.2 で規格化) x N_coh
  r = open(f"{outputDir}/truthFit/{name}/fitResults.txt").read().split("\n")
  fit = dict(zip(r[0].split(), map(float, r[1].split())))
  coh = (a["component"] == 0) & (a["pt"] < 1.2)
  shape = np.histogram(a["pt"][coh], edges, weights=a["w_prior"][coh])[0] / a["w_prior"][coh].sum()
  nCoh = fit["N_coh"] * shape
  # ブートストラップ (20 MeV/c のビン -> 0.1 GeV/c に足す)
  z = np.load(f"{outputDir}/bootstrap/{name}/replicas.npz")
  idx = np.searchsorted(np.round(z["bins"], 3), edges)
  reps = np.add.reduceat(z["hists"], idx[:-1], axis=1)[:, :len(edges) - 1]
  return total - nCoh, reps.std(axis=0, ddof=1), nCoh


def main():
  outDir = os.path.join(outputDir, "compareUnfold")
  os.makedirs(outDir, exist_ok=True)
  c, hw = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
  for name in anClass:
    of, ofErr, nCoh = omnifold(name)
    an = [readAN(m, name) for m in ("Bayes", "SVD")]
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=0))
    ax.errorbar(c, of, ofErr, hw, fmt="o", color="black", label="OmniFold (- coherent fit)")
    for (n, e, lab), col, dx in zip(an, ("blue", "red"), (-0.015, 0.015)):
      ax.errorbar(c + dx, n, e, hw, fmt="s", ms=4, color=col, label=lab)
      rx.errorbar(c + dx, n / of, e / of, fmt="s", ms=4, color=col)
    rx.errorbar(c, np.ones_like(of), ofErr / of, hw, fmt="none", color="black")
    ax.set_yscale("log")
    ax.set_ylabel(r"$N_\mathrm{unfold}$ / (100 MeV/$c$)")
    ax.set_title(f"{name}, incoherent + ND (gen fiducial)")
    ax.legend(frameon=False)
    rx.axhline(1, color="gray", lw=1)
    rx.set_ylim(0.5, 1.5)
    rx.set_ylabel("/ OmniFold")
    rx.set_xlim(edges[0], edges[-1])
    rx.set_xlabel(r"$p_\mathrm{T}^\mathrm{gen}$ (GeV/$c$)")
    fig.savefig(f"{outDir}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    with open(f"{outDir}/{name}.txt", "w") as fh:
      fh.write(f"ptLo ptHi OmniFold err coherentSubtracted {an[0][2].replace(' ', '_')} err {an[1][2].replace(' ', '_')} err\n")
      for i in range(len(c)):
        fh.write(f"{edges[i]:.1f} {edges[i + 1]:.1f} {of[i]:.1f} {ofErr[i]:.1f} {nCoh[i]:.1f} "
                 f"{an[0][0][i]:.1f} {an[0][1][i]:.1f} {an[1][0][i]:.1f} {an[1][1][i]:.1f}\n")
    print(f"[{name}] sum 0.3-1.2: OmniFold {of.sum():.0f} +- {np.sqrt((ofErr ** 2).sum()):.0f} (coh subtracted {nCoh.sum():.0f}), "
          + ", ".join(f"{lab} {n.sum():.0f} ({n.sum() / of.sum():.3f})" for n, e, lab in an))


if __name__ == "__main__":
  main()
