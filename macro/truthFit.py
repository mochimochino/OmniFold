#!/usr/bin/env python3
# ============================================================================
# truthFit.py
# ============================================================================
# 手順 4: 真値レベルのフィット。OmniFold で unfold した gen pT 分布 (w_unfolded) を
#   N_coh x (coherent MC の gen の形) + N_incoh x (incoherent MC の gen の形) + N_ND x H1 (解離)
# でフィットし, 直接生成の N_J/psi^coh, N_J/psi^incoh (真値レベル) を出す。
#   H1: dN/dpT ∝ pT (1 + b/n pT^2)^-n, n = 3.58 固定。b もクラスごとに固定 (pT 0-3.6 GeV/c の
#       pT テンプレートフィットの値; pT < 1.2 では b が決まらず, 解離成分が非コヒーレントの形をまねるため)
#   --bShift s で b を s sigma 動かす (系統誤差用; 出力は truthFit_bShift<s>/)
#   範囲 gen pT < ptMax (= sPlot の pT の上限)。ビンごとの誤差は sqrt(sum w^2)。
# 手順 2 に戻すため, reco の収量に直して書く:
#   N_x^reco = eps_x x N_x^truth(全 gen pT),  eps_x = (reco の選択を通る MC 事象数) / (gen の MC 事象数)
#   (omnifold_unfold.py の初期の重み w_prior = N_x^reco,prior / n_reco,x なので eps_x = N_x^reco,prior / sum w_prior,x)
# 出力 ../output/truthFit/:
#   <class>/truthFit.png, <class>/fitResults.txt
#   yields_reco.txt   subtractFD.py --yields に渡すファイル (<class> <N_coh^reco> <N_incoh^reco>)
# 実行: python3 truthFit.py [--classes ...]   (alienv の O2Physics 環境で)
# ============================================================================
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from scipy.optimize import least_squares

import sPlot_massfit as sp

here = os.path.dirname(os.path.abspath(__file__))
unfoldBase = os.path.join(here, "..", "output", "omnifold")
fdBase = os.path.join(here, "..", "output", "fdSubtracted")
outBase = os.path.join(here, "..", "output", "truthFit")
ptMax = sp.ptEdges[-1]
bins = np.linspace(0.0, ptMax, 61)   # 20 MeV/c
nND = 3.58
# H1 の b はクラスごとに固定: pT 0-3.6 GeV/c の pT テンプレートフィット (解離成分だけの高 pT で決まる) の値
templateDir = "/home/takuma/alice/upc-analysis-run3/output/1118/pTtemplate_fitLo1p5_win2p0-3p3_Expo3_coh0p1"
templateClass = {"All": "all", "0n0n": "0n0n", "Xn0n": "Xn0n", "0nXn": "0nXn", "XnXn": "XnXn"}


def readB(name):
  """(b, 誤差) をテンプレートフィットの fitParams.txt から読む。"""
  import glob
  for line in open(glob.glob(f"{templateDir}/{templateClass[name]}/*_fitParams.txt")[0]):
    s = line.split()
    if s and s[0] == "b_nd":
      return float(s[1]), float(s[2])
  raise RuntimeError(f"b_nd not found for {name}")


def h1Shape(b):
  """H1 の形をビンごとに積分し, [0, ptMax] で 1 に規格化。"""
  x = np.linspace(0.0, ptMax, 6001)
  f = x * (1.0 + b / nND * x * x) ** (-nND)
  c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])
  edgesInt = np.interp(bins, x, c)
  return np.diff(edgesInt) / c[-1]


def fitClass(name):
  a = ROOT.RDataFrame("gen", f"{unfoldBase}/{name}/unfolded.root").AsNumpy(["pt", "component", "w_prior", "w_unfolded"])
  inRange = a["pt"] < ptMax
  h = np.histogram(a["pt"][inRange], bins, weights=a["w_unfolded"][inRange])[0]
  err = np.sqrt(np.histogram(a["pt"][inRange], bins, weights=a["w_unfolded"][inRange] ** 2)[0])
  err = np.maximum(err, 1.0)
  tmpl, frac = {}, {}
  for comp, k in enumerate(("coh", "incoh")):
    sel = a["component"] == comp
    t = np.histogram(a["pt"][sel & inRange], bins, weights=a["w_prior"][sel & inRange])[0]
    tmpl[k] = t / t.sum()
    frac[k] = a["w_prior"][sel & inRange].sum() / a["w_prior"][sel].sum()   # 形の gen pT < ptMax の割合

  b, bErr = readB(name)
  b += bShift * bErr          # 系統誤差用: --bShift +-1 で b を +-1 sigma 動かす
  hND = h1Shape(b)

  def model(p):
    return p[0] * tmpl["coh"] + p[1] * tmpl["incoh"] + p[2] * hND

  p0 = [0.5 * h.sum(), 0.3 * h.sum(), 0.2 * h.sum()]
  res = least_squares(lambda p: (model(p) - h) / err, p0, bounds=(0, np.inf))
  cov = np.linalg.pinv(res.jac.T @ res.jac)
  perr = np.sqrt(np.diag(cov))
  chi2ndf = 2.0 * res.cost / (len(h) - len(p0))

  # reco の収量に直す
  nPrior = {}
  for line in open(f"{fdBase}/{name}/summary.txt"):
    s = line.split()
    if s and s[0] in ("coh", "incoh"):
      nPrior[s[0]] = float(s[1])
  out = dict(chi2ndf=chi2ndf, b=b, bErr=bErr, N_ND=res.x[2], N_NDErr=perr[2])
  for i, k in enumerate(("coh", "incoh")):
    eps = nPrior[k] / a["w_prior"][a["component"] == (0 if k == "coh" else 1)].sum()
    nFull = res.x[i] / frac[k]
    out.update({f"N_{k}": res.x[i], f"N_{k}Err": perr[i], f"N_{k}_full": nFull, f"eps_{k}": eps, f"N_{k}_reco": eps * nFull})

  # 図 (OmniFold の例と同じ形)
  fig, ax = plt.subplots(figsize=(6.4, 4.8))
  ax.hist(bins[:-1], bins=bins, weights=h, color="orange", alpha=0.5, label="OmniFolded")
  for w, lab, col in ((res.x[0] * tmpl["coh"], "coherent", "red"), (res.x[1] * tmpl["incoh"], "incoherent", "blue"),
                      (res.x[2] * hND, "dissociative (H1)", "green"), (model(res.x), "total", "black")):
    ax.hist(bins[:-1], bins=bins, weights=w, histtype="step", color=col, lw=2 if lab == "total" else 1.5, label=lab)
  ax.set_yscale("log")
  ax.set_xlim(bins[0], bins[-1])
  ax.set_xlabel(r"$p_\mathrm{T}^\mathrm{gen}$ (GeV/$c$)")
  ax.set_ylabel("events")
  ax.set_title(f"{name}, $\\chi^2$/ndf = {chi2ndf:.2f}")
  ax.legend(frameon=False)
  fig.tight_layout()
  os.makedirs(f"{outBase}/{name}", exist_ok=True)
  fig.savefig(f"{outBase}/{name}/truthFit.png")
  plt.close(fig)
  with open(f"{outBase}/{name}/fitResults.txt", "w") as fh:
    fh.write(" ".join(out) + "\n" + " ".join(f"{v:.5g}" for v in out.values()) + "\n")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--classes", nargs="+", choices=list(sp.classes), default=list(sp.classes))
  ap.add_argument("--bShift", type=float, default=0.0, help="b を何 sigma 動かすか (系統誤差用)")
  args = ap.parse_args()
  names = args.classes
  global bShift, outBase
  bShift = args.bShift
  if bShift != 0:
    outBase += f"_bShift{bShift:+g}"
  os.makedirs(outBase, exist_ok=True)
  lines = ["# <class> <N_coh^reco> <N_incoh^reco>  (truthFit.py; subtractFD.py --yields に渡す)\n"]
  for name in names:
    r = fitClass(name)
    lines.append(f"{name} {r['N_coh_reco']:.1f} {r['N_incoh_reco']:.1f}\n")
    print(f"[{name}] chi2/ndf = {r['chi2ndf']:.2f}  N_coh = {r['N_coh']:.0f} +- {r['N_cohErr']:.0f}  "
          f"N_incoh = {r['N_incoh']:.0f} +- {r['N_incohErr']:.0f}  N_ND = {r['N_ND']:.0f} +- {r['N_NDErr']:.0f}  b = {r['b']:.3f}  "
          f"-> reco coh/incoh = {r['N_coh_reco']:.0f}/{r['N_incoh_reco']:.0f}")
  with open(f"{outBase}/yields_reco.txt", "w") as fh:
    fh.writelines(lines)


if __name__ == "__main__":
  main()
