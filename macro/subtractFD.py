#!/usr/bin/env python3
# ============================================================================
# subtractFD.py
# ============================================================================
# 手順 2: FD (psi(2S) -> J/psi + X) の差し引き。
#   sPlot のデータ事象 (重み sW_sig; 直接生成 + FD + 解離を含む) に, FD の MC の再構成事象を負の重みで加える。
#   重み = -N_FD^reco / N_MC^reco  (coherent と incoherent それぞれ)
#   N_FD^reco = fdRatio x N_J/psi^reco  (fdRatio = 4.9%, 解離は含めない; reco の収量どうしの比)
#
# N_J/psi^reco (coh, incoh):
#   --yields なし (初回): 1118 の pT テンプレートフィット (massfitELoss) の割合 N_x / N_total を
#                         sum(sW_sig) に掛けたもの
#   --yields <file>    : 2 回目以降。手順 4 の結果から作ったファイル (行: <class> <N_coh^reco> <N_incoh^reco>)
# MC の選択は sPlot と同じ (reco 質量 [fitLo, fitHi], pT < ptEdges[-1], yLo < y < yHi)。
#
# 出力 ../output/fdSubtracted/<class>/:
#   dataForOmniFold.root  TTree "data": mass, pairPt, rapidity, weight, source (0: data sW_sig, 1: FD coh MC, 2: FD incoh MC)
#   fdSubtracted.png      pT 分布: sW_sig, FD (coh, incoh), 差し引き後
#   summary.txt           使った N_J/psi^reco, N_FD^reco, N_MC^reco, 重み
# 実行: python3 subtractFD.py [--classes ...] [--yields file]   (alienv の O2Physics 環境で)
# ============================================================================
import argparse
import glob
import os
import re

import numpy as np
import ROOT

import sPlot_massfit as sp   # 選択条件 (fitLo, fitHi, ptEdges, yLo, yHi) と sPlot の出力先を共有

fdRatio = 0.049
fdMcDir = "/media/takuma/ESD-EAWA/Data/UPCcandMuon/MC/Standalone/Omnifold"
fdMcFiles = {"coh": "psi2s-coh-fd.root", "incoh": "psi2s-incoh-fd.root"}
templateDir = "/home/takuma/alice/upc-analysis-run3/output/1118/pTtemplate_fitLo1p5_win2p0-3p3_Expo3_coh0p1"
templateClass = {"All": "all", "0n0n": "0n0n", "Xn0n": "Xn0n", "0nXn": "0nXn", "XnXn": "XnXn"}
outBase = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "fdSubtracted")


def templateFractions(name):
  """1118 テンプレートフィットの N_coh / N_total, N_incoh / N_total (sPlot と同じ pT 範囲のビンの合計)。"""
  path = glob.glob(f"{templateDir}/{templateClass[name]}/*_bin_yields.txt")[0]
  s = np.zeros(11)
  for line in open(path).readlines()[1:]:
    if float(line.split()[1]) <= sp.ptEdges[-1] + 1e-6:
      s += [float(x) for x in re.findall(r"([-\d.]+)\+/-", line)]   # coh, frac, incoh, frac, fdc, frac, fdi, frac, nd, frac, total
  return {"coh": s[0] / s[10], "incoh": s[2] / s[10]}


def readYields(path):
  """{class: {"coh": N, "incoh": N}} (reco の収量)。"""
  out = {}
  for line in open(path):
    if line.strip() and not line.startswith("#"):
      c, nc, ni = line.split()
      out[c] = {"coh": float(nc), "incoh": float(ni)}
  return out


def fdMcReco(path):
  """FD MC の再構成事象 (PairTuple の passRecoFlag 行 + RecoOnlyTuple), sPlot と同じ選択。"""
  sel = (f"pairMassReco > {sp.fitLo} && pairMassReco < {sp.fitHi} && pairPt2Reco >= 0 && sqrt(pairPt2Reco) < {sp.ptEdges[-1]}"
         f" && pairRapReco > {sp.yLo} && pairRapReco < {sp.yHi}")
  cols = {"mass": [], "pairPt": [], "rapidity": []}
  for tree, cut in (("upc-muon-analysis/omniFoldPairTuple", "passRecoFlag"), ("upc-muon-analysis/omniFoldRecoOnlyTuple", "true")):
    a = (ROOT.RDataFrame(tree, path).Filter(cut).Filter(sel)
         .Define("pt", "sqrt(pairPt2Reco)").AsNumpy(["pairMassReco", "pt", "pairRapReco"]))
    cols["mass"].append(a["pairMassReco"])
    cols["pairPt"].append(a["pt"])
    cols["rapidity"].append(a["pairRapReco"])
  return {k: np.concatenate(v).astype(np.float64) for k, v in cols.items()}


def draw(path, name, data, fd, w):
  """OmniFold の例 (GaussianExample) と同じ形の pT 分布。ビンは表示用だけ (重みは事象ごと)。"""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  bins = np.linspace(sp.ptEdges[0], sp.ptEdges[-1], 41)
  xs = [data["pairPt"], fd["coh"]["pairPt"], fd["incoh"]["pairPt"]]
  ws = [data["sW_sig"], np.full(len(xs[1]), w["coh"]), np.full(len(xs[2]), w["incoh"])]
  fig, ax = plt.subplots(figsize=(6.4, 4.8))
  ax.hist(xs[0], bins=bins, weights=ws[0], color="orange", alpha=0.5, label=r"Data, $sW_{sig}$")
  ax.hist(xs[1], bins=bins, weights=-ws[1], histtype="step", color="red", ls=":", label="FD coherent (MC)")
  ax.hist(xs[2], bins=bins, weights=-ws[2], histtype="step", color="blue", ls=":", label="FD incoherent (MC)")
  ax.hist(np.concatenate(xs), bins=bins, weights=np.concatenate(ws), histtype="step", color="black", lw=2, label="Data $-$ FD")
  ax.set_yscale("log")
  ax.set_ylim(bottom=1.0)
  ax.set_xlim(bins[0], bins[-1])
  ax.set_xlabel(r"$p_\mathrm{T}$ (GeV/$c$)")
  ax.set_ylabel("events")
  ax.set_title(name)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)


def main():
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat(0)
  ap = argparse.ArgumentParser()
  ap.add_argument("--classes", nargs="+", choices=list(sp.classes), default=list(sp.classes))
  ap.add_argument("--yields", help="N_J/psi^reco (coh, incoh) のファイル (2 回目以降)")
  args = ap.parse_args()
  given = readYields(args.yields) if args.yields else None
  fd = {k: fdMcReco(f"{fdMcDir}/{f}") for k, f in fdMcFiles.items()}

  for name in args.classes:
    data = ROOT.RDataFrame("sWeights", f"{sp.outBase}/{name}/sWeights.root").AsNumpy(["mass", "pairPt", "rapidity", "sW_sig"])
    sumSig = data["sW_sig"].sum()
    if given:
      nJpsi = given[name]
    else:
      nJpsi = {k: f * sumSig for k, f in templateFractions(name).items()}
    nFD = {k: fdRatio * nJpsi[k] for k in fd}
    w = {k: -nFD[k] / len(fd[k]["pairPt"]) for k in fd}

    out = {"mass": [data["mass"]], "pairPt": [data["pairPt"]], "rapidity": [data["rapidity"]],
           "weight": [data["sW_sig"]], "source": [np.zeros(len(data["mass"]), dtype=np.int32)]}
    for src, k in ((1, "coh"), (2, "incoh")):
      n = len(fd[k]["pairPt"])
      for col in ("mass", "pairPt", "rapidity"):
        out[col].append(fd[k][col])
      out["weight"].append(np.full(n, w[k]))
      out["source"].append(np.full(n, src, dtype=np.int32))
    out = {k: np.ascontiguousarray(np.concatenate(v)) for k, v in out.items()}

    outDir = os.path.join(outBase, name)
    os.makedirs(outDir, exist_ok=True)
    ROOT.RDF.FromNumpy(out).Snapshot("data", f"{outDir}/dataForOmniFold.root")
    draw(f"{outDir}/fdSubtracted.png", name, data, fd, w)
    with open(f"{outDir}/summary.txt", "w") as fh:
      fh.write(f"# yields from: {args.yields or 'template fractions x sum(sW_sig)'}\n")
      fh.write("component N_Jpsi_reco N_FD_reco N_MC_reco weight\n")
      for k in fd:
        fh.write(f"{k} {nJpsi[k]:.1f} {nFD[k]:.1f} {len(fd[k]['pairPt'])} {w[k]:.6e}\n")
      fh.write(f"sum_sW_sig {sumSig:.1f}\nsum_weight_after {out['weight'].sum():.1f}\n")
    print(f"[{name}] sum sW_sig = {sumSig:.0f}, N_Jpsi^reco coh/incoh = {nJpsi['coh']:.0f}/{nJpsi['incoh']:.0f}, "
          f"N_FD coh/incoh = {nFD['coh']:.0f}/{nFD['incoh']:.0f} -> after = {out['weight'].sum():.0f}")


if __name__ == "__main__":
  main()
