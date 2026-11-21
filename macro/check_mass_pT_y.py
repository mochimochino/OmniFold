#!/usr/bin/env python3
# ============================================================================
# check_mass_pT_y.py
# ============================================================================
# 実データと MC (reco) の質量 / pT / ラピディティ分布を 1 次元で重ねて比較する。
# 入力は ../data.md の通り:
#   Data : my-upc-mass-global-muon/unbinnedFitPairTuple  (mass, pairPt, rapidity)
#   MC   : upc-muon-analysis/omniFoldPairTuple (passRecoFlag のみ)
#        + upc-muon-analysis/omniFoldRecoOnlyTuple      (truth がアクセプタンス外の reco)
#          -> pairMassReco, sqrt(pairPt2Reco), pairRapReco
# 各分布は面積 1 に規格化して形だけを比べる。
#   質量分布 : rapidity 窓 + pT 窓
#   pT 分布  : rapidity 窓 + 質量窓
#   y 分布   : pT 窓 + 質量窓
#
# 実行:
#   export ALIBUILD_WORK_DIR=/home/takuma/alice/sw && eval "$(alienv load O2Physics/latest-o2 -q)"
#   python3 check_mass_pT_y.py
# ============================================================================

import os

import ROOT

ROOT.gROOT.SetBatch(True)

# ============================================================================
# Settings
# ============================================================================
dataFile = "/media/takuma/ESD-EAWA/Data/UPCcandMuon/StandaloneMuon/Unbinned/pass4/incoherent/testZDC/AnalysisResults.root"
dataTree = "my-upc-mass-global-muon/unbinnedFitPairTuple"

mcBaseDir = "/media/takuma/ESD-EAWA/Data/UPCcandMuon/MC/Standalone/Omnifold"
mcPairTree = "upc-muon-analysis/omniFoldPairTuple"
mcRecoOnlyTree = "upc-muon-analysis/omniFoldRecoOnlyTuple"
mcComponents = [  # (tag, file, label, color)
    ("jpsi-coh",       "jpsi-coh.root",       "MC: J/#psi coherent",          ROOT.kRed + 1),
    ("jpsi-incoh",     "jpsi-incoh.root",     "MC: J/#psi incoherent",        ROOT.kBlue + 1),
    ("psi2s-coh-fd",   "psi2s-coh-fd.root",   "MC: #psi(2S) coherent (fd)",   ROOT.kOrange + 7),
    ("psi2s-incoh-fd", "psi2s-incoh-fd.root", "MC: #psi(2S) incoherent (fd)", ROOT.kGreen + 2),
]

# znClass ごとに outputDir/<名前>/ へ出力 (データのみに適用, MC は共通)
# -1 = カットなし, 0=0n0n, 1=Xn0n, 2=0nXn, 3=XnXn
topologies = {-1: "All", 0: "0n0n", 1: "Xn0n", 2: "0nXn", 3: "XnXn"}

# 選択窓
yLo, yHi = -4.0, -2.5          # rapidity
ptLo, ptHi = 0.0, 3.0          # [GeV/c]
massLo, massHi = 2.0, 4.5      # [GeV/c^2] J/psi 窓

# ヒストグラム (変数, nbins, min, max, x軸タイトル, logY)
histDefs = {
    "mass": (100, 2.0, 4.5, "#it{m}_{#mu#mu} (GeV/#it{c}^{2})", False),
    "pt":   (60, 0.0, 3.0, "#it{p}_{T} (GeV/#it{c})", True),
    "y":    (30, -4.0, -2.5, "#it{y}", False),
}

outputDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./../output", "check_mass_pT_y")
fontType = 42


# ============================================================================
# Helpers
# ============================================================================
def applyStyle():
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    for ax in ("X", "Y"):
        ROOT.gStyle.SetLabelFont(fontType, ax)
        ROOT.gStyle.SetTitleFont(fontType, ax)
        ROOT.gStyle.SetTitleSize(0.045, ax)
        ROOT.gStyle.SetLabelSize(0.040, ax)


def selections():
    """変数ごとの選択式 (その変数自身の窓は外す)。列名は m, pt, y に統一済み。"""
    cy = f"y > {yLo} && y < {yHi}"
    cpt = f"pt > {ptLo} && pt < {ptHi}"
    cm = f"m > {massLo} && m < {massHi}"
    return {"mass": f"{cy} && {cpt}", "pt": f"{cy} && {cm}", "y": f"{cpt} && {cm}"}


def bookHists(df, tag):
    hs = {}
    for var, sel in selections().items():
        nb, lo, hi, _, _ = histDefs[var]
        col = "m" if var == "mass" else var
        hs[var] = df.Filter(sel).Histo1D((f"h_{var}_{tag}", "", nb, lo, hi), col)
    return hs


def dataFrame():
    df = ROOT.RDataFrame(dataTree, dataFile)
    return df.Define("m", "mass").Define("pt", "pairPt").Define("y", "rapidity")


def topoFrame(df, topo):
    return df if topo < 0 else df.Filter(f"znClass == {topo}")


def mcFrames(path):
    """reco 側: PairTuple の passRecoFlag 行 + RecoOnlyTuple 全行。"""
    out = []
    for tree, cut in ((mcPairTree, "passRecoFlag && pairPt2Reco >= 0"), (mcRecoOnlyTree, "pairPt2Reco >= 0")):
        df = ROOT.RDataFrame(tree, path).Filter(cut)
        out.append(df.Define("m", "pairMassReco").Define("pt", "sqrt(pairPt2Reco)").Define("y", "pairRapReco"))
    return out


mplTitles = {"mass": r"$m_{\mu\mu}$ (GeV/$c^2$)", "pt": r"$p_\mathrm{T}$ (GeV/$c$)", "y": r"$y$"}
mplColors = {ROOT.kRed + 1: "red", ROOT.kBlue + 1: "blue", ROOT.kOrange + 7: "darkorange", ROOT.kGreen + 2: "green"}


def draw(var, hData, mcHists, topo, outDir):
    """OmniFold の例 (GaussianExample) と同じ形: データは塗りつぶし, MC は step。面積 1 に規格化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _, _, _, _, logY = histDefs[var]

    def contents(h):
        n = h.GetNbinsX()
        edges = np.array([h.GetBinLowEdge(i) for i in range(1, n + 2)])
        y = np.array([h.GetBinContent(i) for i in range(1, n + 1)])
        return edges, y / y.sum() if y.sum() > 0 else y

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    edges, y = contents(hData)
    ax.hist(edges[:-1], bins=edges, weights=y, color="orange", alpha=0.5,
            label="Data" + (f" ({topologies[topo]})" if topo >= 0 else ""))
    for h, label, color in mcHists:
        edges, y = contents(h)
        ax.hist(edges[:-1], bins=edges, weights=y, histtype="step", lw=1.5, color=mplColors.get(color, "black"),
                label=label.replace("#psi", r"$\psi$"))
    if logY:
        ax.set_yscale("log")
    sel = {"mass": rf"${yLo} < y < {yHi}$, ${ptLo} < p_\mathrm{{T}} < {ptHi}$ GeV/$c$",
           "pt": rf"${yLo} < y < {yHi}$, ${massLo} < m < {massHi}$ GeV/$c^2$",
           "y": rf"${ptLo} < p_\mathrm{{T}} < {ptHi}$ GeV/$c$, ${massLo} < m < {massHi}$ GeV/$c^2$"}[var]
    ax.set_xlim(edges[0], edges[-1])
    ax.set_title(sel, fontsize=10)
    ax.set_xlabel(mplTitles[var])
    ax.set_ylabel("normalized events")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outDir, f"{var}.{ext}"))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    applyStyle()

    # 全 znClass 分を先に予約し、データの TTree は 1 回だけ読む
    dfData = dataFrame()
    dataHists = {topo: bookHists(topoFrame(dfData, topo), f"data_{name}") for topo, name in topologies.items()}
    mcBooked = []
    for tag, fname, label, color in mcComponents:
        parts = [bookHists(df, f"{tag}_{i}") for i, df in enumerate(mcFrames(os.path.join(mcBaseDir, fname)))]
        mcBooked.append((parts, label, color))

    for topo, name in topologies.items():
        outDir = os.path.join(outputDir, name)
        os.makedirs(outDir, exist_ok=True)
        for var in histDefs:
            hData = dataHists[topo][var].GetValue()
            mcHists = []
            for parts, label, color in mcBooked:
                h = parts[0][var].GetValue().Clone(f"h_{var}_{label}_{name}")
                h.Add(parts[1][var].GetValue())
                mcHists.append((h, label, color))
            print(f"[{name} {var}] Data {hData.GetEntries():.0f}  " +
                  "  ".join(f"{l.split(': ')[1]} {h.GetEntries():.0f}" for h, l, _ in mcHists))
            draw(var, hData, mcHists, topo, outDir)


if __name__ == "__main__":
    main()
