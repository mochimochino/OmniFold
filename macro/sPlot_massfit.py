#!/usr/bin/env python3
# ============================================================================
# sPlot_massfit.py
# ============================================================================
# 実データの背景を sPlot で除く。znClass ごと, pT スライスごとに質量を unbinned extended fit し,
# イベントごとの sWeight を出す。
#
# 質量モデル (taskforce/MassFitELoss_Data_TmplSlices.py --mode fix --bg Expo3 --coherentPtMax 0.1 と同じ):
#   J/psi DSCB (m0 自由, sigma 固定, tail は MC) + E loss DSCB (形は MC, mean = MC + shift)
#   + psi(2S) DSCB + 背景 exp(p1 m + p2 m^2)
#   pT 中心 < 0.1 GeV/c: J/psi は sigma 1 つ (自由) + コヒーレント MC の tail, E loss なし
# 信号 sW_sig = sW_jpsi + sW_eloss (psi(2S) -> mu mu は背景側)。
# sWeight は fit 範囲 [fitLo, fitHi] 全体で使う。MC の reco も同じ質量範囲, pT < 3.6, -4 < y < -2.5 にそろえる。
#
# 出力 ../output/sPlot/<class>/: sWeights.root (TTree "sWeights"), sliceFits.txt, sWeighted.png,
#   massFit_page<k>.png (pT スライスごとの質量分布とフィット, 下段は pull),
#   massSWeighted_page<k>.png (同じスライスの sW_sig で重み付けした質量分布とフィットの信号成分)
# 実行: python3 sPlot_massfit.py [--classes 0nXn XnXn]   (alienv の O2Physics 環境で)
# ============================================================================
import argparse
import os

import numpy as np
import ROOT

# ---- Settings ----
dataFile = "/media/takuma/ESD-EAWA/Data/UPCcandMuon/StandaloneMuon/Unbinned/pass4/incoherent/testZDC/AnalysisResults.root"
treePath = "my-upc-mass-global-muon/unbinnedFitPairTuple"
classes = {"All": -1, "0n0n": 0, "Xn0n": 1, "0nXn": 2, "XnXn": 3}   # znClass (-1: カットなし)
mcDir = "/media/takuma/ESD-EAWA/Data/UPCcandMuon/MC/Standalone/IncoherentAnalysis/unbined"
jpsiMcFile = f"{mcDir}/mass/asymmetric/MassFit_RooFit_Unbinned_Incoherent_JpsiStandalone/FitParameters.txt"
psi2sMcFile = f"{mcDir}/incoherentpsi2s/mass/asymmetric/MassFit_RooFit_Unbinned_Incoherent_Psi2SStandalone/FitParameters.txt"
taskforceOut = "/home/takuma/alice/upc-analysis-run3/output/1118"
elossShapeFile = f"{taskforceOut}/BumpShape_TmplSlices/bumpShape_MC_DSCB.txt"
coherentShapeFile = f"{taskforceOut}/CoherentPeak/coherentPeak_MC_DSCB.txt"
elossShift = -0.0304             # [GeV/c^2] E loss mean - MC (0n0n free fit, 0.3-0.8 GeV/c; DataFitTmpl_fitLo1p5_win2p0-3p3_Expo3)

ptSegments = [(0.0, 0.3, 0.020), (0.3, 1.2, 0.050)]   # [GeV/c] (lo, hi, width)
ptEdges = [0.0] + [round(lo + (i + 1) * w, 6) for lo, hi, w in ptSegments for i in range(round((hi - lo) / w))]
yLo, yHi = -4.0, -2.5
fitLo, fitHi = 1.5, 4.5          # [GeV/c^2]
coherentPtMax = 0.1              # [GeV/c]
elossPtMin = 0.14                # [GeV/c] E loss 成分を入れるスライス中心の下限 (これより下は MC の E loss の位置が
                                 #   J/psi ピークと区別できず, 成分どうしの相関で sWeight が暴れるため)
elossPtMax = 1.0                 # [GeV/c] E loss 成分を入れるスライス中心の上限
sigmaLJpsi, sigmaRJpsi = 0.09656, 0.07424   # [GeV/c^2] 固定
sigmaFactorPsi2S = 1.1
mPsi2SminusJpsi = 3.6861 - 3.0969
minEntries = 10

outBase = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "sPlot")


def readTable(path):
  with open(path) as fh:
    header = fh.readline().split()
    return [dict(zip(header, map(float, line.split()))) for line in fh if line.strip()]


def findRow(table, centre):
  return next((r for r in table if r["ptLo"] <= centre < r["ptHi"]), None)


def mergeSparse(pt):
  """minEntries 未満のスライスを次と結合 (最後の余りは直前のスライスへ)。"""
  counts = np.histogram(pt, bins=ptEdges)[0]
  edges, acc = [ptEdges[0]], 0
  for c, hi in zip(counts, ptEdges[1:]):
    acc += c
    if acc >= minEntries:
      edges.append(hi)
      acc = 0
  edges[-1] = ptEdges[-1]
  return edges


def loadData(znClass):
  mass = ROOT.RooRealVar("mass", "m_{#mu#mu}", fitLo, fitHi)
  pt = ROOT.RooRealVar("pairPt", "p_{T}", 0.0, ptEdges[-1])
  rap = ROOT.RooRealVar("rapidity", "y", yLo, yHi)
  zn = ROOT.RooRealVar("znClass", "znClass", -2.0, 4.0)
  f = ROOT.TFile.Open(dataFile)
  ds = ROOT.RooDataSet("dsAll", "", ROOT.RooArgSet(mass, pt, rap, zn), ROOT.RooFit.Import(f.Get(treePath)),
                       ROOT.RooFit.Cut("znClass>-10" if znClass < 0 else f"znClass=={znClass}"))
  f.Close()
  return mass, ds


def buildModel(mass, n, centre, mc, keep):
  """スライスの質量モデル。戻り値: (model, 収量 {成分: RooRealVar}, 収量以外の自由パラメータ)。"""
  def var(name, *args):
    v = ROOT.RooRealVar(name, name, *args)
    keep.append(v)
    return v

  def dscb(name, m0, sL, sR, p):
    pdf = ROOT.RooCrystalBall(name, name, mass, m0, sL, sR, *[var(f"{name}_{k}", p[k]) for k in ("alphaL", "nL", "alphaR", "nR")])
    keep.append(pdf)
    return pdf

  m0 = var("m0", 3.097, 2.9, 3.2)
  free = [m0]
  coh = findRow(mc["coh"], centre) if centre < coherentPtMax else None
  if coh:   # コヒーレント主体: sigma 1 つ (自由), tail はコヒーレント MC, E loss なし
    sL = sR = var("sJpsi", 0.080, 0.04, 0.15)
    free.append(sL)
  else:
    sL, sR = var("sLJpsi", sigmaLJpsi), var("sRJpsi", sigmaRJpsi)
  pdfs = {"jpsi": dscb("jpsi", m0, sL, sR, coh or mc["jpsi"])}
  yields = {"jpsi": var("nJpsi", 0.3 * n, 0.0, 2.0 * n)}

  el = findRow(mc["eloss"], centre) if not coh and elossPtMin <= centre < elossPtMax else None
  if el:
    p = [var(f"el_{k}", el[k]) for k in ("sigmaL", "sigmaR", "alphaL", "nTail", "alphaR")]
    pdfs["eloss"] = ROOT.RooCrystalBall("eloss", "eloss", mass, var("elMean", el["mean"] + elossShift), *p, p[3])
    yields["eloss"] = var("nELoss", 0.1 * n, 0.0, n)

  r = sigmaFactorPsi2S / mc["jpsi"]["sigmaL"], sigmaFactorPsi2S / mc["jpsi"]["sigmaR"]
  m0P = ROOT.RooFormulaVar("m0P", f"@0+{mPsi2SminusJpsi}", ROOT.RooArgList(m0))
  sLP = ROOT.RooFormulaVar("sLP", f"@0*{r[0] * mc['psi2s']['sigmaL']}", ROOT.RooArgList(sL))
  sRP = ROOT.RooFormulaVar("sRP", f"@0*{r[1] * mc['psi2s']['sigmaR']}", ROOT.RooArgList(sR))
  keep += [m0P, sLP, sRP]
  pdfs["psi2s"] = dscb("psi2s", m0P, sLP, sRP, mc["psi2s"])
  yields["psi2s"] = var("nPsi2S", 0.02 * n, 0.0, 0.5 * n)

  bgP = [var("bgP1", -2.15838, -20.0, 20.0), var("bgP2", 0.136308, -20.0, 20.0)]
  pdfs["bg"] = ROOT.RooGenericPdf("bg", "exp(TMath::Max(-700.0, TMath::Min(700.0, @1*@0 + @2*@0*@0)))", ROOT.RooArgList(mass, *bgP))
  yields["bg"] = var("nBg", 0.5 * n, 0.0, 2.0 * n)
  free += bgP

  keep += list(pdfs.values())
  model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(*pdfs.values()), ROOT.RooArgList(*yields.values()))
  keep.append(model)
  return model, pdfs, yields, free


def fitSlice(dsAll, mass, lo, hi, mc):
  """1 スライスを fit して sWeight を付ける。戻り値: (fit 結果の行, イベントごとの列, 図用の曲線)。"""
  keep = []
  ds = dsAll.reduce(ROOT.RooFit.Cut(f"pairPt>={lo}&&pairPt<{hi}"))
  n = ds.numEntries()
  model, pdfs, yields, free = buildModel(mass, n, 0.5 * (lo + hi), mc, keep)
  opts = [ROOT.RooFit.Extended(True), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.Save(True), ROOT.RooFit.Strategy(2)]
  res = model.fitTo(ds, *opts)
  if res.status() != 0:   # 収束しなければ結果から 1 回だけやり直す
    res = model.fitTo(ds, *opts)

  # 図用: 各成分の密度 (事象数 / (GeV/c^2)) を細かい格子で計算
  grid = np.linspace(fitLo, fitHi, 601)
  norm = ROOT.RooArgSet(mass)
  curves = {}
  for s, pdf in pdfs.items():
    vals = []
    for x in grid:
      mass.setVal(x)
      vals.append(pdf.getVal(norm))
    curves[s] = np.array(vals) * yields[s].getVal()
  plot = dict(lo=lo, hi=hi, status=res.status(), grid=grid, curves=curves)

  for p in free:          # sPlot: 収量以外は固定
    p.setConstant(True)
  ROOT.RooStats.SPlot("sp", "", ds, model, ROOT.RooArgList(*yields.values()))
  arr = ds.to_numpy()
  out = {k: np.asarray(arr[k], dtype=np.float64) for k in ("mass", "pairPt", "rapidity", "znClass")}
  for s in ("jpsi", "eloss", "psi2s", "bg"):
    out[f"sW_{s}"] = np.asarray(arr[f"{yields[s].GetName()}_sw"], dtype=np.float64) if s in yields else np.zeros(n)
  out["sW_sig"] = out["sW_jpsi"] + out["sW_eloss"]

  row = dict(ptLo=lo, ptHi=hi, entries=n, status=res.status(), covQual=res.covQual(),
             **{f"n_{s}": (yields[s].getVal() if s in yields else 0.0) for s in ("jpsi", "eloss", "psi2s", "bg")})
  plot.update(mass=out["mass"], sw=out["sW_sig"], sig=out["sW_sig"].sum())
  return row, out, plot


def drawSliceFits(prefix, name, plots, ncol=3, nrow=4, width=0.05, weighted=False):
  """pT スライスごとの質量分布とフィット (従来の Datafit_page と同じ確認)。下段は (Data - Fit) / sigma。
  weighted=True: sW_sig で重み付けした (背景を引いた) 質量分布と, フィットの信号成分 (J/psi + E loss) を比べる。"""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  bins = np.arange(fitLo, fitHi + 1e-9, width)
  styles = {"bg": ("green", "--", "Background"), "jpsi": ("magenta", "-", r"J/$\psi$"),
            "eloss": ("red", "-", "E loss"), "psi2s": ("darkorange", "-", r"$\psi$(2S)")}
  if weighted:
    styles = {s: styles[s] for s in ("jpsi", "eloss")}
  per = ncol * nrow
  for page in range((len(plots) + per - 1) // per):
    fig = plt.figure(figsize=(5 * ncol, 4.4 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.3, wspace=0.3)
    for j in range(per):
      inner = outer[j // ncol, j % ncol].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
      axT, axB = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
      if page * per + j >= len(plots):
        axT.axis("off")
        axB.axis("off")
        continue
      p = plots[page * per + j]
      c = 0.5 * (bins[1:] + bins[:-1])
      if weighted:
        k = np.histogram(p["mass"], bins, weights=p["sw"])[0]
        err = np.sqrt(np.histogram(p["mass"], bins, weights=p["sw"] ** 2)[0])
      else:
        k = np.histogram(p["mass"], bins)[0]
        err = np.sqrt(np.maximum(k, 1))
      curves = {s: y for s, y in p["curves"].items() if s in styles}
      total = sum(curves.values())
      if weighted:
        # sWeight を付けた質量分布の期待値は信号の形ではなく w_sig(m) x (全成分の密度)。
        # w_sig(m) はスライス内で質量だけの関数なので, 事象の (m, w) から格子に補間する
        order = np.argsort(p["mass"])
        wGrid = np.interp(p["grid"], p["mass"][order], p["sw"][order])
        total = wGrid * sum(p["curves"].values())
      # ビンごとの期待値 = 密度をビン内で積分
      cum = np.concatenate([[0.0], np.cumsum(0.5 * (total[1:] + total[:-1]) * np.diff(p["grid"]))])
      mu = np.diff(np.interp(bins, p["grid"], cum))
      axT.errorbar(c, k, yerr=err, fmt="o", ms=2.5, color="black")
      for s, y in curves.items():
        col, ls, lab = styles[s]
        axT.plot(p["grid"], y * width, color=col, ls=":" if weighted else ls, lw=1.2)
      axT.plot(p["grid"], total * width, color="blue", lw=1.5)
      axT.set_xlim(fitLo, fitHi)
      if weighted:
        axT.axhline(0, color="gray", lw=0.8, ls="--")
        axT.set_ylim(bottom=min(0.0, 1.1 * (k - err).min()))
      else:
        axT.set_ylim(bottom=0)
      axT.set_xticklabels([])
      axT.set_ylabel(f"events / {1000 * width:.0f} MeV/$c^2$")
      flag = "" if p["status"] == 0 else f"  status={p['status']}"
      axT.set_title(f"{name}, {p['lo']:.2f} < $p_\\mathrm{{T}}$ < {p['hi']:.2f} GeV/$c${flag}", fontsize=10)
      axT.text(0.03, 0.85, f"N(J/$\\psi$+E loss) = {p['sig']:.0f}", transform=axT.transAxes, fontsize=9)
      if j == 0:   # 凡例は全成分を出す (最初のスライスに無い成分も)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=col, ls=":" if weighted else ls) for col, ls, _ in styles.values()] + \
                  [Line2D([], [], color="blue"), Line2D([], [], color="black", marker="o", ls="", ms=3)]
        labels = ([f"{lab} (fit shape)" for _, _, lab in styles.values()] + ["Expected: $w_{sig}(m)\\times$total", "Data, $sW_{sig}$"]
                  if weighted else [lab for _, _, lab in styles.values()] + ["Total", "Data"])
        axT.legend(handles, labels, frameon=False, fontsize=8, loc="upper right")
      axB.axhline(0, color="red", ls="--", lw=0.8)
      axB.plot(c, (k - mu) / (err if weighted else np.sqrt(np.maximum(mu, 1))), "o", ms=2.5, color="black")
      axB.set_ylim(-4, 4)
      axB.set_xlim(fitLo, fitHi)
      axB.set_xlabel(r"$m_{\mu\mu}$ (GeV/$c^2$)")
      axB.set_ylabel("pull")
    fig.savefig(f"{prefix}_page{page + 1}.png", bbox_inches="tight")
    plt.close(fig)


def drawSWeighted(path, name, data):
  """sWeight を掛けた m, pT, y 分布 (信号と背景)。OmniFold の例と同じ形 (ビンは表示用だけ)。"""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  defs = [("mass", 50, fitLo, fitHi, r"$m_{\mu\mu}$ (GeV/$c^2$)", False),
          ("pairPt", 40, 0.0, ptEdges[-1], r"$p_\mathrm{T}$ (GeV/$c$)", True),
          ("rapidity", 30, yLo, yHi, r"$y$", False)]
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
  for ax, (var, nb, lo, hi, title, logY) in zip(axes, defs):
    bins = np.linspace(lo, hi, nb + 1)
    ax.hist(data[var], bins=bins, weights=data["sW_sig"], color="orange", alpha=0.5, label=r"J/$\psi$ + E loss")
    ax.hist(data[var], bins=bins, weights=data["sW_bg"], histtype="step", color="green", lw=2, label="Background")
    if logY:
      ax.set_yscale("log")
      ax.set_ylim(bottom=1.0)
    else:
      ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(title)
    ax.set_ylabel("events (sWeighted)")
    ax.set_title(name)
    ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)


def main():
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat(0)
  ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
  ap = argparse.ArgumentParser()
  ap.add_argument("--classes", nargs="+", choices=list(classes), default=list(classes))
  mc = {"jpsi": readTable(jpsiMcFile)[0], "psi2s": readTable(psi2sMcFile)[0],
        "eloss": readTable(elossShapeFile), "coh": readTable(coherentShapeFile)}

  for name in ap.parse_args().classes:
    outDir = os.path.join(outBase, name)
    os.makedirs(outDir, exist_ok=True)
    mass, dsAll = loadData(classes[name])
    edges = mergeSparse(dsAll.to_numpy()["pairPt"])
    rows, parts, plots = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
      row, out, plot = fitSlice(dsAll, mass, lo, hi, mc)
      rows.append(row)
      parts.append(out)
      plots.append(plot)
      flag = "" if row["status"] == 0 and row["covQual"] == 3 else "  <-- check fit"
      print(f"[{name}] pT {lo:.2f}-{hi:.2f}  N={row['entries']}  sig={out['sW_sig'].sum():.0f}  status={row['status']}{flag}")

    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ROOT.RDF.FromNumpy({k: np.ascontiguousarray(v) for k, v in data.items()}).Snapshot("sWeights", f"{outDir}/sWeights.root")
    with open(f"{outDir}/sliceFits.txt", "w") as fh:
      fh.write(" ".join(rows[0]) + "\n")
      fh.writelines(" ".join(f"{v:.5f}" for v in r.values()) + "\n" for r in rows)
    drawSWeighted(f"{outDir}/sWeighted.png", name, data)
    drawSliceFits(f"{outDir}/massFit", name, plots)
    drawSliceFits(f"{outDir}/massSWeighted", name, plots, weighted=True)
    nBad = sum(r["status"] != 0 or r["covQual"] != 3 for r in rows)
    print(f"[{name}] events {len(data['mass'])}, sum sW_sig = {data['sW_sig'].sum():.0f}, fits to check {nBad}/{len(rows)} -> {outDir}")


if __name__ == "__main__":
  main()
