#!/usr/bin/env python3
# ============================================================================
# omnifold_unfold.py
# ============================================================================
# 手順 3: OmniFold (unbinned)。
#   データ : fdSubtracted/<class>/dataForOmniFold.root (sW_sig と FD MC の負の重み)
#   MC     : 直接生成の J/psi だけ (jpsi-coh + jpsi-incoh)。初期の重みは各成分の reco 数 = N_J/psi^reco (summary.txt)
#   観測量 : (pT, y)。reco の選択は sPlot と同じ。
# 事象の扱い (OmniFold.py の拡張):
#   gen はあるが reco の選択を通らない事象 -> step 1 では push をそのまま引き継ぐ
#   reco だけの事象 (RecoOnlyTuple)       -> step 2 では pull をそのまま引き継ぐ
# 出力 ../output/omnifold/<class>/:
#   unfolded.root  TTree "gen": pt, y, component (0 coh, 1 incoh), w_prior, w_unfolded (gen の選択を通った MC 事象)
#   pt_gen.png, pt_reco.png
# 中心値: 乱数の種を変えて OmniFold を nRuns 回回し, 重み (push) を平均する (学習のばらつきを抑える)
# 実行: python3 omnifold_unfold.py [--classes All] [--iterations 4] [--nRuns 5]   (alienv の O2Physics 環境で)
# ============================================================================
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import tensorflow as tf

import sPlot_massfit as sp   # reco の選択 (fitLo, fitHi, ptEdges, yLo, yHi)

mcDir = "/media/takuma/ESD-EAWA/Data/UPCcandMuon/MC/Standalone/Omnifold"
mcFiles = {"coh": "jpsi-coh.root", "incoh": "jpsi-incoh.root"}
here = os.path.dirname(os.path.abspath(__file__))
dataBase = os.path.join(here, "..", "output", "fdSubtracted")
outBase = os.path.join(here, "..", "output", "omnifold")
epochs, batchSize, patience = 30, 10000, 3


def loadMC():
  """MC の事象ごとの (gen pT, gen y, reco pT, reco y, genOK, recoOK, component)。"""
  recoSel = (f"pairMassReco > {sp.fitLo} && pairMassReco < {sp.fitHi} && pairPt2Reco >= 0 && sqrt(pairPt2Reco) < {sp.ptEdges[-1]}"
             f" && pairRapReco > {sp.yLo} && pairRapReco < {sp.yHi}")
  cols = ["ptG", "yG", "ptR", "yR", "genOK", "recoOK"]
  parts = []
  for comp, (k, f) in enumerate(mcFiles.items()):
    for tree, genOK, hasReco in (("omniFoldPairTuple", True, "passRecoFlag"), ("omniFoldRecoOnlyTuple", False, "true")):
      df = (ROOT.RDataFrame(f"upc-muon-analysis/{tree}", f"{mcDir}/{f}")
            .Define("ptG", "sqrt(pairPt2MC)").Define("yG", "(double)pairRapMC")
            .Define("recoOK", f"{hasReco} && ({recoSel})")
            .Define("ptR", "recoOK ? sqrt(pairPt2Reco) : 0.0").Define("yR", "recoOK ? (double)pairRapReco : 0.0")
            .Define("genOK", "true" if genOK else "false"))
      if not genOK:
        df = df.Filter("recoOK")   # reco だけの事象は reco の選択を通るものだけ
      a = df.AsNumpy(cols)
      a["component"] = np.full(len(a["ptG"]), comp)
      parts.append(a)
  return {c: np.concatenate([p[c] for p in parts]) for c in cols + ["component"]}


def classifier():
  m = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(50, activation="relu"),
                           tf.keras.layers.Dense(50, activation="relu"), tf.keras.layers.Dense(1, activation="sigmoid")])
  m.compile(loss="binary_crossentropy", optimizer="adam", weighted_metrics=[])
  return m


def reweight(x0, w0, x1, w1, xEval):
  """x0 (重み w0, ラベル 0) と x1 (重み w1, ラベル 1) を分ける分類器 f を学習し, xEval で (sum w1 / sum w0) x f/(1-f) を返す。
  学習は両方の合計をそろえて行い (形の比), 全体の大きさは sum w1 / sum w0 で戻す。"""
  scale = w1.sum() / w0.sum()
  w0 = w0 / w0.sum() * len(w0)
  w1 = w1 / w1.sum() * len(w0)
  x = np.concatenate([x0, x1])
  y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])
  w = np.concatenate([w0, w1])
  mean, std = x0.mean(axis=0), x0.std(axis=0)
  idx = np.random.permutation(len(x))
  model = classifier()
  model.fit((x[idx] - mean) / std, y[idx], sample_weight=w[idx], epochs=epochs, batch_size=batchSize, validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)], verbose=0)
  f = np.clip(model.predict((xEval - mean) / std, batch_size=batchSize, verbose=0)[:, 0], 1e-6, 1 - 1e-6)
  del model
  tf.keras.backend.clear_session()   # 毎回作るモデルをメモリに溜めない
  return scale * f / (1.0 - f)


def omnifold(mc, w0, xData, wData, iterations):
  """戻り値: 各 iteration の push の重み (MC 事象ごと; 最終的な重みは w0 x push)。"""
  xGen = np.stack([mc["ptG"], mc["yG"]], axis=1)
  xReco = np.stack([mc["ptR"], mc["yR"]], axis=1)
  gen, reco = mc["genOK"].astype(bool), mc["recoOK"].astype(bool)
  push = np.ones(len(w0))
  out = []
  for it in range(iterations):
    # step 1: reco で MC (w0 x push) -> データ
    pull = push.copy()
    pull[reco] = push[reco] * reweight(xReco[reco], w0[reco] * push[reco], xData, wData, xReco[reco])
    # step 2: gen で MC (w0) -> MC (w0 x pull)
    push = pull.copy()
    push[gen] = reweight(xGen[gen], w0[gen], xGen[gen], w0[gen] * pull[gen], xGen[gen])
    out.append(push.copy())
    print(f"  iteration {it + 1}: sum w0*push (gen) = {(w0 * push)[gen].sum():.0f}")
  return out


def readNJpsi(name):
  n = {}
  for line in open(f"{dataBase}/{name}/summary.txt"):
    s = line.split()
    if s and s[0] in mcFiles:
      n[s[0]] = float(s[1])
  return n


styles = {"prior": dict(color="blue", alpha=0.5), "data": dict(color="orange", alpha=0.5),
          "unfolded": dict(histtype="step", color="black", lw=2)}


def plot(path, x, ws, labels, kinds, xlabel, bins, title):
  fig, ax = plt.subplots(figsize=(6.4, 4.8))
  for xi, wi, lab, kind in zip(x, ws, labels, kinds):
    st = styles[kind]
    ax.hist(xi, bins=bins, weights=wi, label=lab, **st)
  ax.set_yscale("log")
  ax.set_xlim(bins[0], bins[-1])
  ax.set_xlabel(xlabel)
  ax.set_ylabel("events")
  ax.set_title(title)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path)
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--classes", nargs="+", choices=list(sp.classes), default=list(sp.classes))
  ap.add_argument("--iterations", type=int, default=4)
  ap.add_argument("--nRuns", type=int, default=5, help="平均する OmniFold の回数")
  args = ap.parse_args()
  mc = loadMC()
  reco, gen = mc["recoOK"].astype(bool), mc["genOK"].astype(bool)
  bins = np.linspace(0.0, sp.ptEdges[-1], 41)

  for name in args.classes:
    print(f"[{name}]")
    d = ROOT.RDataFrame("data", f"{dataBase}/{name}/dataForOmniFold.root").AsNumpy(["pairPt", "rapidity", "weight"])
    xData, wData = np.stack([d["pairPt"], d["rapidity"]], axis=1), d["weight"]
    # 初期の重み: 各成分の reco 数を N_J/psi^reco にそろえる
    nJ = readNJpsi(name)
    w0 = np.ones(len(mc["ptG"]))
    for comp, k in enumerate(mcFiles):
      sel = mc["component"] == comp
      w0[sel] = nJ[k] / (reco & sel).sum()
    pushes = []
    for r in range(args.nRuns):
      np.random.seed(1 + r)
      tf.random.set_seed(1 + r)
      pushes.append(omnifold(mc, w0, xData, wData, args.iterations)[-1])
    push = np.mean(pushes, axis=0)
    tots = [(w0 * p)[gen].sum() for p in pushes]
    print(f"[{name}] {args.nRuns} runs: total (gen) per run = " + " ".join(f"{t:.0f}" for t in tots)
          + f"  (std {np.std(tots, ddof=1) if len(tots) > 1 else 0:.0f})")

    outDir = os.path.join(outBase, name)
    os.makedirs(outDir, exist_ok=True)
    cols = {"pt": mc["ptG"][gen], "y": mc["yG"][gen], "component": mc["component"][gen].astype(np.int32),
            "w_prior": w0[gen], "w_unfolded": (w0 * push)[gen]}
    ROOT.RDF.FromNumpy({k: np.ascontiguousarray(v) for k, v in cols.items()}).Snapshot("gen", f"{outDir}/unfolded.root")
    plot(f"{outDir}/pt_gen.png", [mc["ptG"][gen]] * 2, [w0[gen], (w0 * push)[gen]],
         ["MC, true (prior)", "OmniFolded"], ["prior", "unfolded"], r"$p_\mathrm{T}^\mathrm{gen}$ (GeV/$c$)", bins, name)
    # reco: データと, 最後の push で重み付けした MC reco (step 1 の確認)
    plot(f"{outDir}/pt_reco.png", [mc["ptR"][reco], d["pairPt"], mc["ptR"][reco]],
         [w0[reco], wData, (w0 * push)[reco]], ["MC, reco (prior)", "Data (sW_sig - FD)", "MC, reco (OmniFolded)"],
         ["prior", "data", "unfolded"], r"$p_\mathrm{T}^\mathrm{reco}$ (GeV/$c$)", bins, name)
    print(f"[{name}] sum data = {wData.sum():.0f}, unfolded coh/incoh = "
          + " / ".join(f"{(w0 * push)[gen & (mc['component'] == c)].sum():.0f}" for c in range(2)) + f" -> {outDir}")


if __name__ == "__main__":
  main()
