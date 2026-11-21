# ToDo (OmniFold unbinned unfolding, J/psi pT, Standalone muon)

流れ: `sPlot_massfit.py` → `subtractFD.py` → `omnifold_unfold.py` → `truthFit.py` → (2 に戻る: `runLoop.py`)、誤差は `bootstrap.py`。

## 今の課題 (2026-11-21)

### OmniFold の学習のばらつき
- 同じデータで乱数の種だけ変えた 5 回の合計 (gen) のばらつき: All 1393 (約 1%), 0n0n 538, Xn0n 290, 0nXn 747, XnXn 212。
  ブートストラップの誤差 (All 1990, Xn0n 242, XnXn 273) と同程度で、Xn0n・XnXn ではむしろ大きい。
- 中心値は 5 回の平均にした (`omnifold_unfold.py --nRuns 5`)。中心値とレプリカの平均のずれは全ビンで 2 sigma 以内になった。
- [ ] 各レプリカは 1 回だけなので、ブートストラップの誤差に学習のばらつきが入っている (データの統計誤差を大きめに見積もっている)。
      学習を安定させる (epochs を増やす、EarlyStopping の patience を緩める、複数回の平均) か、学習のばらつきを別に評価する。

### ループ (手順 2 → 4)
- [ ] 5 回の平均で `runLoop.py` を回し直す (前のループは学習のばらつき 2–5% のせいで 2% の条件で止まらなかった)。
      今の `truthFit/yields_reco.txt` はループの最後の周 (iter6) からさらに 1 周進めた値で、0nXn incoh +5.5%, 0n0n incoh +5.8% 動いた。

### 誤差
- [ ] ブートストラップの誤差を真値レベルのフィットの収量 (N_coh, N_incoh, N_ND) に伝える (今の truthFit の誤差は MC の統計 sqrt(sum w^2) だけ、chi2/ndf が 25–140 と大きい)。
- [ ] sWeight は固定 (レプリカごとに sPlot をやり直していない)。
- [ ] 系統誤差: FD の比 4.9%, b (`truthFit.py --bShift`), E loss の扱い (elossPtMin = 0.14), 背景モデル, OmniFold の iteration 数。

### MC と検出器の再現
- [ ] MC の運動量分解能がデータより約 20% 小さい (質量の sigma がデータ / MC = 1.23、coherent の低 pT の自由な sigma 74–85 MeV vs MC 63 MeV)。
      tuple に単一ミューオンの運動学が無いので補正できない。MC は作り直さない方針。
- [ ] 単一ミューオンの pT の下限がデータ 0.5、MC 0.3 GeV/c。OmniFold の再構成効率 (gen → reco) がこの差の分だけずれる → 断面積に直接効く。
      大きさを `AccEff/` の pT0p5 と pT0p3 の MC で見積もる。
- [ ] gen pT > 1.2 の事象が gen の標本に残っている (0nXn で重みの 5.6%)。
- [ ] Xn0n の 0.90–0.95 のスライスの質量フィットが status = 1。

### 物理
- [ ] 回折の谷 (0.15–0.35 GeV/c) の形は prior (MC の形) に依存する。
- [ ] ZN クラスの migration は補正していない (従来の解析と同じ)。

### 従来の Bayes/SVD との比較 (`compareUnfold.py`, 2026-11-21)
- incoherent + ND (gen fiducial, pT 0.3–1.2) の合計 OmniFold / Bayes iter4: All 1.01, 0nXn 0.96, XnXn 1.02, Xn0n 1.17, **0n0n 1.71**。SVD k=4 は全クラスで Bayes より 15–30% 低い。
- [ ] 0n0n (と Xn0n) の超過の原因: reco pT 0.3–1.0 の coherent の裾。
      従来は pT テンプレートフィットでここを coherent とした (0n0n の 0.3–0.4 で f_C = 4.8、0.8–0.9 でも 1.2)。
      OmniFold では coherent の MC が reco の 0.3–1.2 に 1232 事象しか流さず、データとの差が非 coherent (gen pT 0.3–0.6) に入る。
      データの sW_eloss は 0n0n の 0.3–1.0 で 2943 事象あり (sW_sig 5073 の 58%)。E loss を信号に含める扱いと、MC の分解能の不足が原因の候補。
      確認: sW_sig = sW_jpsi だけ (E loss を除く) で OmniFold を回し直して比べる。

### 断面積
- [ ] OmniFold の結果を dsigma/d|t| に直し、従来の Bayes/SVD (`upc-analysis-run3/macros/performance/CrossSection/CrossSection_Incoherent_Corrected_Step0p1/`) と比較する。
