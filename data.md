# Input data
## 実データ
```
/media/takuma/ESD-EAWA/Data/UPCcandMuon/StandaloneMuon/Unbinned/pass4/incoherent/testZDC/AnalysisResults.root
```
- TTree: `my-upc-mass-global-muon/unbinnedFitPairTuple`

## MCデータ
```
/media/takuma/ESD-EAWA/Data/UPCcandMuon/MC/Standalone/Omnifold/{jpsi-coh,jpsi-incoh,psi2s-coh-fd,psi2s-incoh-fd}.root
```
- TTree: `upc-muon-analysis/omniFoldPairTuple` と `upc-muon-analysis/omniFoldRecoOnlyTuple `

# 解析フロー
初めに、実データとMCで、質量分布、ｐT分布、ラピディティ分布をみる。(m, pT, y)で質量分布から背景事象を除く。
## 背景事象を取り除く
sPlotを使って、背景事象を取り除く。ここではまだ、Binにしない。
$$\gamma \gamma \rightarrwo \mu \mu$$成分をここで取り除く。

Coherent成分に関しては、pTが0であることから、質量とｐTの相関が非常に強い。そのため、ｐT範囲ごとにスライスして、sWeightをかけてやるのがよい？ただし、スライスの幅を変えるなどして、スライス間の異常がないか確認することは重要。

## feed-downの差し引き

## pT template fit


