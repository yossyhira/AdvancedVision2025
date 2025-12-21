# AdvancedVision2025
現在作成中です．
千葉工業大学大学院未来ロボティクス専攻 2025年度 アドバンスドビジョンで作成した課題．
本リポジトリでは，Python / PyTorch を用いて実装した Jupyter Notebook（`.ipynb`）形式のコードを公開しています． 
ローカル環境での動作確認はおこなっておらず，Google Colab 上での実行を前提としています．

## 概要
MNISTの手書き数字データセットの画像を入力として何の数が書かれていたかを予測を出力するタスクをおこなった画像分類のCNNモデルです．

本モデルは中間層にReLu，損失関数は交差エントロピーを使用しています．また，最終層でクラスごとのスコア（logits）を出力し，交差エントロピー損失を用いることで Softmax に基づく多クラス分類をおこなっています．

## 特徴
- MNIST（28×28, グレースケール）対応
- 中間層：ReLU
- 出力層：Softmax(交差エントロピー損失内で適用)
- 訓練 / 検証  = **8 : 2 分割**
- Google Colab でそのまま実行可能

## ネットワーク構成
<p align="center">
  <img src="./README_Fig/block.png" width="500">
</p>

## インストール方法

1. 本リポジトリを GitHub からクローン(下記コード)，または ZIP としてダウンロードしてください．
```
git clone https://github.com/yossyhira/AdvancedVision2025.git
```
2. ダウンロードした `adv.ipynb` ファイルを [Google Drive](https://accounts.google.com/Login?hl=ja&service=writely&lp=1) にアップロードしてください．

※ MNIST データセットは torchvision.datasets.MNIST を用いて取得し，
ノートブックの初回実行時に自動的にダウンロードされます．


## 実行方法（Google Colab）

1. [Google Colab](https://colab.research.google.com/)にアクセスしてください．
2. 「ファイル」→「ノートブックを開く」→「Google Drive」から，本リポジトリの `adv.ipynb` ファイルを開いてください．
3. ランタイムの種類を「Python 3」に設定してください．
4. 「ランタイム」→「ランタイムのタイプを変更」から GPU（T4 など）を有効にしてください．
5. ノートブック上部から順にセルを実行してください．

## 動作環境

- 実行環境：Google Colab
- GPU：NVIDIA Tesla T4
- Python：3.12.12
- PyTorch：2.9.0+cu126
- torchvision：0.24.0+cu126
- NumPy：2.0.2
- Matplotlib：3.10.0
- scikit-learn：1.6.1
- CUDA：12.6

## 学習

本モデルで MNIST データセットを用いて5-epoch学習をおこないました．
学習データとして訓練用画像48,000枚，検証用画像12,000枚を使用しました(**8 : 2 分割**)．
以下は，学習時の精度（accuracy）および損失（loss）の推移グラフです．
<p align="center">
  <img src="./README_Fig/accloss.png" width="1000">
</p>

## 結果

以下は，上記の学習済みモデルでテストデータに対して予測をおこなった結果です．
テストデータとして10,000枚を使用しました．
### 精度
- テスト精度（accuracy）：98.4 %
- テスト損失（loss）：4.6 %

### 混同行列
<p align="center">
  <img src="./README_Fig/mat.png" width="750">
</p>

### 認識の成功/失敗例
上段が成功例，下段が失敗例です．
<p align="center">
  <img src="./README_Fig/exp.png" width="500">
</p>

## 謝辞

## ライセンス
* このソフトウェアパッケージは，3条項BSDライセンスの下，再頒布および使用が許可されます．

* © 2025 Yoshitaka Hirata