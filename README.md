# AdvancedVision2025
現在作成中です．
千葉工業大学大学院未来ロボティクス専攻 2025年度 アドバンスドビジョンで作成した課題．
本リポジトリは Python / PyTorch を用いた Jupyter Notebook（`.ipynb`）形式のコードです．  
ローカル環境での動作確認はおこなっておらず，Google Colab 上での実行を前提としています．

## 概要
MNISTの手書き数字データセットの画像を入力して何の数が書かれていたかを出力するタスクをおこなった画像分類のCNNモデルです．

本モデルは中間層にReLu，損失関数はクロスエントロピーを使用しています．また，最終層でクラスごとのスコア（logits）を出力し，交差エントロピー損失を用いることで Softmax に基づく多クラス分類をおこなっています．

## 特徴
- MNIST（28×28, グレースケール）対応
- 中間層：ReLU
- 出力層：Softmax(交差エントロピー損失内で適用)
- 訓練 / 検証 / テスト = **7 : 2 : 1 分割**
- Google Colab でそのまま実行可能

## ネットワーク構成
<p align="center">
  <img src="https://github.com/user-attachments/assets/20b676da-cb75-4ccb-9c58-7de2746f71d8" width="500">
</p>

## インストール方法

1. 本リポジトリを GitHub からクローン(下記コード)，または ZIP としてダウンロードする
```
git clone https://github.com/yossyhira/AdvancedVision2025.git
```
2. ダウンロードした `adv.ipynb` ファイルを [Google Drive](https://accounts.google.com/Login?hl=ja&service=writely&lp=1) にアップロードする

※ MNIST データセットは torchvision.datasets.MNIST を用いて取得し，
ノートブックの初回実行時に自動的にダウンロードされます．


## 実行方法（Google Colab）

1. [Google Colab](https://colab.research.google.com/)にアクセスする
2. 「ファイル」→「ノートブックを開く」→「Google Drive」から，本リポジトリの `adv.ipynb` ファイルを開く
3. ランタイムの種類を「Python 3」に設定する
4. 「ランタイム」→「ランタイムのタイプを変更」から GPU（T4 など）を有効にする
5. ノートブック上部から順にセルを実行する

## 動作環境

本コードは以下の環境で動作確認をおこなっています．

- 実行環境：Google Colab
- Python：3.12.12
- PyTorch：2.9.0+cu126
- torchvision：0.24.0+cu126
- NumPy：2.0.2
- Matplotlib：3.10.0
- scikit-learn：1.6.1
- CUDA：12.6

## 学習

本モデルは MNIST データセットを用いて5-epoch学習をおこなった．
学習時の精度（accuracy）および損失（loss）の推移グラフを示す．
<p align="center">
  <img src="https://github.com/user-attachments/assets/11e9d848-5cf8-4cb6-a394-78d113f5e51a" width="700">
</p>

## 結果

上記の学習済みモデルでテストデータに対して推論をおこなった結果を以下に示す．

### 精度
- テスト精度（accuracy）：98.4 %
- テスト損失（loss）：4.6 %

### 混同行列
<p align="center">
  <img src="https://github.com/user-attachments/assets/89ecee1f-28ac-431c-b2b6-5283bbba268c" width="500">
</p>

### 認識の成功/失敗例
上段が成功例，下段が失敗例です．
<p align="center">
  <img src="https://github.com/user-attachments/assets/81f34af8-0a8a-46ed-9805-ef713a7ba703" width="500">
</p>

## ライセンス
* このソフトウェアパッケージは，3条項BSDライセンスの下，再頒布および使用が許可されます．

* © 2025 Yoshitaka Hirata