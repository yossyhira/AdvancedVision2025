[日本語](README.md) || [English](README_EN.md)

# AdvancedVision2025
千葉工業大学大学院未来ロボティクス専攻 2025年度 アドバンスドビジョンで作成した課題．

本リポジトリでは，Python / PyTorch を用いて実装した Jupyter Notebook（`.ipynb`）形式のコードを公開しています． 
ローカル環境での動作確認はおこなっておらず，Google Colab 上での実行を前提としています．

## 概要
本モデルは28×28のグレースケール画像を入力として，10クラスの分類をおこなうCNN モデルです．

本モデルの性能評価としてMNISTデータセットを使用して5-epoch学習をした結果，精度98.6% 損失4.3%となりました.

## 特徴
- MNIST（28×28, グレースケール）対応
- 中間層：ReLU
- 出力層：Softmax
- Google Colab でそのまま実行可能

## ネットワーク構造
本モデルは2つの畳込み層とプーリング層，3つの全結合層から構成されています.
そして，28×28のグレースケール画像を入力として，中間層にReLu関数を使用し，最終層でSoftmax関数に基づく10クラス分類をおこなっています．

以下に本モデルのネットワーク構造のブロック図とアルゴリズムの数式を示します.

### ブロック図
<p align="center">
  <img src="./README_Fig/block.png" width="500">
</p>

### 数式
#### 入力
入力画像 $\mathbf{x}$ は 次式で表されます．

$$
\mathbf{x} \in \mathbb{R}^{1 \times 28 \times 28}
$$

#### 第1畳み込み層＋ReLU
入力画像 $\mathbf{x}$に対して5*5のカーネルを用いた畳み込み演算をします．
出力チャンネルは6チャンネルあり，k番目のチャンネルの(i, j)画素目は 次式で表すことができる．

$$
z_k(i, j) = \sum_{n=1}^{5} w(i, j)\, x(i, j) + b_k
$$








## 実行方法（Google Colab）

1. 以下のボタンから，Google Colab 上でノートブックを直接開くことができます．

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yossyhira/AdvancedVision2025/blob/main/adv.ipynb)

2. ランタイムのタイプを「Python 3」に設定し, ハードウェア アクセラレータを GPU（T4 推奨）に設定してください(赤丸部分)．
<p align="center">
  <img src="./README_Fig/Colab_jp.png" width="500">
</p>

3. ノートブック上部から順にセルを実行してください(実行にはgoogleアカウントへログイン必須)．




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

## 性能評価
本モデルの性能評価としてMNISTデータセットを使用しました.
MNISTデータセットは手書き数字(0~9)の画像データセットで60,000枚の学習用データと10,000枚のテスト用データで構成されています.

以下は学習内容とその結果です.
### 学習
60,000枚の学習用データを訓練用画像と検証用画像で8:2に分割しました(訓練用画像48,000枚，検証用画像12,000枚).

以下は，5-epoch学習させた際の精度（accuracy）および損失（loss）の推移グラフです．
<p align="center">
  <img src="./README_Fig/accloss.png" width="1000">
</p>

### 結果
以下は，上記の学習済みモデルでテスト用データに対して予測をおこなった結果です．
 
#### 精度
- テスト精度（accuracy）：98.6 %
- テスト損失（loss）：4.3 %

#### 混同行列
以下はテストデータに対する予測結果の混同行列です．
縦軸は正解ラベル，横軸はモデルによる予測ラベルです．
<p align="center">
  <img src="./README_Fig/mat.png" width="750">
</p>

#### 認識の成功/失敗例
上段が成功例，下段が失敗例です．
<p align="center">
  <img src="./README_Fig/exp.png" width="500">
</p>

## 参考文献
- 性能評価で使用した MNIST データセットは，Y. LeCun らによって公開されたデータセットであり，本実装では [torchvision.datasets.MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) を通じて取得したものを使用しました．

- CNNモデルを構成するためのPythonのコードの書き方について下記のサイトを参考にさせて頂きました.
  - [畳み込みニューラルネットワーク(CNN)をわかりやすく基本から実装まで解説](https://zero2one.jp/learningblog/cnn-for-beginners/)

- CNNモデルのネットワーク構成について下記の論文を参考にさせて頂きました.
  - [LeNet](https://direct.mit.edu/neco/article-abstract/1/4/541/5515/Backpropagation-Applied-to-Handwritten-Zip-Code)
  - [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- CNNのネットワーク構成のブロック図は下記のサイトを参考にさせて頂きました.
  - [File:AlexNet_block_diagram.svg](https://commons.wikimedia.org/wiki/File:AlexNet_block_diagram.svg)

## ライセンス
* このソフトウェアパッケージは，3条項BSDライセンスの下，再頒布および使用が許可されます．

* © 2025 Yoshitaka Hirata