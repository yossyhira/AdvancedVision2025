# AdvancedVision2025
千葉工業大学大学院未来ロボティクス専攻 2025年度 アドバンスドビジョンで作成した課題．

本リポジトリでは，Python / PyTorch を用いて実装した Jupyter Notebook（`.ipynb`）形式のコードを公開しています． 
ローカル環境での動作確認はおこなっておらず，Google Colab 上での実行を前提としています．

## 目次
- [概要](#概要)
- [特徴](#特徴)
- [ネットワーク構造](#ネットワーク構造)
  - [ブロック図](#ブロック図)
  - [数式](#数式)
- [実行方法（Google Colab）](#実行方法google-colab)
- [動作環境](#動作環境)
- [性能評価](#性能評価)
- [参考文献](#参考文献)
- [ライセンス](#ライセンス)


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
入力画像 $\boldsymbol{x}$ は 次式で表されます．

$$
\boldsymbol{x} \in \mathbb{R}^{1 \times 28 \times 28}
$$

#### 第1畳み込み層＋ReLU
入力画像 $\boldsymbol{x}$ に対して $5 \times 5$ のカーネルを用いた畳み込み演算をおこないます．
入力画像の空間サイズを保持するため，パディング（padding）を施した上で畳み込みをします．
パディングとは，入力画像の周囲に画素を追加する操作であり，本モデルではゼロパディング(囲う画素値が0)を使用し,画像の周囲を2画素分拡張します．
したがって，パディング後の入力画像は $\tilde{\boldsymbol{x}} \in \mathbb{R}^{1 \times (28 + 4) \times (28 + 4)}$ となります．

畳み込み後の出力チャンネル数を 6とし， $k$ 番目のチャンネルにおける出力特徴マップの位置 $(i, j)$ の値は次式で与えられます．
また，本層では入力チャンネル数が 1 であるため，チャンネル方向の総和は省略して記述をします．

$$
z^{(1)}_k(i, j) = \sum_{u=1}^{5}\sum_{v=1}^{5} w^{(1)}_k(u, v) \tilde{x}(i - 1 + u, j - 1 + v) + b^{(1)}_k
$$


* (i, j): 畳み込み後の出力特徴マップにおける画素の位置
* (u, v): 畳み込みカーネルの座標系における画素の位置
* $\tilde{x}(i - 1 + u, j - 1 + v)$: 入力画像の画素値
* $w^{(1)}_k(u, v)$ : 第1畳み込み層における $k$ 番目の出力チャンネルのカーネル重み  
* $b^{(1)}_k$ : $k$ 番目の出力チャンネルに対応するバイアス 

* $\boldsymbol{z}^{(1)}_k \in \mathbb{R}^{28 \times 28}$ : 第1畳み込み層における $k$ 番目のチャンネルの畳み込み計算後の出力特徴マップ

* $z^{(1)}_k(i, j)$ : 出力特徴マップにおける位置 (i,j) の要素


得られた畳み込み出力 $z^{(1)}_k(i, j)$ に対して，次式に示す ReLU 関数を適用します．

$$
h^{(1)}_k(i, j) = \max\bigl(0, z^{(1)}_k(i, j)\bigr)
$$


* $\boldsymbol{h}^{(1)}_k \in \mathbb{R}^{28 \times 28}$ :  $k$ 番目のチャンネルのReLU関数計算後の出力特徴マップ

* $h^{(1)}_k(i, j)$ : 出力特徴マップにおける位置 (i,j) の要素


#### 第1プーリング層
本モデルではプーリングの手法として次式で表されるmax poolingを使用します．
max poolingとはプーリング窓内の最大値のみを残す手法で, 本層では $2 \times 2$ の窓を使用します．
具体的には，ReLU関数適用後の出力 $\boldsymbol{h}^{(1)}$ の各チャンネルに対して次式を計算します．

$$
p^{(1)}_k(i, j) = \max_{\substack{0 \le a < 2 \\ 0 \le b < 2}}h^{(1)}_k(i \cdot s + a, j \cdot s + b) \quad (s = 2)
$$



* $\boldsymbol{p}^{(1)}_k \in \mathbb{R}^{14 \times 14}$ : 第1プーリング層におけるkチャンネル目のmax poolng後の出力特徴マップ

* $p^{(1)}_k(i, j)$ : 出力特徴マップ上の位置 (i,j) における要素

* $a, b$ :プーリング窓内の位置を示すインデックス

* $s$：ストライド

ここでいうストライドとは，プーリング窓の移動量のことです．

#### 第2畳み込み層

第1プーリング層の出力特徴マップ $\boldsymbol{p^{(1)}}$ を入力として，6 チャンネルに対する $5 \times 5$ カーネルを用いた畳み込み演算をおこないます．
本層ではパディングは使用しません．
出力チャンネル数を 16 とし， $k$ 番目の出力チャンネルにおける特徴マップ上の位置 $(i, j)$ の出力値は次式で与えられます．

$$
z^{(2)}_k(i, j) = \sum_{c=1}^{6} \sum_{u=1}^{5} \sum_{v=1}^{5} w_{k,c}(u, v) p^{(1)}_c(i - 1 + u, j - 1 + v) + b^{(2)}_k
$$


* $(i, j)$：畳み込み後の出力特徴マップにおける画素位置  
* $(u, v)$：畳み込みカーネル内の画素位置  
* $p^{(1)}_c(i - 1 + u, j - 1 + v)$：第1プーリング層の $c$ 番目チャンネルにおける入力画素値  
* $w_{k,c}(u, v)$：出力チャネル $k$，入力チャネル $c$ に対応する畳み込みカーネルの重み  
* $b^{(2)}_k$：出力チャネル $k$ に対応するバイアス  
* $\boldsymbol{z}^{(2)}_k \in \mathbb{R}^{10 \times 10}$ : 第2畳み込み層における $k$ 番目のチャンネルの畳み込み計算後の出力特徴マップ
* $z^{(2)}_k(i, j)$：出力特徴マップにおける位置 (i,j) の要素

得られた畳み込み出力 $z^{(2)}_k(i, j)$ に対して，次式に示す ReLU 関数を適用する．

$$
h^{(2)}_k(i, j) = \max(0, z^{(2)}_k(i, j))
$$
* $\boldsymbol{h}^{(2)}_k \in \mathbb{R}^{10 \times 10}$ :  $k$ 番目のチャンネルのReLU関数計算後の出力特徴マップ

* $h^{(2)}_k(i, j)$ : 出力特徴マップにおける位置 (i,j) の要素



#### 第2プーリング層

本モデルでは第1プーリング層と同様に，プーリング手法として max pooling を使用します($2 \times 2$ のプーリング窓).  
第2畳み込み層の ReLU関数適用後の出力特徴マップ $\boldsymbol{h}^{(2)}$ の各チャンネルに対して次式を計算します．

$$
p^{(2)}_k(i, j) = \max_{\substack{0 \le a < 2 \\ 0 \le b < 2}}h^{(2)}_k(i \cdot s + a, j \cdot s + b) \quad (s = 2)
$$


* $\boldsymbol{p}^{(2)}_k \in \mathbb{R}^{5 \times 5} $：第2プーリング層における $k$ 番目チャンネルの出力特徴マップ  
* $p^{(2)}_k(i, j)$：出力特徴マップ上の位置 $(i, j)$ における要素  
* $a, b$ :プーリング窓内の位置を示すインデックス  
* $s$：ストライド



#### 第1全結合層＋ReLU
全結合層に入力するために，第2プーリング層の出力である特徴マップ $\boldsymbol{p}^{(2)} \in \mathbb{R}^{16 \times 5 \times 5}$ の全要素を並べ替え，400 個の成分を持つベクトル  $\boldsymbol{f}$ に変換（flatten）します．このとき， $\boldsymbol{f}\in\mathbb{R}^{400}$ の列ベクトルとなります．また，これ以降 $\mathbb{R}^{n}$ と表記した場合は，列ベクトル(n行1列)と定義します．

次式に示すように，この $\boldsymbol{f}$ を第1全結合層に入力して120次元のベクトルを出力します．

$$
\boldsymbol{z}^{(3)}=\boldsymbol{W}^{(1)}\boldsymbol{f} + \boldsymbol{b}^{(3)}
$$

* $\boldsymbol{W}^{(1)}\in\mathbb{R}^{120\times 400}$：第1全結合層の重み行列  
* $\boldsymbol{b}^{(3)}\in\mathbb{R}^{120}$：第1全結合層のバイアス  
* $\boldsymbol{z}^{(3)}\in\mathbb{R}^{120}$： 第1全結合層の全結合層の出力

得られた $\boldsymbol{z}^{(3)}$ の各成分 $i$ に対して，次式で表される ReLU 関数を適用します．

$$
h^{(3)}_i = \max(0, z^{(3)}_i)
$$

* $\boldsymbol{h}^{(3)}\in\mathbb{R}^{120}$： ReLU計算後の120次元のベクトル
* $h^{(3)}_i$： ReLU計算後の出力


#### 第2全結合層＋ReLU
次式に示すように，第1全結合層とReLU関数を経た出力 $\boldsymbol{h}^{(3)}$ を第2全結合層に入力して84次元のベクトルを出力します．

$$
\boldsymbol{z}^{(4)}=\boldsymbol{W}^{(2)}\boldsymbol{h}^{(3)} + \boldsymbol{b}^{(4)}
$$

* $\boldsymbol{W}^{(2)}\in\mathbb{R}^{84\times 120}$：第2全結合層の重み行列  
* $\boldsymbol{b}^{(4)}\in\mathbb{R}^{84}$：第2全結合層のバイアス  
* $\boldsymbol{z}^{(4)}\in\mathbb{R}^{84}$： 第2全結合層の全結合層の出力

得られた $\boldsymbol{z}^{(4)}$ の各成分 $i$ に対して，次式で表される ReLU 関数を適用します．

$$
h^{(4)}_i = \max(0, z^{(4)}_i)
$$

* $\boldsymbol{h}^{(4)}\in\mathbb{R}^{84}$： ReLU計算後の84次元のベクトル
* $h^{(4)}_i$： ReLU計算後の出力



#### 最終層(全結合層＋Softmax)
次式に示すように，第2全結合層とReLU関数を経た出力 $\boldsymbol{h}^{(4)}$ を最終全結合層に入力して10次元のベクトルを出力します．

$$
\boldsymbol{z}^{(5)}=\boldsymbol{W}^{(3)}\boldsymbol{h}^{(4)} + \boldsymbol{b}^{(5)}
$$

* $\boldsymbol{W}^{(3)}\in\mathbb{R}^{10\times 84}$：最終全結合層の重み行列  
* $\boldsymbol{b}^{(5)}\in\mathbb{R}^{10}$：最終全結合層のバイアス  
* $\boldsymbol{z}^{(5)}\in\mathbb{R}^{10}$： 最終全結合層の全結合層の出力

最終全結合層を経て出力された $\boldsymbol{z}^{(5)}$ はlogitsと呼ばれ，後述するSoftmax関数で確率化する前のクラスごとにおけるスコアとなっています．

最終全結合層で出力されたlogitsを確率に直す必要があるので，次式で示すSoftmax関数を適用します．

$$
y_i = \frac{\exp(z^{(5)}_i)}{\sum_{u=1}^{10} \exp(z^{(5)}_u)} 
$$

* $i$：i個目のクラス
* $z^{(5)}_i$：最終全結合層の出力 $\boldsymbol{z}^{(5)}$ におけるクラス $i$ の logits  
* $z^{(5)}_u$：最終全結合層の出力 $\boldsymbol{z}^{(5)}$ におけるクラス $u$ の logits  
* $\boldsymbol{y}\in\mathbb{R}^{10}$：Softmax 関数によって得られる，各クラスに対する予測確率ベクトル
* $y_i$：入力画像がクラス $i$ に属する予測確率  

なお，実装では PyTorch の仕様により，交差エントロピー損失関数の内部で
LogSoftmax 関数を使用しているため，暗黙的にSoftmax関数が適用されており，プログラム内で明示的に Softmax関数 を使用していません．
また, 学習済みモデルを用いた分類時にはSoftmax関数を通さずlogitsのスコアが最も高いクラスを予測結果としています．理由は，Softmaxを通すか否かにかかわらず選ばれるスコアが最も高いクラスは同じだからです．

#### 損失関数
本モデルは Softmax 関数によりクラスごとの確率分布を出力する多クラス分類問題であるため，
損失関数として交差エントロピー損失を使用します．
正解ラベルを one-hot 表現として， $\boldsymbol{y}^*$ で表します．one-hot 表現とは，正解ラベルを1,それ以外のラベルを0としたベクトルです．
交差エントロピー損失は次式で表されます．

$$
\mathcal{L} = -\sum_{i=1}^{10} y^*_i \log(y_i)
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

- 以下は数式を書く際に参考にした文献です．
  - 畳み込み層：[【深層学習】CNNまとめ（仕組み、ちょっとだけ数式）](https://qiita.com/nakamin/items/5096924cf4460054077d)
  - プーリング層：[Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
  - 全結合層：[PyTorch:Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

## ライセンス
* このソフトウェアパッケージは，3条項BSDライセンスの下，再頒布および使用が許可されます．

* © 2025 Yoshitaka Hirata