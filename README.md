# AdvancedVision2025
現在作成中です．
千葉工業大学大学院未来ロボティクス専攻 2025年度 アドバンスドビジョンで作成した課題．
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

本リポジトリは Python / PyTorch を用いた Jupyter Notebook（`.ipynb`）形式のコードです．  
ローカル環境での動作確認はおこなっておらず，Google Colab 上での実行を前提としています．

1. 本リポジトリを GitHub からクローン(下記コード)，または ZIP としてダウンロードする
```
git clone https://github.com/yossyhira/AdvancedVision2025.git
```
2. ダウンロードした `adv.ipynb` ファイルを [Google Drive](https://accounts.google.com/Login?hl=ja&service=writely&lp=1) にアップロードする

## 実行方法（Google Colab）

1. [Google Colab](https://colab.research.google.com/)にアクセスする
2. 「ファイル」→「ノートブックを開く」→「Google Drive」から，本リポジトリの `adv.ipynb` ファイルを開く
3. ランタイムの種類を「Python 3」に設定する
4. 必要に応じて「ランタイム」→「ランタイムのタイプを変更」から GPU（T4 など）を有効にする
5. ノートブック上部から順にセルを実行する
## 動作環境
### 必要なソフトウェア　

## ライセンス
* このソフトウェアパッケージは，3条項BSDライセンスの下，再頒布および使用が許可されます．

* © 2025 Yoshitaka Hirata