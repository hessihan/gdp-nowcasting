機械学習を用いた日本のGDPナウキャスト
====

## 疑似コード
### 疑似ナウキャスト

![alg1](https://user-images.githubusercontent.com/59720853/72235602-6c289200-3616-11ea-9360-b869d0dff561.png)

### 予測時点版検証を用いた真正ナウキャスト

![alg2](https://user-images.githubusercontent.com/59720853/72235619-882c3380-3616-11ea-9e41-b2fa6ef56949.png)

### 1次速報値データ検証を用いた真正ナウキャスト

![alg3](https://user-images.githubusercontent.com/59720853/72235631-95e1b900-3616-11ea-927d-0c88de692dfd.png)



## ファイルの概要
* main.py
realnowcast.pyで定義したクラスを用いて、15日前、45日前、75日前の疑似・真正ナウキャストを実行する。realnowcast.Mode.show()でプリントされる結果が表示される。検証(バリデーション)で用いるハイパーパラメータのリストを与えて実行する。各検証の様子をvisualize.pyで定義した関数を用いて図示する。

* realnowcast.py

* masterdata_data_xarray.pkl
OECD.statsより入手したリアルタイムデータを編集して、xarray.Datasetオブジェクトとして保存した。
