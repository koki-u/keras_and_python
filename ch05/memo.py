#========================================================
#単語分散表現
#========================================================

#単語分散表現：語彙内の単語やフレーズをベクトルにマッピングする
#テキストを数値ベクトルとして変換する。

#one-hot エンコーディングはコーパス内の任意の2単語環の内積が0になる
#->情報検索分野の手法を用いて単語をベクトル化した。
#有名な手法としては、TF-IDF, 潜在意味解析, トピックモデル

#分散表現は、ある単語の意味を、その単語の文脈中に出現する別の単語との関係によって捉えることを目指している。
#woed2vec/GloVeを見ていく

#======================================================
#word2vec
#======================================================

#教師なし学習で、入力として大きなテキストコーパスをあたえて、出力として単語のベクトル空間を生成する。
#one-hotエンコーディングよりも密な表現をできる

#アーキテクチャとして有名なのは
#CBOWとSkip-gram
#CBOW:モデルに文脈語を与え、その中心語を予測する。さらに、文脈語の順序は予測に影響を与えないと仮定する。
#Skip-gram:モデルに中心語を与え、その文脈語を予測する。

#CBOWはSkip-gramより高速であるものの、Skip-gramの方が低頻度語の予測に優れている
