#GANを使う章

#gitからkeras-adversarialというライブラリを使うことで、べんりにできる

#pip3 install git+https://github.com/bstriner/keras-adverasrial.git



#特徴
#生成モデルGと識別モデルDがそれぞれ異なるモデルということ
#LeakyReLUを使用することで、GANのパフォーマンスがあがる
#Tensor boardで確認しながら、おこなう。

#全結合層 -> 中間にConv2DとUpSampling2Dによる繰り返し->BatchNormalization正規化->LeakyReLUでつなぐ

#WaveNet
#Dilated Casual Convolutionを使うことで、畳込みフィルターに孔を開けたような
#特別な畳込みを実現する。
