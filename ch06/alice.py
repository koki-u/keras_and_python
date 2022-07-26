import encodings
from keras.layers import  Dense, Activation, SimpleRNN
from keras.models import Sequential
import numpy as np

Input_file = "Keras_and_Python/ch06/11-0.txt"
with open(Input_file, "r", encoding="utf-8") as f:
    lines = [line.strip().lower() for line in f
            if len(line) != 0]
    text = " ".join(lines)

#文字の集合をインデックス処理する
chars = set(text)
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

#入力テキストとラベルテキストを作成する。
SEQLEN = 10
STEP = 1

input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])

print("Vectorizing input and label text...")

#=================================================================
#入力テキストとラベルテキストをベクトル化する
#================================================================
#RNNへの入力の各行は、前に示した入力テキストの一つに対応する。
#各入力文字をnb_chars次元のone-hotエンコードされたベクトルとして表現する。
#各入力行のテンソルのサイズは(SEQLEN, nb_chars)
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1



# Build the model. We use a single RNN with a fully connected layer
# to compute the most likely predicted output char
# 値が小さすぎるとやはり自然なテキストとして表現するのは難しくなる。
HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False,
                    input_shape=(SEQLEN, nb_chars),
                    unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# We train the model in batches and test output generated at each step
for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration #: {}".format(iteration))
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    # testing model
    # randomly choose a row from input_chars, then use it to
    # generate text from model for next 100 chars
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating from seed: {}".format(test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for j, ch in enumerate(test_chars):
            Xtest[0, j, char2index[ch]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
        # move forward with test_chars + ypred
        test_chars = test_chars[1:] + ypred
    print()