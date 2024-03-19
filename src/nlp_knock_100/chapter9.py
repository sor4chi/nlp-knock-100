from .util import chapter, section
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig


category2id = {"b": 0, "t": 1, "e": 2, "m": 3}
word2id = {}
device = torch.device("mps")
id_counter = 1

VEC_FILE_PATH = "data/chapter7/GoogleNews-vectors-negative300.bin"

word2vec_model = KeyedVectors.load_word2vec_format(VEC_FILE_PATH, binary=True)


def register_word(word):
    global id_counter
    if word not in word2id:
        word2id[word] = id_counter
        id_counter += 1


def get_id(word):
    if word in word2id:
        return word2id[word]
    return 0


def clean_text(text):
    return (
        text.lower()
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
        .replace("!", "")
        .replace("?", "")
        .replace("'s", "")
        .replace("'", "")
        .replace('"', "")
    )


def tokenize(text):
    words = clean_text(text).split(" ")
    words = [word for word in words if word != ""]
    return [get_id(word) for word in words]


def start_80():
    section("80. ID番号への変換")
    # 問題51で構築した学習データ中の単語にユニークなID番号を付与したい．学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，学習データ中で2回以上出現する単語にID番号を付与せよ．そして，与えられた単語列に対して，ID番号の列を返す関数を実装せよ．ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．

    # まずは学習データを読み込む
    train_df = pd.read_csv("data/chapter6/train.csv")
    titles = train_df["TITLE"].values

    # 単語の出現回数をカウントする
    word_count = {}
    for title in titles:
        words = clean_text(title).split(" ")
        words = [word for word in words if word != ""]
        for word in words:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1

    # 出現回数が2回以上の単語にID番号を付与する
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    for word, count in sorted_word_count:
        if count < 2:
            break
        register_word(word)


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        output_size,
        padding_idx=0,
        pretrained_weights=None,
        bidirectional=False,
    ):
        super(RNN, self).__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(
                input_size, embedding_size, padding_idx=padding_idx
            )
        self.rnn = nn.RNN(
            embedding_size, hidden_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(
            x[:, -1, :]
        )  # [:, -1, :]は最後の隠れ層の出力を取り出すためのスライス
        x = self.softmax(x)
        return x


def start_81():
    section("81. RNNによる予測")
    # 80で作成したID番号の列をRNNに入力し，カテゴリを予測するモデルを実装せよ．
    # 上のRNNクラスを実装した。学習は82で行う。


def start_82():
    section("82. 確率的勾配降下法による学習")
    # 81で構築したモデルを確率的勾配降下法（SGD）により学習せよ．
    # 学習データを用いて，エポック数を10として学習せよ．

    # 学習データを読み込む
    train_df = pd.read_csv("data/chapter6/train.csv")
    test_df = pd.read_csv("data/chapter6/test.csv")
    titles = train_df["TITLE"].values
    category = train_df["CATEGORY"].values

    # モデルを構築する
    input_size = len(word2id) + 1  # id=0(未知語) + known words
    hidden_size = 50  # 隠れ層のサイズ、とりあえず指示通り50に設定
    embedding_size = 300
    output_size = len(category2id)
    model = RNN(input_size, embedding_size, hidden_size, output_size)

    # 損失関数と最適化手法を定義する
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    def accuracy(model):
        test_titles = test_df["TITLE"].values
        test_category = test_df["CATEGORY"].values
        accuracy = 0
        for title, cat in zip(test_titles, test_category):
            x = tokenize(title)
            x = torch.tensor(x, dtype=torch.int64).view(1, -1)
            y = torch.tensor([category2id[cat]], dtype=torch.int64)

            output = model(x)
            _, predicted = torch.max(output, 1)
            if predicted == y:
                accuracy += 1
        return accuracy / len(test_titles)

    loss_history = []
    accuracy_history = []

    # 学習を行う
    for epoch in range(10):
        total_loss = 0
        for title, cat in zip(titles, category):
            x = tokenize(title)
            x = torch.tensor(x, dtype=torch.int64).view(1, -1)
            y = torch.tensor([category2id[cat]], dtype=torch.int64)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"epoch: {epoch + 1}, loss: {total_loss}")
        loss_history.append(total_loss)
        acc = accuracy(model)
        print(f"accuracy: {acc}")
        accuracy_history.append(acc)

    # 検証
    print(f"final accuracy: {accuracy(model)}")

    # 損失の推移を描画する
    plt.title("loss history")
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("data/chapter9/loss_history_82.png")
    plt.close()

    # 正解率の推移を描画する
    plt.title("accuracy history")
    plt.ylim(0, 1)
    plt.plot(accuracy_history)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("data/chapter9/accuracy_history_82.png")
    plt.close()

    # 学習したモデルを保存する
    torch.save(model.state_dict(), "data/chapter9/rnn_model_82.pth")


def start_83():
    section("83. ミニバッチ化・GPU上での学習")
    # ミニバッチ化
    # 82のコードを改変し，B事例ごとに損失・勾配を計算し，行列Wの値を更新せよ（ミニバッチ化）．
    # さらに，GPU上で学習（83）を実行せよ．

    # 学習データを読み込む
    train_df = pd.read_csv("data/chapter6/train.csv")
    test_df = pd.read_csv("data/chapter6/test.csv")
    train_df["CATEGORY"] = train_df["CATEGORY"].map(category2id)
    test_df["CATEGORY"] = test_df["CATEGORY"].map(category2id)

    # モデルを構築する
    input_size = len(word2id) + 2  # id=0(未知語) + known words + padding
    padding_size = len(word2id) + 1  # テンソルを同じサイズにするためのパディングの長さ
    hidden_size = 50  # 隠れ層のサイズ、とりあえず指示通り50に設定
    embedding_size = 300
    output_size = len(category2id)
    model = RNN(input_size, embedding_size, hidden_size, output_size, padding_size).to(
        device
    )
    batch_size = 64

    def my_collate_fn(batch):
        sequences = [x[0] for x in batch]
        sequences = [
            torch.tensor(x, dtype=torch.int64, device=device) for x in sequences
        ]
        labels = [x[1] for x in batch]
        x = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_size
        )
        labels = torch.tensor(labels, dtype=torch.int64, device=device)
        return x, labels

    train_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(train_df["TITLE"], train_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate_fn,
    )
    test_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(test_df["TITLE"], test_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
    )

    # 損失関数と最適化手法を定義する
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(
    #     model.parameters(), lr=0.001
    # )  # Adamの方がロスの下がり方が早くてよかった

    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    # 学習を行う
    epochs = 30
    loaders = {"train": train_loader, "test": test_loader}
    for epoch in range(epochs):
        print(f"epoch: {epoch + 1} / {epochs}")
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            total_loss = 0
            corrects = 0
            for x, y in loaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(x)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    corrects += torch.sum(predicted == y).item()
            if phase == "train":
                loss_history.append(total_loss)
                train_accuracy_history.append(corrects / len(train_df))
            else:
                test_accuracy_history.append(corrects / len(test_df))
            print(
                f"{phase} loss: {total_loss}, accuracy: {corrects / len(loaders[phase].dataset)}"
            )

    # 検証
    print(f"final accuracy: {test_accuracy_history[-1]}")

    # 損失の推移を描画する
    plt.title("loss history")
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("data/chapter9/loss_history_83.png")
    plt.close()

    # 正解率の推移を描画する
    plt.title("accuracy history")
    plt.ylim(0, 1)
    plt.plot(train_accuracy_history, label="train")
    plt.plot(test_accuracy_history, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("data/chapter9/accuracy_history_83.png")
    plt.close()

    # 学習したモデルを保存する
    torch.save(model.state_dict(), "data/chapter9/rnn_model_83.pth")


def start_84():
    section("84. 単語ベクトルの導入")

    train_df = pd.read_csv("data/chapter6/train.csv")
    test_df = pd.read_csv("data/chapter6/test.csv")
    train_df["CATEGORY"] = train_df["CATEGORY"].map(category2id)
    test_df["CATEGORY"] = test_df["CATEGORY"].map(category2id)

    # word2vecのモデルを読み込む
    VOCAB_SIZE = len(word2id) + 2
    EMBEDDING_DIM = 300
    weights = torch.zeros(VOCAB_SIZE, EMBEDDING_DIM)
    for i, word in enumerate(word2id):
        if word in word2vec_model:
            weights[i] = torch.from_numpy(word2vec_model[word])
        else:
            weights[i] = torch.from_numpy(
                np.random.normal(scale=0.1, size=(EMBEDDING_DIM,))
            )
    weights = weights.to(device)

    # モデルを構築する
    input_size = VOCAB_SIZE
    padding_size = VOCAB_SIZE - 1
    embedding_size = 300
    hidden_size = 50  # 隠れ層のサイズ、とりあえず指示通り50に設定
    output_size = len(category2id)
    model = RNN(
        input_size, embedding_size, hidden_size, output_size, padding_size, weights
    ).to(device)
    batch_size = 64

    def my_collate_fn(batch):
        sequences = [x[0] for x in batch]
        sequences = [
            torch.tensor(x, dtype=torch.int64, device=device) for x in sequences
        ]
        labels = [x[1] for x in batch]
        x = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_size
        )
        labels = torch.tensor(labels, dtype=torch.int64, device=device)
        return x, labels

    train_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(train_df["TITLE"], train_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate_fn,
    )
    test_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(test_df["TITLE"], test_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
    )

    # 損失関数と最適化手法を定義する
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(
        model.parameters(), lr=0.001
    )  # Adamの方がロスの下がり方が早くてよかった

    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    # 学習を行う
    epochs = 30
    loaders = {"train": train_loader, "test": test_loader}
    for epoch in range(epochs):
        print(f"epoch: {epoch + 1} / {epochs}")
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            total_loss = 0
            corrects = 0
            for x, y in loaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(x)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    corrects += torch.sum(predicted == y).item()
            if phase == "train":
                loss_history.append(total_loss)
                train_accuracy_history.append(corrects / len(train_df))
            else:
                test_accuracy_history.append(corrects / len(test_df))
            print(
                f"{phase} loss: {total_loss}, accuracy: {corrects / len(loaders[phase].dataset)}"
            )

    # 検証
    print(f"final accuracy: {test_accuracy_history[-1]}")

    # 損失の推移を描画する
    plt.title("loss history")
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("data/chapter9/loss_history_84.png")
    plt.close()

    # 正解率の推移を描画する
    plt.title("accuracy history")
    plt.ylim(0, 1)
    plt.plot(train_accuracy_history, label="train")
    plt.plot(test_accuracy_history, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("data/chapter9/accuracy_history_84.png")
    plt.close()

    # 学習したモデルを保存する
    torch.save(model.state_dict(), "data/chapter9/rnn_model_84.pth")


class BiRNN(nn.Module):
    # 多層双方向RNN
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        output_size,
        padding_idx=0,
        pretrained_weights=None,
        num_layers=1,
    ):
        super(BiRNN, self).__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(
                input_size, embedding_size, padding_idx=padding_idx
            )
        self.rnn = nn.RNN(
            embedding_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x


def start_85():
    section("85. 双方向RNN・多層化")

    train_df = pd.read_csv("data/chapter6/train.csv")
    test_df = pd.read_csv("data/chapter6/test.csv")
    train_df["CATEGORY"] = train_df["CATEGORY"].map(category2id)
    test_df["CATEGORY"] = test_df["CATEGORY"].map(category2id)

    # word2vecのモデルを読み込む
    VOCAB_SIZE = len(word2id) + 2
    EMBEDDING_DIM = 300
    weights = torch.zeros(VOCAB_SIZE, EMBEDDING_DIM)
    for i, word in enumerate(word2id):
        if word in word2vec_model:
            weights[i] = torch.from_numpy(word2vec_model[word])
        else:
            weights[i] = torch.from_numpy(
                np.random.normal(scale=0.1, size=(EMBEDDING_DIM,))
            )
    weights = weights.to(device)

    # モデルを構築する
    input_size = VOCAB_SIZE
    padding_size = VOCAB_SIZE - 1
    embedding_size = 300
    hidden_size = 50  # 隠れ層のサイズ、とりあえず指示通り50に設定
    output_size = len(category2id)
    model = BiRNN(
        input_size, embedding_size, hidden_size, output_size, padding_size, weights, 3
    ).to(device)
    batch_size = 64

    def my_collate_fn(batch):
        sequences = [x[0] for x in batch]
        sequences = [
            torch.tensor(x, dtype=torch.int64, device=device) for x in sequences
        ]
        labels = [x[1] for x in batch]
        x = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_size
        )
        labels = torch.tensor(labels, dtype=torch.int64, device=device)
        return x, labels

    train_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(train_df["TITLE"], train_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate_fn,
    )
    test_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(test_df["TITLE"], test_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
    )

    # 損失関数と最適化手法を定義する
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(
        model.parameters(), lr=0.001
    )  # Adamの方がロスの下がり方が早くてよかった

    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    # 学習を行う
    epochs = 30
    loaders = {"train": train_loader, "test": test_loader}

    for epoch in range(epochs):
        print(f"epoch: {epoch + 1} / {epochs}")
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            total_loss = 0
            corrects = 0
            for x, y in loaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(x)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    corrects += torch.sum(predicted == y).item()
            if phase == "train":
                loss_history.append(total_loss)
                train_accuracy_history.append(corrects / len(train_df))
            else:
                test_accuracy_history.append(corrects / len(test_df))
            print(
                f"{phase} loss: {total_loss}, accuracy: {corrects / len(loaders[phase].dataset)}"
            )

    # 検証
    print(f"final accuracy: {test_accuracy_history[-1]}")

    # 損失の推移を描画する
    plt.title("loss history")
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("data/chapter9/loss_history_85.png")
    plt.close()

    # 正解率の推移を描画する
    plt.title("accuracy history")
    plt.ylim(0, 1)
    plt.plot(train_accuracy_history, label="train")
    plt.plot(test_accuracy_history, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("data/chapter9/accuracy_history_85.png")
    plt.close()

    # 学習したモデルを保存する
    torch.save(model.state_dict(), "data/chapter9/rnn_model_85.pth")


class TextCNN(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        output_size,
        padding_idx=0,
        pretrained_weights=None,
    ):
        super(TextCNN, self).__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(
                input_size, embedding_size, padding_idx=padding_idx
            )
        self.conv1 = nn.Conv2d(1, 100, (3, embedding_size))
        self.conv2 = nn.Conv2d(1, 100, (4, embedding_size))
        self.conv3 = nn.Conv2d(1, 100, (5, embedding_size))
        self.fc = nn.Linear(300, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = torch.relu(self.conv1(x)).squeeze(3)
        x2 = torch.relu(self.conv2(x)).squeeze(3)
        x3 = torch.relu(self.conv3(x)).squeeze(3)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat([x1, x2, x3], 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def start_86():
    section("86. 畳み込みニューラルネットワーク (CNN)")
    # 85で作成したモデルに加えて，畳み込みニューラルネットワーク（CNN）を実装せよ．
    # さらに，畳み込みニューラルネットワークを用いたモデルと，RNNを用いたモデルの両方を学習し，比較せよ．

    train_df = pd.read_csv("data/chapter6/train.csv")
    test_df = pd.read_csv("data/chapter6/test.csv")
    train_df["CATEGORY"] = train_df["CATEGORY"].map(category2id)
    test_df["CATEGORY"] = test_df["CATEGORY"].map(category2id)

    # word2vecのモデルを読み込む
    VOCAB_SIZE = len(word2id) + 2
    EMBEDDING_DIM = 300
    weights = torch.zeros(VOCAB_SIZE, EMBEDDING_DIM)
    for i, word in enumerate(word2id):
        if word in word2vec_model:
            weights[i] = torch.from_numpy(word2vec_model[word])
        else:
            weights[i] = torch.from_numpy(
                np.random.normal(scale=0.1, size=(EMBEDDING_DIM,))
            )
    weights = weights.to(device)

    # モデルを構築する
    input_size = VOCAB_SIZE
    padding_size = VOCAB_SIZE - 1
    embedding_size = 300
    output_size = len(category2id)
    model = TextCNN(input_size, embedding_size, output_size, padding_size, weights).to(
        device
    )
    batch_size = 64

    def my_collate_fn(batch):
        sequences = [x[0] for x in batch]
        sequences = [
            torch.tensor(x, dtype=torch.int64, device=device) for x in sequences
        ]
        labels = [x[1] for x in batch]
        x = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=padding_size
        )
        labels = torch.tensor(labels, dtype=torch.int64, device=device)
        return x, labels

    train_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(train_df["TITLE"], train_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate_fn,
    )

    test_loader = DataLoader(
        [
            (tokenize(title), cat)
            for title, cat in zip(test_df["TITLE"], test_df["CATEGORY"])
        ],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
    )

    # 損失関数と最適化手法を定義する
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    epochs = 30
    loaders = {"train": train_loader, "test": test_loader}
    for epoch in range(epochs):
        print(f"epoch: {epoch + 1} / {epochs}")
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            total_loss = 0
            corrects = 0
            for x, y in loaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(x)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    corrects += torch.sum(predicted == y).item()
            if phase == "train":
                loss_history.append(total_loss)
                train_accuracy_history.append(corrects / len(train_df))
            else:
                test_accuracy_history.append(corrects / len(test_df))
            print(
                f"{phase} loss: {total_loss}, accuracy: {corrects / len(loaders[phase].dataset)}"
            )

    # 検証
    print(f"final accuracy: {test_accuracy_history[-1]}")

    # 損失の推移を描画する
    plt.title("loss history")
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("data/chapter9/loss_history_86.png")
    plt.close()

    # 正解率の推移を描画する
    plt.title("accuracy history")
    plt.ylim(0, 1)
    plt.plot(train_accuracy_history, label="train")
    plt.plot(test_accuracy_history, label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("data/chapter9/accuracy_history_86.png")
    plt.close()

    # 学習したモデルを保存する
    torch.save(model.state_dict(), "data/chapter9/cnn_model_86.pth")


def start_87():
    section("87. 確率的勾配降下法によるCNNの学習")
    pass  # 86で実装済み


def start_88():
    section("88. パラメータチューニング")
    pass  # 86で実装済み


class BertDataset(Dataset):
    def __init__(self, tokenizer, titles, categories):
        self.tokenizer = tokenizer
        self.titles = titles
        self.categories = categories

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        inputs = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=20,
            pad_to_max_length=True,
            truncation=True,
        )
        return {
            "input_ids": torch.tensor(
                inputs["input_ids"], dtype=torch.int64, device=device
            ),
            "attention_mask": torch.tensor(
                inputs["attention_mask"], dtype=torch.int64, device=device
            ),
            "labels": torch.tensor(
                self.categories[idx], dtype=torch.int64, device=device
            ),
        }


class BertNewsClassifier(nn.Module):
    def __init__(self, model, dropout_rate=0.1, hidden_size=768, output_size=4):
        super(BertNewsClassifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = output["last_hidden_state"][:, 0, :]
        output = self.fc(output)
        return output


def start_89():
    section("89. 事前学習済み言語モデルからの転移学習")
    # 事前学習済み言語モデル（例えばBERTなど）を使用し，ニュースカテゴリ分類モデルを実装せよ．

    train_df = pd.read_csv("data/chapter6/train.csv")
    valid_df = pd.read_csv("data/chapter6/valid.csv")
    test_df = pd.read_csv("data/chapter6/test.csv")
    train_df["CATEGORY"] = train_df["CATEGORY"].map(category2id)
    valid_df["CATEGORY"] = valid_df["CATEGORY"].map(category2id)
    test_df["CATEGORY"] = test_df["CATEGORY"].map(category2id)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = BertDataset(tokenizer, train_df["TITLE"], train_df["CATEGORY"])
    valid_dataset = BertDataset(tokenizer, valid_df["TITLE"], valid_df["CATEGORY"])
    test_dataset = BertDataset(tokenizer, test_df["TITLE"], test_df["CATEGORY"])

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BertNewsClassifier(BertModel.from_pretrained("bert-base-uncased")).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    loss_history = []
    train_accuracy_history = []
    valid_accuracy_history = []

    epochs = 10
    loaders = {"train": train_loader, "valid": valid_loader}
    for epoch in range(epochs):
        print(f"epoch: {epoch + 1} / {epochs}")
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            total_loss = 0
            corrects = 0
            for batch in loaders[phase]:
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                with torch.set_grad_enabled(phase == "train"):
                    output = model(input_ids, attention_mask)
                    loss = criterion(output, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    corrects += torch.sum(predicted == labels).item()
            if phase == "train":
                loss_history.append(total_loss)
                train_accuracy_history.append(corrects / len(train_df))
            else:
                valid_accuracy_history.append(corrects / len(valid_df))
            print(
                f"{phase} loss: {total_loss}, accuracy: {corrects / len(loaders[phase].dataset)}"
            )

    # testで検証
    model.eval()
    corrects = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids, attention_mask)
        _, predicted = torch.max(output, 1)
        corrects += torch.sum(predicted == labels).item()
    print(f"test accuracy: {corrects / len(test_df)}")

    # 損失の推移を描画する
    plt.title("loss history")
    plt.plot(loss_history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("data/chapter9/loss_history_89.png")
    plt.close()

    # 正解率の推移を描画する
    plt.title("accuracy history")
    plt.ylim(0, 1)
    plt.plot(train_accuracy_history, label="train")
    plt.plot(valid_accuracy_history, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("data/chapter9/accuracy_history_89.png")
    plt.close()

    # 学習したモデルを保存する
    torch.save(model.state_dict(), "data/chapter9/bert_model_89.pth")


def start():
    chapter("第9章: RNN, CNN")
    start_80()
    start_81()
    start_82()
    start_83()
    start_84()
    start_85()
    start_86()
    start_87()
    start_88()
    start_89()
