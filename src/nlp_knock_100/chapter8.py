from .util import chapter, section
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


VEC_FILE_PATH = "data/chapter7/GoogleNews-vectors-negative300.bin"
PATH_TRAIN = "data/chapter6/train.csv"
PATH_VALID = "data/chapter6/valid.csv"
PATH_TEST = "data/chapter6/test.csv"
PATH_VECTORIZED_X_TRAIN = "data/chapter8/vectorized_X_train.npy"
PATH_VECTORIZED_Y_TRAIN = "data/chapter8/vectorized_Y_train.npy"
PATH_VECTORIZED_X_VALID = "data/chapter8/vectorized_X_valid.npy"
PATH_VECTORIZED_Y_VALID = "data/chapter8/vectorized_Y_valid.npy"
PATH_VECTORIZED_X_TEST = "data/chapter8/vectorized_X_test.npy"
PATH_VECTORIZED_Y_TEST = "data/chapter8/vectorized_Y_test.npy"
WEIGHT_PATH = "data/chapter8/weight.pth"
MODEL_PATH = "data/chapter8/model.pth"

model = KeyedVectors.load_word2vec_format(VEC_FILE_PATH, binary=True)
CATEGORY_MAP = {"b": 0, "t": 1, "e": 2, "m": 3}


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


def start_70():
    section("70. 単語ベクトルの和による特徴量")
    train = pd.read_csv(PATH_TRAIN)
    valid = pd.read_csv(PATH_VALID)
    test = pd.read_csv(PATH_TEST)

    def vectorize(text):
        words = text.split(" ")
        words = [clean_text(word) for word in words]
        vectors = [model[word] for word in words if word in model]
        mean_vector = np.mean(vectors, axis=0)
        print(mean_vector.shape)
        if mean_vector.shape != (300,):
            print(text)
            for word in words:
                if word not in model:
                    print(word)
            raise Exception("Invalid shape")
        return mean_vector

    X_train = np.array([vectorize(text) for text in train["TITLE"]])
    Y_train = np.array([CATEGORY_MAP[category] for category in train["CATEGORY"]])
    X_valid = np.array([vectorize(text) for text in valid["TITLE"]])
    Y_valid = np.array([CATEGORY_MAP[category] for category in valid["CATEGORY"]])
    X_test = np.array([vectorize(text) for text in test["TITLE"]])
    Y_test = np.array([CATEGORY_MAP[category] for category in test["CATEGORY"]])

    print(X_train.shape)
    print(Y_train.shape)
    print(X_valid.shape)
    print(Y_valid.shape)
    print(X_test.shape)
    print(Y_test.shape)

    np.save(PATH_VECTORIZED_X_TRAIN, X_train)
    np.save(PATH_VECTORIZED_Y_TRAIN, Y_train)
    np.save(PATH_VECTORIZED_X_VALID, X_valid)
    np.save(PATH_VECTORIZED_Y_VALID, Y_valid)
    np.save(PATH_VECTORIZED_X_TEST, X_test)
    np.save(PATH_VECTORIZED_Y_TEST, Y_test)


def start_71():
    section("71. 単層ニューラルネットワークによる予測")
    X_train = torch.tensor(np.load(PATH_VECTORIZED_X_TRAIN), dtype=torch.float32)

    # 300次元の入力(titleの単語ベクトル)を4次元(カテゴリ数)に変換する重み
    W = np.random.randn(300, 4)
    W = torch.tensor(W, dtype=torch.float32, requires_grad=True)

    print(W.shape)
    print(X_train.shape)
    print(X_train[0].shape)

    # X_1 = \text{title}_1 の単語ベクトル
    X_1 = X_train[0]
    # W^T \cdot X_1
    X_1_W = torch.mv(W.T, X_1)
    # \hat{y_1} = \text{softmax}(X_1, W)
    model = nn.Softmax(dim=0)
    output_train = model(X_1_W)

    # \hat{y_1} = \text{softmax}(x_1, W)
    # x_1 に対する予測値
    print(output_train)

    X_train_4 = torch.tensor(X_train[:4], dtype=torch.float32)

    # W^T \cdot X_{[1:4]}
    X_train_4_W = torch.mm(X_train_4, W)
    model = nn.Softmax(dim=1)
    output_train_4 = model(X_train_4_W)

    # \hat{Y} = \text{softmax}(X_{[1:4]}, W)
    # X_{[1:4]} に対する予測値
    print(output_train_4)


def start_72():
    section("72. 損失と勾配の計算")
    X_train = torch.tensor(np.load(PATH_VECTORIZED_X_TRAIN), dtype=torch.float32)
    Y_train = torch.tensor(np.load(PATH_VECTORIZED_Y_TRAIN), dtype=torch.int64)
    X_train_1 = X_train[0]
    Y_train_1 = Y_train[0]
    X_train_4 = X_train[:4]
    Y_train_4 = Y_train[:4]

    W_1 = np.random.randn(300, 4)
    W_1 = torch.tensor(W_1, dtype=torch.float32, requires_grad=True)
    W_4 = np.random.randn(300, 4)
    W_4 = torch.tensor(W_4, dtype=torch.float32, requires_grad=True)

    model_sm_dim_0 = nn.Softmax(dim=0)
    model_sm_dim_1 = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()

    # 事例x_1に対する損失
    output_train_1 = model_sm_dim_0(torch.mv(W_1.T, X_train_1))
    loss_train_1 = criterion(output_train_1, Y_train_1)
    print(loss_train_1)

    # 事例x_1に対する勾配
    W_1.grad = None
    loss_train_1.backward()
    print(W_1.grad)

    # 事例集合x_{[1:4]}に対する損失
    output_train_4 = model_sm_dim_1(torch.mm(X_train_4, W_4))
    loss_train_4 = criterion(output_train_4, Y_train_4)
    print(loss_train_4)

    # 事例集合x_{[1:4]}に対する勾配
    W_4.grad = None
    loss_train_4.backward()
    print(W_4.grad)


losses = []
accuracies = []


def start_73():
    section("73. 確率的勾配降下法による学習")
    X_train = torch.tensor(
        np.load(PATH_VECTORIZED_X_TRAIN), dtype=torch.float32, device="mps"
    )
    Y_train = torch.tensor(
        np.load(PATH_VECTORIZED_Y_TRAIN), dtype=torch.int64, device="mps"
    )
    X_valid = torch.tensor(
        np.load(PATH_VECTORIZED_X_VALID), dtype=torch.float32, device="mps"
    )
    Y_valid = torch.tensor(
        np.load(PATH_VECTORIZED_Y_VALID), dtype=torch.int64, device="mps"
    )

    model = nn.Linear(300, 4).to("mps")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 1000

    def accuracy(output, Y):
        _, predicted = torch.max(output, 1)
        return (predicted == Y).sum().item() / len(Y)

    for epoch in range(epochs):
        optimizer.zero_grad()
        # output_train = model(torch.mm(X_train, W))
        output_train = model(X_train)
        loss_train = criterion(output_train, Y_train)
        loss_train.backward()
        optimizer.step()

        # output_valid = model(torch.mm(X_valid, W))
        output_valid = model(X_valid)
        loss_valid = criterion(output_valid, Y_valid)
        losses.append((loss_train, loss_valid))
        accuracies.append(
            (accuracy(output_train, Y_train), accuracy(output_valid, Y_valid))
        )
        print(f"epoch: {epoch}, loss_train: {loss_train}, loss_valid: {loss_valid}")

    # torch.save(W, WEIGHT_PATH)
    torch.save(model, MODEL_PATH)


def start_74():
    section("74. 正解率の計測")
    # load model
    model = torch.load(MODEL_PATH)
    X_test = torch.tensor(
        np.load(PATH_VECTORIZED_X_TEST), dtype=torch.float32, device="mps"
    )
    Y_test = torch.tensor(
        np.load(PATH_VECTORIZED_Y_TEST), dtype=torch.int64, device="mps"
    )

    output_test = model(X_test)
    _, predicted = torch.max(output_test, 1)
    accuracy = (predicted == Y_test).sum().item() / len(Y_test)
    print(f"Accuracy: {accuracy}")


def start_75():
    section("75. 損失と正解率のプロット")
    plt.plot([loss[0].item() for loss in losses], label="train")
    plt.plot([loss[1].item() for loss in losses], label="valid")
    plt.legend()
    plt.savefig("data/chapter8/loss.png")
    plt.close()

    plt.plot([acc[0] for acc in accuracies], label="train")
    plt.plot([acc[1] for acc in accuracies], label="valid")
    plt.legend()
    plt.savefig("data/chapter8/accuracy.png")
    plt.close()


def start_76():
    section("76. チェックポイント")
    # 問題75のコードを改変し、各エポックのパラメータをファイルに保存するようにしてください。
    X_train = torch.tensor(
        np.load(PATH_VECTORIZED_X_TRAIN), dtype=torch.float32, device="mps"
    )
    Y_train = torch.tensor(
        np.load(PATH_VECTORIZED_Y_TRAIN), dtype=torch.int64, device="mps"
    )
    X_valid = torch.tensor(
        np.load(PATH_VECTORIZED_X_VALID), dtype=torch.float32, device="mps"
    )
    Y_valid = torch.tensor(
        np.load(PATH_VECTORIZED_Y_VALID), dtype=torch.int64, device="mps"
    )

    model = nn.Linear(300, 4).to("mps")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 1000

    for epoch in range(epochs):
        optimizer.zero_grad()
        output_train = model(X_train)
        loss_train = criterion(output_train, Y_train)
        loss_train.backward()
        optimizer.step()

        output_valid = model(X_valid)
        loss_valid = criterion(output_valid, Y_valid)
        losses.append((loss_train, loss_valid))
        print(f"epoch: {epoch}, loss_train: {loss_train}, loss_valid: {loss_valid}")

        if epoch % 100 == 0:
            torch.save(model, f"data/chapter8/model_{epoch}.pth")
            torch.save(optimizer, f"data/chapter8/optimizer_{epoch}.pth")


from torch.utils.data import DataLoader, TensorDataset

losses_77 = []
accuracies_77 = []


def start_77():
    section("77. ミニバッチ化")
    X_train = torch.tensor(
        np.load(PATH_VECTORIZED_X_TRAIN), dtype=torch.float32, device="mps"
    )
    Y_train = torch.tensor(
        np.load(PATH_VECTORIZED_Y_TRAIN), dtype=torch.int64, device="mps"
    )
    X_valid = torch.tensor(
        np.load(PATH_VECTORIZED_X_VALID), dtype=torch.float32, device="mps"
    )
    Y_valid = torch.tensor(
        np.load(PATH_VECTORIZED_Y_VALID), dtype=torch.int64, device="mps"
    )

    model = nn.Linear(300, 4).to("mps")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 100
    batch_size = 32

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def accuracy(output, Y):
        _, predicted = torch.max(output, 1)
        return (predicted == Y).sum().item() / len(Y)

    for epoch in range(epochs):
        min_loss_train = 1e9
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            output_train = model(X_batch)
            loss_train = criterion(output_train, Y_batch)
            loss_train.backward()
            optimizer.step()
            min_loss_train = min(min_loss_train, loss_train.item())

        output_valid = model(X_valid)
        loss_valid = criterion(output_valid, Y_valid).item()
        losses_77.append((min_loss_train, loss_valid))
        accuracy_train = accuracy(output_train, Y_batch)
        accuracy_valid = accuracy(output_valid, Y_valid)
        accuracies_77.append((accuracy_train, accuracy_valid))
        print(f"epoch: {epoch}, loss_train: {min_loss_train}, loss_valid: {loss_valid}")

    plt.plot([loss[0] for loss in losses_77], label="train")
    plt.plot([loss[1] for loss in losses_77], label="valid")
    plt.legend()
    plt.savefig("data/chapter8/loss_77.png")
    plt.close()

    plt.plot([acc[0] for acc in accuracies_77], label="train")
    plt.plot([acc[1] for acc in accuracies_77], label="valid")
    plt.legend()
    plt.savefig("data/chapter8/accuracy_77.png")
    plt.close()

    output_valid = model(X_valid)
    acc = accuracy(output_valid, Y_valid)
    print(f"Accuracy: {acc}")
    torch.save(model, "data/chapter8/model_77.pth")
    torch.save(optimizer, "data/chapter8/optimizer_77.pth")


def start_78():
    section("78. GPU上での学習")
    pass  # 77のコードでGPU上で学習している


def start_79():
    section("79. 多層ニューラルネットワーク")
    X_train = torch.tensor(
        np.load(PATH_VECTORIZED_X_TRAIN), dtype=torch.float32, device="mps"
    )
    Y_train = torch.tensor(
        np.load(PATH_VECTORIZED_Y_TRAIN), dtype=torch.int64, device="mps"
    )
    X_valid = torch.tensor(
        np.load(PATH_VECTORIZED_X_VALID), dtype=torch.float32, device="mps"
    )
    Y_valid = torch.tensor(
        np.load(PATH_VECTORIZED_Y_VALID), dtype=torch.int64, device="mps"
    )

    model = nn.Sequential(
        nn.Linear(300, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 4),
    ).to("mps")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 100
    batch_size = 32

    losses_79 = []
    accuracies_79 = []

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def accuracy(output, Y):
        _, predicted = torch.max(output, 1)
        return (predicted == Y).sum().item() / len(Y)

    for epoch in range(epochs):
        min_loss_train = 1e9
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            output_train = model(X_batch)
            loss_train = criterion(output_train, Y_batch)
            loss_train.backward()
            optimizer.step()
            min_loss_train = min(min_loss_train, loss_train.item())

        output_valid = model(X_valid)
        loss_valid = criterion(output_valid, Y_valid).item()
        losses_79.append((min_loss_train, loss_valid))
        accuracy_train = accuracy(output_train, Y_batch)
        accuracy_valid = accuracy(output_valid, Y_valid)
        accuracies_79.append((accuracy_train, accuracy_valid))
        print(f"epoch: {epoch}, loss_train: {min_loss_train}, loss_valid: {loss_valid}")

    plt.plot([loss[0] for loss in losses_79], label="train")
    plt.plot([loss[1] for loss in losses_79], label="valid")
    plt.legend()
    plt.savefig("data/chapter8/loss_79.png")
    plt.close()

    plt.plot([acc[0] for acc in accuracies_79], label="train")
    plt.plot([acc[1] for acc in accuracies_79], label="valid")
    plt.legend()
    plt.savefig("data/chapter8/accuracy_79.png")
    plt.close()

    output_valid = model(X_valid)
    acc = accuracy(output_valid, Y_valid)
    print(f"Accuracy: {acc}")
    torch.save(model, "data/chapter8/model_79.pth")
    torch.save(optimizer, "data/chapter8/optimizer_79.pth")


def start():
    chapter("第8章: ニューラルネット")
    start_70()
    start_71()
    start_72()
    start_73()
    start_74()
    start_75()
    start_76()
    start_77()
    start_78()
    start_79()
