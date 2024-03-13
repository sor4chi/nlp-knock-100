import os

from .util import chapter, section
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors
import nltk
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

PATH_DATA = "data/chapter6/newsCorpora.csv"
PATH_TRAIN = "data/chapter6/train.csv"
PATH_VALID = "data/chapter6/valid.csv"
PATH_TEST = "data/chapter6/test.csv"
PATH_TRAIN_FEAT = "data/chapter6/train.feature.txt"
PATH_VALID_FEAT = "data/chapter6/valid.feature.txt"
PATH_TEST_FEAT = "data/chapter6/test.feature.txt"

# download nltk data
nltk.download("punkt")


def start_50():
    section("50. データの入手・整形")
    df = pd.read_csv(PATH_DATA, sep="\t", header=None)
    df.columns = [
        "ID",
        "TITLE",
        "URL",
        "PUBLISHER",
        "CATEGORY",
        "STORY",
        "HOSTNAME",
        "TIMESTAMP",
    ]
    extract_publishers = [
        "Reuters",
        "Huffington Post",
        "Businessweek",
        "Contactmusic.com",
        "Daily Mail",
    ]
    df = df[df["PUBLISHER"].isin(extract_publishers)]
    train, valid_test = train_test_split(df, test_size=0.2, random_state=42)
    valid, test = train_test_split(valid_test, test_size=0.5, random_state=42)
    train.to_csv(PATH_TRAIN, index=False)
    valid.to_csv(PATH_VALID, index=False)
    test.to_csv(PATH_TEST, index=False)
    print(f"train: {len(train)}")
    print(f"valid: {len(valid)}")
    print(f"test: {len(test)}")


def start_51():
    section("51. 特徴量抽出")

    train = pd.read_csv(PATH_TRAIN)
    valid = pd.read_csv(PATH_VALID)
    test = pd.read_csv(PATH_TEST)

    def extract_feature(df, path):
        df = df[["CATEGORY", "TITLE"]]
        df.to_csv(path, sep="\t", index=False, header=False)

    extract_feature(train, PATH_TRAIN_FEAT)
    extract_feature(valid, PATH_VALID_FEAT)
    extract_feature(test, PATH_TEST_FEAT)


model = None
label_encoder = None
X_train = None
y_train = None
X_valid = None
y_valid = None
X_test = None
y_test = None
column_names = None


def start_52():
    global \
        model, \
        label_encoder, \
        X_train, \
        y_train, \
        X_valid, \
        y_valid, \
        X_test, \
        y_test, \
        column_names
    section("52. 学習")

    train = pd.read_csv(PATH_TRAIN_FEAT, sep="\t", header=None)
    valid = pd.read_csv(PATH_VALID_FEAT, sep="\t", header=None)
    test = pd.read_csv(PATH_TEST_FEAT, sep="\t", header=None)

    X_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]
    X_valid = valid.iloc[:, 1:]
    y_valid = valid.iloc[:, 0]
    X_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]

    # titleベクトル化,それ以外はカテゴリカル変数としてOneHotEncoderで変換
    method = "CountVectorizer"
    vectorized_X_train_title_df = None
    vectorized_X_valid_title_df = None
    vectorized_X_test_title_df = None
    if method == "CountVectorizer":
        vectorizer = CountVectorizer()
        X_train_title = vectorizer.fit_transform(X_train.iloc[:, 0])
        X_valid_title = vectorizer.transform(X_valid.iloc[:, 0])
        X_test_title = vectorizer.transform(X_test.iloc[:, 0])
        vectorized_X_train_title_df = pd.DataFrame(
            X_train_title.toarray(), columns=vectorizer.get_feature_names_out()
        )
        vectorized_X_valid_title_df = pd.DataFrame(
            X_valid_title.toarray(), columns=vectorizer.get_feature_names_out()
        )
        vectorized_X_test_title_df = pd.DataFrame(
            X_test_title.toarray(), columns=vectorizer.get_feature_names_out()
        )
    if method == "TfidfVectorizer":
        vectorizer = TfidfVectorizer()
        X_train_title = vectorizer.fit_transform(X_train.iloc[:, 0])
        X_valid_title = vectorizer.transform(X_valid.iloc[:, 0])
        X_test_title = vectorizer.transform(X_test.iloc[:, 0])
        vectorized_X_train_title_df = pd.DataFrame(
            X_train_title.toarray(), columns=vectorizer.get_feature_names_out()
        )
        vectorized_X_valid_title_df = pd.DataFrame(
            X_valid_title.toarray(), columns=vectorizer.get_feature_names_out()
        )
        vectorized_X_test_title_df = pd.DataFrame(
            X_test_title.toarray(), columns=vectorizer.get_feature_names_out()
        )
    if method == "Word2Vec":
        word2vec = KeyedVectors.load_word2vec_format(
            "data/chapter6/GoogleNews-vectors-negative300.bin", binary=True
        )

        def get_vector(text: str) -> np.ndarray:
            words = nltk.word_tokenize(text)
            res = np.mean(
                [word2vec[word] for word in words if word in word2vec], axis=0
            )
            return res

        X_train_title = pd.DataFrame([get_vector(text) for text in X_train.iloc[:, 0]])
        X_valid_title = pd.DataFrame([get_vector(text) for text in X_valid.iloc[:, 0]])
        X_test_title = pd.DataFrame([get_vector(text) for text in X_test.iloc[:, 0]])
        length_of_vector = len(X_train_title.columns)
        vectorized_X_train_title_df = pd.DataFrame(
            X_train_title, columns=[f"vec{i}" for i in range(length_of_vector)]
        )
        vectorized_X_valid_title_df = pd.DataFrame(
            X_valid_title, columns=[f"vec{i}" for i in range(length_of_vector)]
        )
        vectorized_X_test_title_df = pd.DataFrame(
            X_test_title, columns=[f"vec{i}" for i in range(length_of_vector)]
        )

    X_train_categorical = X_train.iloc[:, 1:]
    X_valid_categorical = X_valid.iloc[:, 1:]
    X_test_categorical = X_test.iloc[:, 1:]
    encoder = OneHotEncoder()
    X_train_categorical = encoder.fit_transform(X_train_categorical).toarray()
    X_valid_categorical = encoder.transform(X_valid_categorical).toarray()
    X_test_categorical = encoder.transform(X_test_categorical).toarray()
    X_train_categorical_df = pd.DataFrame(
        X_train_categorical, columns=encoder.get_feature_names_out()
    )
    X_valid_categorical_df = pd.DataFrame(
        X_valid_categorical, columns=encoder.get_feature_names_out()
    )
    X_test_categorical_df = pd.DataFrame(
        X_test_categorical, columns=encoder.get_feature_names_out()
    )

    X_train = pd.concat([vectorized_X_train_title_df, X_train_categorical_df], axis=1)
    X_valid = pd.concat([vectorized_X_valid_title_df, X_valid_categorical_df], axis=1)
    X_test = pd.concat([vectorized_X_test_title_df, X_test_categorical_df], axis=1)

    # 目的変数を数値に変換
    labels = ["b", "e", "t", "m"]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_train = label_encoder.fit_transform(y_train)
    y_valid = label_encoder.transform(y_valid)
    y_test = label_encoder.transform(y_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    column_names = X_train.columns


Y_train_pred = None
Y_valid_pred = None
Y_test_pred = None


def start_53():
    global model, X_train, X_valid, X_test, Y_train_pred, Y_valid_pred, Y_test_pred
    if any([model, X_train, X_valid, X_test]) is False:
        start_52()

    section("53. 予測")
    Y_train_pred = model.predict(X_train)
    Y_valid_pred = model.predict(X_valid)
    Y_test_pred = model.predict(X_test)


def start_54():
    global Y_train_pred, Y_valid_pred, Y_test_pred
    if any([Y_train_pred is None, Y_valid_pred is None, Y_test_pred is None]):
        start_53()
    section("54. 正解率の計測")

    print(f"train: {accuracy_score(y_train, Y_train_pred)}")
    print(f"valid: {accuracy_score(y_valid, Y_valid_pred)}")
    print(f"test: {accuracy_score(y_test, Y_test_pred)}")
    """
    - CountVectorizer
    train: 0.9959707646176912
    valid: 0.9062968515742129
    test: 0.9145427286356822

    - TfidfVectorizer
    train: 0.9434032983508246
    valid: 0.8875562218890555
    test: 0.8755622188905547

    - Word2Vec
    train: 0.9189467766116941
    valid: 0.8928035982008995
    test: 0.8920539730134932
    """


def start_55():
    section("55. 混同行列の作成")
    print(f"train:\n{confusion_matrix(y_train, Y_train_pred)}")
    print(f"valid:\n{confusion_matrix(y_valid, Y_valid_pred)}")
    print(f"test:\n{confusion_matrix(y_test, Y_test_pred)}")

    def make_heatmap(matrix, title):
        if not os.path.exists(f"data/chapter6/55_{title}.png"):
            sns.heatmap(matrix, annot=True, cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(title)
            plt.savefig(f"data/chapter6/55_{title}.png")
            plt.clf()
            print(f"Saved as data/chapter6/55_{title}.png")
        else:
            print(f"data/chapter6/55_{title}.png already exists")

    # ヒートマップを作っておく
    make_heatmap(confusion_matrix(y_train, Y_train_pred), "train")
    make_heatmap(confusion_matrix(y_valid, Y_valid_pred), "valid")
    make_heatmap(confusion_matrix(y_test, Y_test_pred), "test")


def start_56():
    section("56. 適合率, 再現率, F1スコアの計測")
    print("precision:")
    print(f"train: {precision_score(y_train, Y_train_pred, average=None)}")
    print(f"valid: {precision_score(y_valid, Y_valid_pred, average=None)}")
    print(f"test: {precision_score(y_test, Y_test_pred, average=None)}")
    print("recall:")
    print(f"train: {recall_score(y_train, Y_train_pred, average=None)}")
    print(f"valid: {recall_score(y_valid, Y_valid_pred, average=None)}")
    print(f"test: {recall_score(y_test, Y_test_pred, average=None)}")
    print("f1:")
    print(f"train: {f1_score(y_train, Y_train_pred, average=None)}")
    print(f"valid: {f1_score(y_valid, Y_valid_pred, average=None)}")
    print(f"test: {f1_score(y_test, Y_test_pred, average=None)}")


def start_57():
    section("57. 特徴量の重みの確認")
    print("feature importance:")
    print("10 most important features")
    sort = np.argsort(np.abs(model.coef_), axis=1)[:, ::-1]
    for i, label in enumerate(label_encoder.classes_):
        print(f"\n{label}:")
        for j in sort[i, :10]:
            print(f"{column_names[j]}: {model.coef_[i, j]}")

    """
    feature importance:
    10 most important features

    b:
    bank: 1.8988593149670525
    fed: 1.7849745950159583
    ecb: 1.7784003707484959
    obamacare: 1.6696241861572094
    china: 1.6382758238497697
    activision: -1.6364587355359113
    oil: 1.6205406628415915
    yellen: 1.5588823011137614
    ukraine: 1.5529078107426462
    euro: 1.4971287172253054

    e:
    google: -1.638663105129022
    kardashian: 1.6181600386740227
    chris: 1.4975069923696211
    transformers: 1.3615462182822335
    paul: 1.345797217295505
    film: 1.307695934225445
    thrones: 1.304316479574884
    jay: 1.2923757496632033
    gm: -1.2916348498496562
    beyonce: 1.2728669756585145

    m:
    ebola: 2.5889371929450817
    cancer: 2.1662885376090832
    fda: 2.0465964553407843
    mers: 1.888260882202835
    cases: 1.8610153372125589
    drug: 1.8178848814492912
    study: 1.7333848526267548
    cdc: 1.560275638022442
    medical: 1.490495354676093
    doctors: 1.4328117988017797

    t:
    google: 2.754755544364861
    facebook: 2.683246019216671
    microsoft: 2.4133313250330213
    apple: 2.2736170534639535
    climate: 2.2227084412508633
    nasa: 1.9955322551551309
    activision: 1.834205258614424
    tesla: 1.8131779505911376
    gm: 1.7848164562667022
    heartbleed: 1.7708137959802435
"""


def start_58():
    section("58. 正則化パラメータの変更")

    def train_and_predict(C: float):
        model = LogisticRegression(C=C)
        model.fit(X_train, y_train)
        Y_test_pred = model.predict(X_test)
        return accuracy_score(y_test, Y_test_pred)

    if not os.path.exists("data/chapter6/58.png"):
        predicts = []
        c_values = [10**i for i in range(-3, 4)]
        for c in c_values:
            print(f"predicting with C={c}")
            predicts.append(train_and_predict(c))
        print(predicts)
        plt.plot(c_values, predicts)
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("accuracy")
        plt.title("accuracy by C")
        plt.savefig("data/chapter6/58.png")
        plt.clf()
        print("Saved as data/chapter6/58.png")
    else:
        print("data/chapter6/58.png already exists")


def start_59():
    section("59. ハイパーパラメータの探索")
    return  # 時間がかかるので動かさない

    def objective(trial):
        # 色々なパラメータを試す
        params = {
            "C": trial.suggest_loguniform("C", 1e-3, 1e3),
            "solver": trial.suggest_categorical(
                "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            ),
            "random_state": 42,
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        Y_valid_pred = model.predict(X_valid)
        return accuracy_score(y_valid, Y_valid_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)


def start():
    chapter("第6章: 機械学習")
    start_50()
    start_51()
    start_52()
    start_53()
    start_54()
    start_55()
    start_56()
    start_57()
    start_58()
    start_59()
