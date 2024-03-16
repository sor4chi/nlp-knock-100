import os

from .util import chapter, section
from gensim.models import KeyedVectors, Word2Vec
import scipy.stats
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

VEC_FILE_PATH = "data/chapter7/GoogleNews-vectors-negative300.bin"

model = KeyedVectors.load_word2vec_format(VEC_FILE_PATH, binary=True)


def start_60():
    section("60. 単語ベクトルの読み込みと表示")
    print(model["United_States"])


def start_61():
    section("61. 単語の類似度")
    print(model.similarity("United_States", "U.S."))


def start_62():
    section("62. 類似度の高い単語10件")
    simular_words = model.most_similar("United_States", topn=10)
    for word, sim in simular_words:
        print(f"{word}: {sim}")


def start_63():
    section("63. 加法構成性によるアナロジー")
    simular_words = model.most_similar(
        positive=["Spain", "Athens"], negative=["Madrid"], topn=10
    )
    for word, sim in simular_words:
        print(f"{word}: {sim}")


EVALUATION_DATA_PATH = "data/chapter7/questions-words.txt"


def start_64():
    section("64. アナロジーデータでの実験")
    return  # Skip this test because it takes too long
    data = []
    with open(EVALUATION_DATA_PATH, "r") as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    data = [line for line in data if line[0] != ":"]
    data = [line.split() for line in data]
    data = [line for line in data if len(line) == 4]
    correct_count = 0
    for d in data:
        try:
            simular_words = model.most_similar(
                positive=[d[1], d[2]], negative=[d[0]], topn=1
            )
            if simular_words[0][0] == d[3]:
                correct_count += 1
        except:
            pass
    print(f"正解率: {correct_count / len(data)}")
    """
    正解率: 0.7358780188293083
    """


def start_65():
    global analogies_simular_words
    section("65. アナロジータスクでの正解率")
    return  # Skip this test because it takes too long
    # 64でやってるのでスキップ


WORD_SIMILARITY_353_PATH = "data/chapter7/wordsim353/combined.csv"


def start_66():
    section("66. WordSimilarity-353での評価")

    data = []
    with open(WORD_SIMILARITY_353_PATH, "r") as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    data = [line.split(",") for line in data]
    data = [line for line in data if len(line) == 3]
    data = data[1:]
    human_judgement = [float(d[2]) for d in data]
    model_judgement = [
        model.similarity(d[0], d[1]) if d[0] in model and d[1] in model else 0
        for d in data
    ]
    print(scipy.stats.spearmanr(human_judgement, model_judgement))
    """
    SpearmanrResult(correlation=0.7000166486272194, pvalue=1.0255192493763449e-53)
    """

    # スピアマン相関係数は順位相関係数の一種であり、2つの変数それぞれを昇順に並べたときの順位の相関係数を示す。


WORLD_CSV_PATH = "data/chapter7/countries.csv"


def start_67():
    section("67._k-meansクラスタリング")
    # 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．
    # 国名に関する単語ベクトルを抽出
    countries = []
    with open(WORLD_CSV_PATH, "r") as f:
        countries = f.readlines()
    countries = [line.strip() for line in countries]

    country_vectors = [model[country] for country in countries if country in model]
    # k-meansクラスタリングをクラスタ数k=5として実行
    N_CLUSTERS = 5
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(country_vectors)
    labels = kmeans.labels_
    for label, country in zip(labels, countries):
        print(f"{label}: {country}")


def start_68():
    section("68. Ward法によるクラスタリング")
    # 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行し，デンドログラムを描画せよ．
    countries = []
    with open(WORLD_CSV_PATH, "r") as f:
        countries = f.readlines()
    countries = [line.strip() for line in countries]
    filtered_countries = [country for country in countries if country in model]
    country_vectors = [model[country] for country in filtered_countries]
    # Ward法による階層型クラスタリングを実行
    sch.dendrogram(
        sch.linkage(country_vectors, method="ward"), labels=filtered_countries
    )
    plt.savefig("data/chapter7/ward.png")
    plt.close()


def start_69():
    section("69. t-SNEによる可視化")
    # データをt-SNEで可視化
    countries = []
    with open(WORLD_CSV_PATH, "r") as f:
        countries = f.readlines()
    countries = [line.strip() for line in countries]
    filtered_countries = [country for country in countries if country in model]
    country_vectors = [model[country] for country in filtered_countries]
    country_vectors = np.array(country_vectors)
    tsne = TSNE(n_components=2, random_state=0)
    country_vectors_tsne = tsne.fit_transform(country_vectors)
    for country, vec in zip(filtered_countries, country_vectors_tsne):
        plt.scatter(vec[0], vec[1])
        plt.annotate(country, (vec[0], vec[1]))
    plt.savefig("data/chapter7/tsne.png")


def start():
    chapter("第7章: 単語ベクトル")
    start_60()
    start_61()
    start_62()
    start_63()
    start_64()
    start_65()
    start_66()
    start_67()
    start_68()
    start_69()
