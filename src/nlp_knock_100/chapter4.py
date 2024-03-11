import os

from .util import chapter, section
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression


words = []

def start_30():
    # memo: `mecab path/to/text -o path/to/output` で形態素解析結果を出力
    section("30. 形態素解析結果の読み込み")

    with open("data/chapter4/neko.txt.mecab") as f:
        lines = f.readlines()
        for line in lines:
            if line == "EOS\n":
                continue
            word, feature = line.split("\t")
            features = feature.split(",")
            words.append({
                "surface": word,
                "base": features[6],
                "pos": features[0],
                "pos1": features[1]
            })

    print(words[:10])


def start_31():
    global words
    if not words:
        start_30()
    section("31. 動詞")
    LIMIT = 5  # suppress long output
    count = 0
    for word in words:
        if word["pos"] == "動詞":
            count += 1
            print(word["surface"])
        if count >= LIMIT:
            break


def start_32():
    global words
    if not words:
        start_30()
    section("32. 動詞の原形")
    LIMIT = 5  # suppress long output
    count = 0
    for word in words:
        if word["pos"] == "動詞":
            count += 1
            print(word["base"])
        if count >= LIMIT:
            break

def start_33():
    global words
    if not words:
        start_30()
    section("33. 「AのB」")
    LIMIT = 5  # suppress long output
    count = 0
    for i in range(len(words) - 2):
        if words[i]["surface"] == "の" and words[i - 1]["pos"] == "名詞" and words[i + 1]["pos"] == "名詞":
            count += 1
            print(words[i - 1]["surface"], words[i]["surface"], words[i + 1]["surface"])
        if count >= LIMIT:
            break


def start_34():
    global words
    if not words:
        start_30()
    section("34. 名詞の連接")
    LIMIT = 5  # suppress long output
    count = 0
    nouns = []
    for word in words:
        if word["pos"] == "名詞":
            nouns.append(word["surface"])
        else:
            if len(nouns) > 1:
                count += 1
                print("".join(nouns))
            nouns = []
        if count >= LIMIT:
            break


def start_35():
    global words
    if not words:
        start_30()
    section("35. 単語の出現頻度")
    LIMIT = 5  # suppress long output
    count = 0
    word_count = {}
    for word in words:
        if word["surface"] in word_count:
            if word["pos"] != "記号":
                word_count[word["surface"]] += 1
        else:
            word_count[word["surface"]] = 1
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    for word, count_of_word in word_count:
        count += 1
        print(word, count_of_word)
        if count >= LIMIT:
            break


def start_36():
    global words
    if not words:
        start_30()
    section("36. 頻度上位10語")
    LIMIT = 10  # suppress long output
    word_count = {}
    for word in words:
        if word["surface"] in word_count:
            if word["pos"] != "記号":
                word_count[word["surface"]] += 1
        else:
            word_count[word["surface"]] = 1
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    if not os.path.exists("data/chapter4/36.png"):
        word_list, count_list = zip(*word_count[:LIMIT])
        plt.bar(word_list, count_list)
        plt.title("頻度上位10語")
        plt.xlabel("単語")
        plt.ylabel("出現頻度")
        plt.savefig("data/chapter4/36.png")
        print("Saved as data/chapter4/36.png")
    else:
        print("data/chapter4/36.png already exists")


def start_37():
    global words
    if not words:
        start_30()
    section("37. 「猫」と共起頻度の高い上位10語")
    # 共起とは、ある単語が出現したときに、その前後にどのような単語が出現するかということ
    LIMIT = 10  # suppress long output
    word_count = {}
    for i in range(len(words) - 1):
        if words[i]["surface"] == "猫":
            if words[i + 1]["surface"] in word_count:
                if words[i + 1]["pos"] != "記号":
                    word_count[words[i + 1]["surface"]] += 1
            else:
                word_count[words[i + 1]["surface"]] = 1
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    if not os.path.exists("data/chapter4/37.png"):
        word_list, count_list = zip(*word_count[:LIMIT])
        plt.bar(word_list, count_list)
        plt.title("「猫」と共起頻度の高い上位10語")
        plt.xlabel("単語")
        plt.ylabel("出現頻度")
        plt.savefig("data/chapter4/37.png")
        print("Saved as data/chapter4/37.png")
    else:
        print("data/chapter4/37.png already exists")


def start_38():
    global words
    if not words:
        start_30()
    section("38. ヒストグラム")
    word_count = {}
    for word in words:
        if word["surface"] in word_count:
            if word["pos"] != "記号":
                word_count[word["surface"]] += 1
        else:
            word_count[word["surface"]] = 1
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    if not os.path.exists("data/chapter4/38.png"):
        count_list = [count for word, count in word_count]
        plt.hist(count_list, bins=20, range=(1, 20))
        plt.title("ヒストグラム")
        plt.xlabel("出現頻度")
        plt.ylabel("単語の種類数")
        plt.savefig("data/chapter4/38.png")
        print("Saved as data/chapter4/38.png")
    else:
        print("data/chapter4/38.png already exists")

def start_39():
    global words
    if not words:
        start_30()
    section("39. Zipfの法則")
    word_count = {}
    for word in words:
        if word["surface"] in word_count:
            if word["pos"] != "記号":
                word_count[word["surface"]] += 1
        else:
            word_count[word["surface"]] = 1
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    count_list = [count for word, count in word_count]
    if not os.path.exists("data/chapter4/39.png"):
        plt.plot(count_list)
        count_list = np.array(count_list)
        X = np.log(np.arange(1, len(count_list) + 1))
        Y = np.log(count_list)
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), Y)
        a = model.coef_[0]
        b = model.intercept_
        plt.plot(np.exp(X), np.exp(a * X + b), "r")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Zipfの法則")
        plt.xlabel("出現頻度順位")
        plt.ylabel("出現頻度")
        plt.text(1e1, 1e4, f"y = {a:.2f}x + {b:.2f}", fontsize=10)
        plt.savefig("data/chapter4/39.png")
        print("Saved as data/chapter4/39.png")
    else:
        print("data/chapter4/39.png already exists")

def start():
    chapter("第4章: 形態素解析")
    start_30()
    start_31()
    start_32()
    start_33()
    start_34()
    start_35()
    start_36()
    start_37()
    start_38()
    start_39()
