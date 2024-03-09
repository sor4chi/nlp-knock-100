from .util import section, chapter
import random


def start_00():
    section("00. 文字列の逆順")
    s = "stressed"
    print(s[::-1])


def start_01():
    section("01. 「パタトクカシーー」")
    s = "パタトクカシーー"
    print(s[::2])


def start_02():
    section("02. 「パトカー」＋「タクシー」＝「パタトクカシーー」")
    s1 = "パトカー"
    s2 = "タクシー"
    print("".join([a + b for a, b in zip(s1, s2)]))


def start_03():
    section("03. 円周率")
    s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    tokens = s.replace(",", "").replace(".", "").split(" ")
    print([len(token) for token in tokens])


def start_04():
    section("04. 元素記号")
    s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    tokens = s.replace(",", "").replace(".", "").split(" ")
    idxs = [1, 5, 6, 7, 8, 9, 15, 16, 19]
    result = {}
    for i, token in enumerate(tokens, 1):
        result[token[:1] if i in idxs else token[:2]] = i
    print(result)


def n_gram(s, n):
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def start_05():
    section("05. n-gram")

    s = "I am an NLPer"
    print(n_gram(s, 2))  # 文字bi-gram
    print(n_gram(s.split(" "), 2))  # 単語bi-gram


def start_06():
    section("06. 集合")

    s1 = "paraparaparadise"
    s2 = "paragraph"
    X = set(n_gram(s1, 2))
    Y = set(n_gram(s2, 2))

    print(X | Y)  # 和集合
    print(X & Y)  # 積集合
    print(X - Y)  # 差集合
    print("se" in X)  # Xに"se"が含まれるか
    print("se" in Y)  # Yに"se"が含まれるか


def start_07():
    section("07. テンプレートによる文生成")

    def template(x, y, z):
        return f"{x}時の{y}は{z}"

    print(template(12, "気温", 22.4))


def start_08():
    section("08. 暗号文")

    def chiper(s: str, key: int) -> str:
        encrypted = ""
        for c in s:
            if ord("a") <= ord(c) <= ord("z"):
                encrypted += chr(key - ord(c))
            else:
                encrypted += c
        return encrypted

    KEY = 219
    s = "Hello, World!"
    encrypted = chiper(s, KEY)
    print(encrypted)
    decrypted = chiper(encrypted, KEY)
    print(decrypted)


def start_09():
    section("09. Typoglycemia")
    s = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    tokens = s.split(" ")
    result = []
    for token in tokens:
        if len(token) <= 4:
            result.append(token)
        else:
            head, *body, tail = token
            random.shuffle(body)
            result.append(head + "".join(body) + tail)
    print(" ".join(result))


def start():
    chapter("第1章: 準備運動")
    start_00()
    start_01()
    start_02()
    start_03()
    start_04()
    start_05()
    start_06()
    start_07()
    start_08()
    start_09()
