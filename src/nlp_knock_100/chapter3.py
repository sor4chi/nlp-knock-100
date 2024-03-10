from .util import section, chapter
import json
import re
import requests

body = ""


def get_body():
    global body
    if body == "":
        raise Exception("body is empty, run start_20 first")
    return body


def start_20():
    global body
    section("20. JSONデータの読み込み")
    LIMIT = 5  # suppress long output
    with open("data/chapter3/jawiki-country.json") as f:
        for line in f:
            data = json.loads(line)
            if data["title"] == "イギリス":
                body = data["text"]
                print("\n".join(body.split("\n")[:LIMIT]))
                print("...")
                break


def start_21():
    section("21. カテゴリ名を含む行を抽出")
    body = get_body()
    for line in body.split("\n"):
        if "Category" in line:
            print(line)


def start_22():
    section("22. カテゴリ名の抽出")
    body = get_body()
    regex = re.compile(r"\[\[Category:(.+?)(?:\|.*)?\]\]")
    for line in body.split("\n"):
        match = regex.search(line)
        if match:
            print(match.group(1))


def start_23():
    section("23. セクション構造")
    body = get_body()
    regex = re.compile(r"(==+)([^=]+)\1")
    for line in body.split("\n"):
        match = regex.search(line)
        if match:
            print(f"{match.group(2)}: {len(match.group(1)) - 1}")


def start_24():
    section("24. ファイル参照の抽出")
    body = get_body()
    regex = re.compile(r"\[\[ファイル:(.+?)(?:\|.*)?\]\]")
    for line in body.split("\n"):
        match = regex.search(line)
        if match:
            print(match.group(1))

template_info = {}

def start_25():
    global template_info
    section("25. テンプレートの抽出")
    body = get_body()
    regex = re.compile(r"{{基礎情報(.+?)\n}}", re.DOTALL)
    match = regex.search(body)
    if match:
        info = match.group(1)
        regex = re.compile(r"\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))", re.DOTALL)
        for match in regex.finditer(info):
            template_info[match.group(1)] = match.group(2)
    LIMIT = 5  # suppress long output
    cnt = 0
    for k, v in template_info.items():
        if cnt >= LIMIT:
            print("...")
            break
        cnt += 1
        print(f"{k}: {v}")


def start_26():
    global template_info
    if not template_info:
        start_25()
    section("26. 強調マークアップの除去")
    regex = re.compile(r"('{2,5})(.+?)\1")
    for k, v in template_info.items():
        if re.search(regex, v):
            print(f"{k}: {v}")
        template_info[k] = re.sub(regex, r"\2", v)  # ''emphasis'' -> emphasis
    LIMIT = 5  # suppress long output
    cnt = 0
    for k, v in template_info.items():
        if cnt >= LIMIT:
            print("...")
            break
        cnt += 1
        print(f"{k}: {v}")

def start_27():
    global template_info
    if not template_info:
        start_26()
    section("27. 内部リンクの除去")
    regex = re.compile(r"\[\[([^|]+?)\]\]")
    for k, v in template_info.items():
        template_info[k] = re.sub(regex, r"\1", v)
    regex = re.compile(r"\[\[.+?\|(.+?)\]\]")
    for k, v in template_info.items():
        s = re.sub(regex, r"\1", v)
        template_info[k] = s.split("|")[-1]

    LIMIT = 5  # suppress long output
    cnt = 0
    for k, v in template_info.items():
        if cnt >= LIMIT:
            print("...")
            break
        cnt += 1
        print(f"{k}: {v}")


def start_28():
    global template_info
    if not template_info:
        start_27()
    section("28. MediaWikiマークアップの除去")
    regex = re.compile(r"<.+?>")
    for k, v in template_info.items():
        template_info[k] = re.sub(regex, "", v)
    LIMIT = 5  # suppress long output
    cnt = 0
    for k, v in template_info.items():
        if cnt >= LIMIT:
            print("...")
            break
        cnt += 1
        print(f"{k}: {v}")


def start_29():
    section("29. 国旗画像のURLを取得する")
    media_wiki_api = "https://www.mediawiki.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{template_info['国旗画像']}",
        "iiprop": "url"
    }
    res = requests.get(media_wiki_api, params=params)
    data = res.json()
    pages = data["query"]["pages"]
    for _, page in pages.items():
        print(page["imageinfo"][0]["url"])


def start():
    chapter("第3章: 正規表現")
    start_20()
    start_21()
    start_22()
    start_23()
    start_24()
    start_25()
    start_26()
    start_27()
    start_28()
    # start_29() 毎回API叩くのはよくないのでコメントアウト
