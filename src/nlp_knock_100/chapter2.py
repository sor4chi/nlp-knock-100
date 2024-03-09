from .util import section, chapter
import os
from subprocess import check_output


def start_10():
    section("10. 行数のカウント")
    os.system("wc -l data/chapter2/popular-names.txt")


def start_11():
    section("11. タブをスペースに置換")
    cmd = "cat data/chapter2/popular-names.txt | tr '\t' ' '"
    stdout = check_output(cmd, shell=True).decode()
    LIMIT = 5  # suppress long output
    print("\n".join(stdout.split("\n")[:LIMIT]))


def start_12():
    section("12. 1列目をcol1.txtに,2列目をcol2.txtに保存")
    cmd = "cat data/chapter2/popular-names.txt | cut -f 1 > data/chapter2/col1.txt"
    os.system(cmd)
    cmd = "cat data/chapter2/popular-names.txt | cut -f 2 > data/chapter2/col2.txt"
    os.system(cmd)


def start_13():
    section("13. col1.txtとcol2.txtをマージ")
    cmd = (
        "paste data/chapter2/col1.txt data/chapter2/col2.txt > data/chapter2/merged.txt"
    )
    stdout = check_output(cmd, shell=True).decode()
    LIMIT = 5  # suppress long output
    print("\n".join(stdout.split("\n")[:LIMIT]))


def start_14():
    section("14. 先頭からN行を出力")
    N = 5
    cmd = f"head -n {N} data/chapter2/popular-names.txt"
    stdout = check_output(cmd, shell=True).decode()
    print(stdout)


def start_15():
    section("15. 末尾のN行を出力")
    N = 5
    cmd = f"tail -n {N} data/chapter2/popular-names.txt"
    stdout = check_output(cmd, shell=True).decode()
    print(stdout)


def start_16():
    section("16. ファイルをN分割する")
    N = 100
    cmd = f"split -l {N} data/chapter2/popular-names.txt data/chapter2/popular-names-"
    os.system(cmd)
    print("split completed")


def start_17():
    section("17. １列目の文字列の異なり")
    cmd = "cut -f 1 data/chapter2/popular-names.txt | sort | uniq"
    stdout = check_output(cmd, shell=True).decode()
    LIMIT = 5  # suppress long output
    print("\n".join(stdout.split("\n")[:LIMIT]))
    print("...")
    print("total:", stdout.count("\n"))


def start_18():
    section("18. 各行を3コラム目の数値の降順にソート")
    cmd = "sort -k 3 -n data/chapter2/popular-names.txt"
    stdout = check_output(cmd, shell=True).decode()
    LIMIT = 5  # suppress long output
    print("\n".join(stdout.split("\n")[:LIMIT]))
    print("...")
    print("\n".join(stdout.split("\n")[-LIMIT:]))


def start_19():
    section("19. 各行の1コラム目の文字列の出現頻度を求め、出現頻度の高い順に並べる")
    cmd = "cut -f 1 data/chapter2/popular-names.txt | sort | uniq -c | sort -r"
    stdout = check_output(cmd, shell=True).decode()
    LIMIT = 5  # suppress long output
    print("\n".join(stdout.split("\n")[:LIMIT]))
    print("...")


def start():
    chapter("第2章: UNIXコマンド")
    start_10()
    start_11()
    start_12()
    start_13()
    start_14()
    start_15()
    start_16()
    start_17()
    start_18()
    start_19()
