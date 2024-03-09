def green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def chapter(name: str) -> None:
    print(f"\n{red(f'===== {name} =====')}")


def section(name: str) -> None:
    print(f"\n{green(name)}")
