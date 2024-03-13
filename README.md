# NLP 100 本ノック

- [NLP 100 本ノック](https://nlp100.github.io/ja/) に取り組んでみる

## Setup

```sh
gunzip -c data/chapter3/jawiki-country.json.gz > data/chapter3/jawiki-country.json
```

<https://taku910.github.io/mecab/>

から MeCab本体とIPA 辞書 のtar.gzをダウンロードして解凍

```sh
tar xvfz mecab-x.x.tar.gz
cd mecab-x.x
./configure --with-charset=utf8
make
make check
sudo make install
```

```sh
tar xvfz mecab-ipadic-x.x.tar.gz
cd mecab-ipadic-x.x
./configure --with-charset=utf8 --enable-utf8-only
make
sudo make install
```

## Word2Vec

[GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?resourcekey=0-wjGZdNAUop6WykTtMip30g) をダウンロードして解凍

```sh
gunzip -c GoogleNews-vectors-negative300.bin.gz > data/chapter7/GoogleNews-vectors-negative300.bin
```

