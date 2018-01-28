#!/bin/bash
#face/の初期化をします(水増ししたやつを削除します。)
#example  initface.sh DA  hoge_DA.jpgを削除
if [ $# -ne 1 ]; then
  echo "【使い方】 initface.sh 任意の識別子" 1>&2
  exit 1
fi

rm -rf ./face/Kagami/*_$1.jpg
rm -rf ./face/Kato/*_$1.jpg
rm -rf ./face/Kuroda/*_$1.jpg
rm -rf ./face/Yamane/*_$1.jpg
rm -rf ./face/Yamada/*_$1.jpg
rm -rf ./face/Uchiyama/*_$1.jpg
rm -rf ./face/Sasaki/*_$1.jpg
rm -rf ./face/Suetomi/*_$1.jpg
rm -rf ./face/Tanaka/*_$1.jpg