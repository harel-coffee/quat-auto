#!/bin/bash

rm -rf html

git clone -b gh-pages git@github.com:stg7/quat.git html

cd html
rm -rf index.html
cd ..
make html

if [[ -f html/index.html ]]; then
    cd html
    git add .
    git commit -m "update of github.io quat page @ $(date)"
    git push origin gh-pages
else
    echo "something is wrong with make html"
fi
