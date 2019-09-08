#!/bin/bash

rm -rf html

git clone -b gh-pages git@github.com:stg7/quat.git html

make html

cd html
git add .
git commit -m "update of github.io quat page @ $(date)"
git push origin gh-pages

