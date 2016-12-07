#!/bin/bash
# runs the example MNIST program
# takes a long time

set -ex

cd $(dirname $0)

if [ ! -s make_json ]
then
    echo "compiling 'make_json.c'"
    gcc -Ofast make_json.c -o make_json
    echo "ready"
fi

./make_json -t60000 -n10000 \
    | jq -f example.jq --argjson config "$(cat config-mnist.json)"
