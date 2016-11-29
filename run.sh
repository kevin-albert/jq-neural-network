#!/bin/bash

set -e

if [ ! -s make_json ]
then
    echo "compiling 'make_json.c'"
    gcc -Ofast make_json.c -o make_json
fi

./make_json \
    | jq -f main.jq --argjson config "$(cat config-mnist.json)"
