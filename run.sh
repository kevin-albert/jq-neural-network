#!/bin/bash
jq -f main.jq \
  --argjson config '{ "input_size": 3, "hidden_size": 10, "output_size": 2 }' \
  input.json
