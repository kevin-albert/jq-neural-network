#
# Usage: 
# jq -f main.jq --argjson \
#   config '{ "input_size": 3, "hidden_size": 10, "output_size": 2 }' \
#   fail.json
#

include "neural_net";
include "math_fns";

def max_index:
  . as $array
  | reduce range(0; .|length) as $i 
    ( 0; 
      if $array[$i] > $array[.] then $i
      else .
      end
    ) ;

def main:
  (
    config($config) 
    | .num_errors = 0
    | .num_total = 0
  ) as $state
  
  # track correct / incorrect guesses
  # process each input record and emit a new state
  | reduce .[] as $record ($state; 
        .input_activations = $record.input 
      | .expected = $record.expected
      | forwardpass

      # training - propagate errors and update weights
      | if $record.train then
          backwardpass
        
      # not training - compare expected / actual value
        else 
          if (.output_activations | max_index) != (.expected | max_index)
            then .num_errors += 1
            else .
          end
          | .num_total += 1
        end
  ) 

  # calculate error rate (excluding training values)
  | if .num_total > 0 then .error_rate = .num_errors / .num_total else . end ;


# go
main