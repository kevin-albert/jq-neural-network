include "util";
include "neural_net";

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
    initial_state($config) 
    | .num_errors = 0
    | .num_test = 0
    | .num_train = 0
  ) as $state
  
  # track correct / incorrect guesses
  # process each input record and emit a new state
  | reduce inputs as $record ($state; 
        .input_activations = $record.input 
      | .expected = $record.expected
      | forwardpass

      # training - propagate errors and update weights
      | if $record.train 
        then
            backwardpass
          | updateweights
          | .num_train += 1

      # not training - compare expected / actual value
        else 
          if (.output_activations | max_index) != (.expected | max_index)
            then .num_errors += 1
            else .
          end
          | .num_test += 1
        end

      # log something every now and then
      | (.num_test + .num_train) as $num_total 
      | if $num_total > 0 and $num_total % 100 == 0
        then
          log(  "total: " + ($num_total|tostring) + 
              ", train: " + (.num_train|tostring) +
              ", test: " + (.num_test|tostring) +
              ", errors: " + (.num_errors|tostring))
        else . 
        end
  ) 

  # calculate error rate (excluding training values)
  | if .num_test > 0 then 
      .error_rate = .num_errors / .num_test 
    else . end 
  | {
      error_rate: .error_rate,
      num_errors: .num_errors,
      num_train: .num_train,
      num_test: .num_test,
      num_total: (.num_train+.num_test)
    } ;


# go
main