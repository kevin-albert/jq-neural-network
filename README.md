# jq-neural-network
A neural network written completely in jq. Sample program processes the MNIST image dataset with a success rate of 94%.

## Running the sample program:
```
$ ./example/run.sh
```
This will first compile `make_json.c`, which parses MNIST data files and produces JSON.  Then it will run `make_json` and pipe it into jq, first processing the training set and then the test set. When finished, jq will emit the error rate of the test set.  

This may take days to run to completion.

## Why
Two reasons:

1. To demonstrate how powerful the jq language is by writing a non-trivial full program with it
2. To exercise writing a neural network without using the "standard" tool set (no python)

## How it works
A typical feedforward neural network would be implemented by allocating several arrays and continually updating their values. Because data in jq is immutable, this continual state update is implemented as a reduction over all input, where the state of the network is the value being accumulated in the reduction. A benefit of this is that the state of the network could be easily saved / loaded to and from json files in between program executions.  

Configuration is supplied via a JSON argument to jq (see `run.sh` for an example). Sample config is `config-trivial.json`.  

The input is a stream of records like: 
```js
{ 
  "input": [...],       // array of numbers (should be the same size as config.input_size)
  "expected": [...],    // array of numbers (should be the same size as config.output_size)
  "train": true|false   // "true" means backpropagation is done. "false" means that the 
                        // expected / output are compared and error rate updated
}
```

It occasionally logs its progress to `stderr` in the form of jq debug statements:
```js
["DEBUG:","total: 66400, train: 59999, test: 6401, errors: 446"]
```

Its final output is the number of records processed and error rate:
```js
{
  "error_rate": 0.0593,
  "num_errors": 593,
  "num_train": 59999,
  "num_test": 10000,
  "num_total": 69999
}
```

## Project overview
The example program, configuration, and data live in `example/example.jq`. Neural network library is in `neural_net.jq`, and sample configuration file for a tiny network is in `config-trivial.json`  
