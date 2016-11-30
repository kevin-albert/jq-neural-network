# jq-neural-network
A neural network written completely in jq.

## How to run:
```
$ ./run.sh
```
This will first compile `make_json.c`, which parses MNIST data files and produces JSON.  Then it will run `make_json` and pipe it into jq, first processing the training set and then the test set. When finished, jq will emit the error rate of the test set.  

This will take a **LONG** time to run. Maybe a day. Days? I don't know. I've never waited for it to finish with the whole data
set.

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

## Project overview
The main reduction is in `main.jq`. Neural network filters (forwardpass, backwardpass, updateweights) are contained in `neural_net.jq` and vector / matrix / utility functions live in `util.jq`.  

Additionally, MNIST data files are included along with a parser (`make_json.c`).
