include "math_fns";

def array_init(size; filter): [ range(size) | filter ];

def array_2d(size1; size2; filter): 
  array_init(size1; array_init(size2; filter)) ; 

def config(args):
  args as $args 
  | . = {} 
  
  # input layer
  | .input_gradients =      (array_init($args.input_size; 0))
  | .input_deltas =         (array_init($args.input_size; 0))

  # input -> hidden weights
  | .ih_weights =           (array_2d($args.input_size; $args.hidden_size; rand(0.1)))
  | .ih_deltas =            (array_2d($args.input_size; $args.hidden_size; 0))
  | .ih_gradients =         (array_2d($args.input_size; $args.hidden_size; 0))

  # hidden layer
  | .hidden_biases =        (array_init($args.hidden_size; rand(0.1)))
  | .hidden_gradients =     (array_init($args.hidden_size; 0))
  | .hidden_deltas =        (array_init($args.hidden_size; 0))

  # hidden -> output weights
  | .ho_weights =           (array_2d($args.hidden_size; $args.output_size; rand(0.1)))
  | .ho_deltas =            (array_2d($args.hidden_size; $args.output_size; 0))
  | .ho_gradients =         (array_2d($args.hidden_size; $args.output_size; 0))

  # output layer
  | .output_biases =        (array_init($args.output_size; rand(0.1)))
  | .output_gradients =     (array_init($args.output_size; 0))
  | .output_deltas =        (array_init($args.output_size; 0)) ;


# process .input_activations to product .output_activations
def forwardpass:
  # calculate hidden activations
  .hidden_activations = (
      vec_add(
        .hidden_biases; 
        multiply([.input_activations]; .ih_weights) | .[0])
      | map(sigmoid) )

  # calculate output activations
  | .output_activations = (
      vec_add(
        .output_biases; 
        multiply([.hidden_activations]; .ho_weights) | .[0])
      | map(sigmoid) ) ;


# generate deltas between .output_activations and the expected data set
# produce .output_gradients, .hidden_gradients, .input_gradients
def backwardpass:
  # calculate output gradients
  .output_gradients = (
        [ vec_sub(.output_activations; .expected) ]
      | transpose 
      | map(.[0]) ) 

  # calculate hidden gradients
  | .hidden_gradients = 
      hadamard(
        .hidden_activations | map(dsigmoid);
        multiply(.ho_weights; .output_gradients | map([.])) | map(.[0])
      )

  # calculate input gradients
  | .input_gradients = 
      hadamard(
        .input_activations;
        multiply(.ih_weights; .hidden_gradients | map([.])) | map(.[0])
      ) ;

# def weight_update