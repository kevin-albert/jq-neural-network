include "util";

def array_init(size; filter): [ range(size) | filter ];

def array_2d(size1; size2; filter): 
  array_init(size1; array_init(size2; filter)) ;

def initial_state(args):
  args as $args 
  |{
    # momentum (not used yet. someday)
    # alpha:            0.04,

    # learning rate
    eta:              0.1,

    # input layer
    input_gradients:  (array_init($args.input_size; 0)),

    # input -> hidden weights
    ih_weights:       (array_2d($args.input_size; $args.hidden_size; rand(0.1))),

    # hidden layer
    hidden_biases:    (array_init($args.hidden_size; rand(0.1))),

    # hidden -> output weights
    ho_weights:       (array_2d($args.hidden_size; $args.output_size; rand(0.1))),

    # output layer
    output_biases:    (array_init($args.output_size; rand(0.1)))
  } ;


#
# algorithms mimic the style of 
# http://briandolhansky.com/blog/2014/10/30/artificial-neural-networks-matrix-form-part-5
#

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


# adjust parameters based on previously computed gradients
def updateweights: 
  # update input -> hidden weights
  .eta as $eta
  | .ih_weights = matrix_add(
      multiply(.hidden_gradients | map([.]); [.input_activations])
        | transpose
        | matrix_scale(-$eta);
      .ih_weights)

  # update hidden biases
  | .hidden_biases = vec_add(.hidden_biases; .hidden_gradients|map(. * -$eta)) 

  # update hidden -> output weights
  | .ho_weights = matrix_add(
      multiply(.output_gradients | map([.]); [.hidden_activations])
        | transpose
        | matrix_scale(-$eta);
      .ho_weights) 

  # update output biases
  | .output_biases = vec_add(.output_biases; .output_gradients|map(. * -$eta)) ;
  