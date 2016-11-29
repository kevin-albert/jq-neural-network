
def log(thing):
  (thing|debug) as $_
  | . ; 

def rand(scale):  
  . * 15485863 
  | . % 1000
  | . - 500
  | . / 1000
  | . * scale ;

# σ(x)
def sigmoid:
  2.718281828459045 as $E
  | 1 / (1 + pow($E; -.)) ;

# dsigmoid(y) = σ'(x) where y = σ(x)
def dsigmoid:
  . * (1 - .) ;

def dot(a; b):
  a as $a | b as $b
  | reduce range(0; $a|length) as $i (0; . + ($a[$i] * $b[$i]) ) ;

def vec_add(a; b):
  a as $a | b as $b 
  | reduce range(0; $a|length) as $i ([]; . + [$a[$i] + $b[$i]]) ;

def vec_sub(a; b):
  a as $a | b as $b
  | reduce range(0; $a|length) as $i ([]; . + [$a[$i] - $b[$i]]) ;

def matrix_add(A; B):
  A as $A | B as $B
  | reduce range(0; $A|length) as $i ([]; . + [vec_add($A[$i]; $B[$i])]) ;

def matrix_scale(s):
  s as $s |
  map(map(. * s)) ;

def multiply(A; B):
  A as $A | B as $B
  | ($B[0]|length) as $p
  | ($B|transpose) as $BT
  | reduce range(0; $A|length) as $i
       ([]; reduce range(0; $p) as $j 
         (.; .[$i][$j] = dot( $A[$i]; $BT[$j] ) )) ;

def hadamard(A; B):
  A as $A | B as $B 
  | reduce range(0; $A|length) as $i
      ([]; . + [$A[$i] * $B[$i]]) ;
