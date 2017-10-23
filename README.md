# Build

`npm i`
`npm run features`

This script does a bunch of things:
  
    1. reads SGF files from the sgf-problems repo
    2. solves each tsumego
    3. picks all relevant subproblems
    4. computes all relevant features
    5. generates JSON files with those features
  
 Then a Python script reads the JSON files
 and feeds to TensorFlow. The output of TF
 is a large NN that evaluates the status of
 a given tsumego.

