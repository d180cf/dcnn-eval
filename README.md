The idea is to build a DCNN that would answer one question:
is the group safe? More precisely, it will tell whether
the group cannot be captured if the opponent starts
and can recapture any ko. If the group is safe, the player
can tenuki. If it's not safe, a move is needed to protect it.

# Features

The input to the DCNN will be a list of planes with features
or in other terms in will be a tensor of shape `[N, N, F]`
where `F` is the number of features and `N x N` is the area
that these features are computed for. A reasonable choice would
be `11 x 11` or `9 x 9` because most tsumegos fit in this area.

The features are:

- location is outside the board
- stone is black
- stone is white
- stone is in atari, i.e. it has only 1 liberty
- stone has adjcent stones of the same color

More features to be implemented:

- stone can be captured in a ladder (aka the lambda-1 sequence)
- stone can be captured with a net (aka the lambda-2 sequence)
- location is the center of a nakade shape
- safe group, i.e. stone belongs to the outer wall or to an alive shape
- some sort of eye-detection, heuristics and so on

# DCNN

This set of feature tensors is fed to a Python script that uses
[TensorFlow](https://github.com/tensorflow/tensorflow) to find the DCNN parameters. Once the DCNN parameters are found, they can be exported to a file and the tsumego solver can use [keras.js](https://github.com/transcranial/keras-js)
to evaluate the board and refine the search.

For now the DCNN design is simple and mimics convnets for [recognizing letters](https://www.tensorflow.org/get_started/mnist/pros):

1. Conv layer #1 with `[3, 3]` kernel to map a `[11, 11, 5]` feature tensor to a `[9, 9, 32]` one.
2. Conv layer #2 with `[3, 3]` kernel to get a `[7, 7, 32]` tensor.
3. Conv layer #3 with `[3, 3]` kernel to get a `[5, 5, 32]` tensor.
4. A densely-connected layer to get a `[1024]` vector.
5. Dropout to reduce overfitting.
6. Readout to get a `[2]` vector: the prediction whether the target is safe.

The error rate of this DCNN is 20% (the error rate of a random number generator is 50% since there are only two outputs).

There is also a [paper](http://www.cs.cityu.edu.hk/~hwchun/research/PDF/Julian%20WONG%20-%20CCCT%202004%20a.pdf) describing a DCNN that has accuracy rate 97% in evaluating tsumego status.

# How inputs to DCNN are generated

There is a set of 100 or so handmade (well, mostly taken from goproblems.com) tsumegos with proper outer wall and marked target. They are stored in the `sgf-problems` module.

```
npm run solve-all
```

Solves each tsumego and outputs a tree of subproblems. When generating the tree the script picks moves that change the safety status of the target group. Every node (not just leafs) in this tree is a subproblem. This step takes a while, but its output is compact.

```
npm run vplay-all
```

Plays out all the moves in the tree and generates a separate SGF file for each node. Each subproblem is labeled with `TS[1]` if the target group is safe. Also, `npm run stats` prints how many boards have safe or unsafe target per board size:

```
size   safe unsafe
------------------
   0      0   3196
   1   1968  11768
   2   8569  24258
   3  23069  36662
   4  44738  44791
   5  41719  44377
   6  43741  31277
   7  21476  22465
   8  15373   8902
   9   4532   5070
  10   2307    981
  11    518    517
  12    114     47
  13     39     22
```

Boards with too few available moves can be ignored as it's easier to run the usual DFS than to run a DCNN. There are about 80K boards with 7+ available moves - a good enough training set.

```
npm run check-all
```

Verifies the `TS[1]` labels. It picks a small percentage of SGF files, solves them and checks if the status is correct. This step is optional.

```
npm run feats-all
```

Computes features for all subproblems. It outputs a JSON file with feature planes and metadata for each subproblem. The feature planes tensor has the shape of `[board.size + 2, board.size + 2, F]` where `F` is the number of features. The point is to later extract `N x N x F` subtensors. These JSON files will be then read by the Python script.

```
npm run dcnn
```

From each JSON file with features a script takes `N x N` areas around a stone in the target block. The result is a feature tensor of known shape: `[N, N, F]` where `N x N` is the chosen size of the sliding window and `F` is the number of features. There are also 8 symmetries generated by rotation and transposition and there are usually several stones in the target block, so the number of feature tensors that can be extracted from a JSON file is `T x 8` where `T` is the size of the target block. Given that the number of JSON files is about 500K, this gives a decent training set.
