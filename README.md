The idea is to build a DCNN that would answer one question:
is the group safe? More precisely, it will tell whether
the group cannot be captured if the opponent starts
and can recapture any ko. If the group is safe, the player
can tenuki. If it's not safe, a move is needed to protect it.

# Features

The input to the DCNN will be a list of planes with features
or in other terms in will be a tensor of shape `[F, N, N]`
where `F` is the number of features and `N x N` is the area
that these features are computed for. Each `N x N` frame is
a matrix composed of `0`s and `1`s only.

The features are:

- If the location is outside the board.
- If the stone is black.
- If the stone is white.
- If the stone is in atari, i.e. it has only 1 liberty.
- If the stone has adjcent stones of the same color.

More features to be implemented:

- If the stone can be captured in a ladder (aka the lambda-1 sequence).
- If the stone can be captured with a net (aka the lambda-2 sequence).
- If this location is the center of a nakade shape.
- If the stone is surely alive, i.e. belongs to the outer wall or to an alive shape.
- Some sort of eye-detection, heuristics and so on.

Features are computed with a JS script:

```
npm run features
```

This script reads SGF files with annotated tsumegos, solves them
to find all subproblems, for each subproblem computes features and
adds a label about the status of the group. The output is about
60,000 labeled tsumegos: 50,000 for training the DCNN and another
10,000 for testing it.

# DCNN

Then this set of feature tensors is fed to a Python script that
uses TensorFlow to find the DCNN parameters.

Once the DCNN parameters are found, it can be used in the tsumego
solver to evaluate the board and refine the search.

For now the DCNN design is simple and is taken from the ["MNIST for Experts"](https://www.tensorflow.org/get_started/mnist/pros) TensorFlow tutorial:

1. The input is a set of `[11, 11, 5]` tensors where `11 x 11` is an area on the board with the target stone in the center and `5` is the number of features described above.
2. The 1-st layer transofrms this tensor into a let's say `[5, 5, 10]` one with a convolution and max pooling. The area is shirnked to `5 x 5`, but the number of features is doubled.
3. The 2-nd layer does the same and makes a `[2, 2, 20]` tensor.
4. Densely-connected layer transforms the `[2, 2, 20]` tensor into a `[256]` vector.
5. Dropout to reduce overfitting.
6. Readout to transform the `[256]` vector into a value in the `0..1` range: the prediction whether the target group is safe.

DCNN is trained with a Python script:

```
npm run train-nn
```

This script reads the generated set of 60,000 tsumegos, trains the DCNN
on the first 50,000 and uses the rest to check the DCNN's accuracy.
