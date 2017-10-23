The idea is to build a DCNN that would answer one question:
can this group be captured? More precisely, it will tell
whether the group can be captured if the opponent starts
and can recapture any ko.

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

Then this set of feature tensors is fed to a Python script that
uses TensorFlow to find the DCNN parameters.

Once the DCNN parameters are found, it can be used in the tsumego
solver to evaluate the board and refine the search.
