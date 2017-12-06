const DCNN = require('./js/dcnn');
const json = require('./.bin/tf-model.json');

const dcnn = new DCNN(json);

// defstarts = 1 if the defender makes the first move
// defstarts = 0 if the attacker makes the first move
window.evaldcnn = function evaldcnn(board, [x, y], defstarts) {
    return dcnn.eval(board, [x, y], defstarts);
};
