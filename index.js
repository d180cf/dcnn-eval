const DCNN = require('./js/dcnn');
const json = require('./.bin/tf-model.json');

const dcnn = new DCNN(json);

window.evaldcnn = function evaldcnn(board, [x, y]) {
    return dcnn.eval(board, [x, y]);
};
