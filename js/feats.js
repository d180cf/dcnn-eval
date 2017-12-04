/**
 * Input: SGF file with tsumego.
 * Output: JSON files with features.
 */

const tsumego = require('tsumego.js');

const fstext = require('./fstext');
const { features, F_COUNT } = require('./features');

exports.compute = function compute(sgf, player) {
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const target = solver.target;
    const tblock = board.get(target);
    const color = tsumego.sign(tblock);
    const [x, y] = tsumego.stone.coords(target);
    const status = sgf.indexOf('TS[+]') > 0 ? +1 : sgf.indexOf('TS[-]') > 0 ? -1 : 0;
    const planes = new Array(board.size ** 2 * F_COUNT);

    features(planes, board, { x, y }, player);

    return {
        // if status = +1, then the group is safe and the label = +1
        // if status = -1, then the group is dead and the label = -1
        // if status = 0, then it depends on who plays first, i.e. label = player
        status: status || player,
        target: [...board.stones(tblock)].map(s => tsumego.stone.coords(s)).reduce((r, x) => [...r, ...x], []),
        areasize: +/\bAS\[(\d+)\]/.exec(sgf)[1],
        shape: [board.size, board.size, F_COUNT],
        planes: planes
    };
};
