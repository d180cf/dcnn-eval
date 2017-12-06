/**
 * Input: SGF file with tsumego.
 * Output: JSON files with features.
 */

const fs = require('fs');
const fspath = require('path');
const tsumego = require('tsumego.js');

const fstext = require('./fstext');
const { features, F_COUNT } = require('./features');

exports.compute = compute;

function compute(input, output) {
    const sgf = fstext.read(input);
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const target = solver.target;
    const tblock = board.get(target);
    const color = tsumego.sign(tblock);
    const [x, y] = tsumego.stone.coords(target);
    const safe = +/\bTS\[(\d+)\]/.exec(sgf)[1];
    const defs = +/\bDS\[(\d+)\]/.exec(sgf)[1];
    const feats = new Array(board.size ** 2 * F_COUNT);

    features(feats, board, { x, y }, defs);

    const config = {
        safe: safe,
        bsize: board.size,
        target: [...board.stones(tblock)].map(s => tsumego.stone.coords(s)),
        asize: +/\bAS\[(\d+)\]/.exec(sgf)[1],
        shape: [board.size, board.size, F_COUNT],
        features: feats
    };

    fstext.write(output, JSON.stringify(config));
}
