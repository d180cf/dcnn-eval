/**
 * Input: SGF file with variations.
 * Output: Labeled SGF files for all subproblems.
 * 
 * Each subproblem has annotations:
 * 
 *  - MA[..] - the target stone
 *  - TS[..] - target status, + means safe, - means dead
 *  - AS[..] - the number of available moves
 */

const fs = require('fs');
const md5 = require('md5');
const fspath = require('path');
const mkdirp = require('mkdirp');
const tsumego = require('tsumego.js');
const fstext = require('./fstext');

const [, , inputFile, outputDir] = process.argv;

try {
    const text = fstext.read(inputFile);
    const sgf = tsumego.SGF.parse(text);
    const solver = new tsumego.Solver(text);
    const board = solver.board;
    const target = solver.target;
    const color = tsumego.sign(board.get(target));

    let nvars = 0;
    let ndups = 0;
    let nsafe = 0;
    let ndead = 0;

    for (const [move, status] of expand(board, sgf.vars)) {
        const size1 = [...solver.getValidMovesFor(+1)].length;
        const size2 = [...solver.getValidMovesFor(-1)].length;
        const size = Math.max(size1, size2);
        const hash = md5(board.sgf).slice(0, 7);
        const path = fspath.join(outputDir, size + '', status, hash + '.sgf');

        nvars++;

        if (status == '+') nsafe++;
        if (status == '-') ndead++;

        const data = board.sgf.slice(0, -1)
            + 'AS[' + size + ']'
            + 'MA[' + sgf.steps[0].MA[0] + ']'
            + 'TS[' + status + '])';

        ndups += fstext.write(path, data);
    }

    console.log('vars = ' + nvars + ' dups = ' + ndups + ' safe = ' + nsafe + ' dead = ' + ndead);
} catch (err) {
    throw err;
}

/**
 * Enumerates all subproblems.
 * 
 * @param {tsumego.Board} board 
 * @param {tsumego.SGF.Node[]} vars
 */
function* expand(board, vars) {
    for (const node of vars) {
        for (const step of node.steps) {
            const move = step.B ?
                'B[' + step.B[0] + ']' :
                'W[' + step.W[0] + ']';

            const status = step.TS && step.TS[0] || '=';
            board.play(tsumego.stone.fromString(move));
            yield [move, status];
        }

        if (node.vars)
            yield* expand(board, node.vars);

        for (const step of node.steps)
            board.undo();
    }
}