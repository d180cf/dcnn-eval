/**
 * Input: SGF file with variations.
 * Output: Labeled SGF files for all positions.
 */

const fs = require('fs');
const fspath = require('path');
const mkdirp = require('mkdirp');
const clargs = require('command-line-args');
const tsumego = require('tsumego.js');

const args = clargs([
    { name: 'input', type: String },
    { name: 'output', type: String }, // e.g. /foo/bar/*.sgf
]);

try {
    const sgf = tsumego.SGF.parse(fs.readFileSync(args.input, 'utf8'));
    const board = new tsumego.Board(sgf);

    for (const move of expand(board, sgf.vars)) {
        const path = args.output.replace('*', tsumego.hex(board.hash));
        console.log(path);
        mkdirp.sync(fspath.dirname(path));
        const data = board.sgf.slice(0, -1) + 'TR' + move.slice(1) + ')';
        fs.writeFileSync(path, data, 'utf8');
    }
} catch (err) {
    throw err;
}

/**
 * Enumerates all subproblems.
 * 
 * @param {tsumego.Board} board 
 * @param {tsumego.SGF.Node[]} vars
 * @returns {Iterable<string>}
 */
function* expand(board, vars) {
    for (const node of vars) {
        for (const step of node.steps) {
            const move = step.B ?
                'B[' + step.B[0] + ']' :
                'W[' + step.W[0] + ']';

            board.play(tsumego.stone.fromString(move));
            yield move;
        }

        if (node.vars)
            yield* expand(board, node.vars);

        for (const step of node.steps)
            board.undo();
    }
}