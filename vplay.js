/**
 * Input: SGF file with variations.
 * Output: Labeled SGF files for all positions.
 */

const fs = require('fs');
const md5 = require('md5');
const fspath = require('path');
const mkdirp = require('mkdirp');
const tsumego = require('tsumego.js');

const [, , inputFile, outputDir] = process.argv;

try {
    const sgf = tsumego.SGF.parse(fs.readFileSync(inputFile, 'utf8'));
    const board = new tsumego.Board(sgf);

    let nvars = 0;
    let ndups = 0;

    for (const move of expand(board, sgf.vars)) {
        const hash = md5(board.sgf).slice(0, 7);
        const path = fspath.join(outputDir, hash + '.sgf');

        nvars++;

        if (fs.existsSync(path)) {
            ndups++;
        } else {
            mkdirp.sync(fspath.dirname(path));
            const data = board.sgf.slice(0, -1)
                + 'MA[' + sgf.steps[0].MA[0] + ']'
                + 'TR' + move.slice(1) + ')';
            fs.writeFileSync(path, data, 'utf8');
        }
    }

    console.log('vars = ' + nvars + ' dups = ' + ndups);
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