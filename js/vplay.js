/**
 * Input: SGF file with variations.
 * Output: Labeled SGF files for all positions.
 * 
 * Each subproblem has annotations:
 * 
 *  - MA[..] - the target stone
 *  - TS[..] - target status, 1 = safe, 0 = unsafe
 *  - AS[..] - the number of available moves
 */

const fs = require('fs');
const md5 = require('md5');
const fspath = require('path');
const mkdirp = require('mkdirp');
const tsumego = require('tsumego.js');

const [, , inputFile, outputDir] = process.argv;

try {
    const text = fs.readFileSync(inputFile, 'utf8');
    const sgf = tsumego.SGF.parse(text);
    const solver = new tsumego.Solver(text);
    const board = solver.board;
    const target = solver.target;
    const color = tsumego.sign(board.get(target));
    const player = text.indexOf('PL[B]') > 0 ? +1 : text.indexOf('PL[W]') > 0 ? -1 : 0;

    if (!player)
        throw Error('PL[.] needs to tell who makes the first move');

    let nvars = 0;
    let ndups = 0;
    let nsafe = 0;

    for (const [move, bestmove] of expand(board, sgf.vars)) {
        // "safe" means that if PL[.] makes the first move, MA[.] lives;
        // the fact that the current move has the same color as MA[.] means
        // that MA[.] was unsafe and needs an extra move to change this status;
        // correspondingly, if the move is of the opposite color, MA[.] was safe
        const size = [...solver.getValidMovesFor(move[0] == 'B' ? -1 : +1)].length;
        const safe = board.get(tsumego.stone.fromString(move)) * color > 0;
        const hash = md5(board.sgf).slice(0, 7);
        const path = fspath.join(outputDir, hash + '.sgf');

        nvars++;

        if (safe)
            nsafe++;

        if (fs.existsSync(path)) {
            ndups++;
        } else {
            mkdirp.sync(fspath.dirname(path));
            const data = board.sgf.slice(0, -1)
                + (bestmove ? 'TR[' + bestmove + ']' : '')
                + 'AS[' + size + ']'
                + 'MA[' + sgf.steps[0].MA[0] + ']'                
                + 'DS[' + (player * color > 0 ? 1 : 0) + ']'
                + 'TS[' + (safe ? 1 : 0) + '])';
            fs.writeFileSync(path, data, 'utf8');
        }
    }

    console.log('vars = ' + nvars + ' dups = ' + ndups + ' safe = ' + (nsafe / nvars * 100 | 0) + '%');
} catch (err) {
    throw err;
}

/**
 * Enumerates all subproblems.
 * 
 * @param {tsumego.Board} board 
 * @param {tsumego.SGF.Node[]} vars
 * @returns {Iterable<[string, string]>}
 */
function* expand(board, vars) {
    for (const node of vars) {
        for (const step of node.steps) {
            const move = step.B ?
                'B[' + step.B[0] + ']' :
                'W[' + step.W[0] + ']';

            board.play(tsumego.stone.fromString(move));
            yield [move, step.TR && step.TR[0]];
        }

        if (node.vars)
            yield* expand(board, node.vars);

        for (const step of node.steps)
            board.undo();
    }
}