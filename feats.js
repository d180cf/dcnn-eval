/**
 * Input: SGF file with a tsumego.
 * Output: JSON file with features.
 */

const fs = require('fs');
const fspath = require('path');
const mkdirp = require('mkdirp');
const clargs = require('command-line-args');
const sgf = require('sgf-problems');
const tsumego = require('tsumego.js');

const args = clargs([
    { name: 'input', type: String },
    { name: 'output', type: String },
]);

try {
    const sgf = fs.readFileSync(args.input, 'utf8');
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const color = tsumego.sign(board.get(solver.target));
    const [x, y] = tsumego.stone.coords(solver.target);
    const feat = features(board, { x, y }, args.fpsize);
    const json = JSON.stringify(feat);

    mkdirp.sync(fspath.dirname(args.output));
    fs.writeFileSync(args.output, json, 'utf8');
} catch (err) {
    throw err;
}

/**
 * Computes features of the given board
 * and returns them as a list of feature
 * planes where each number is in `0..1` range.
 * 
 * @param {tsumego.Board} board
 * @param {{x: number, y: number}} target
 * @returns {number[][][]}
 */
function features(board, target) {
    const result = tensor([5, board.size + 2, board.size + 2]); // +2 to include the wall

    const FI_A = 0;
    const FI_E = 1;
    const FI_N = 2;
    const FI_1 = 3;
    const FI_S = 4;

    const color = tsumego.sign(board.get(target.x, target.y));

    for (let x = -1; x < board.size + 1; x++) {
        for (let y = -1; y < board.size + 1; y++) {
            const i = x + 1;
            const j = y + 1;

            if (!board.inBounds(x, y)) {
                result[FI_A][i][j] = 0;
                result[FI_E][i][j] = 0;
                result[FI_N][i][j] = 1;
                result[FI_1][i][j] = 0;
                result[FI_S][i][j] = 0;
            } else {
                const block = board.get(x, y);
                const nlibs = tsumego.block.libs(block);
                const nsize = tsumego.block.size(block);

                result[FI_A][i][j] = block * color > 0 ? 1 : 0;
                result[FI_E][i][j] = block * color < 0 ? 1 : 0;
                result[FI_N][i][j] = 0;
                result[FI_1][i][j] = nlibs == 1 ? 1 : 0;
                result[FI_S][i][j] = nsize > 1 ? 1 : 0;
            }
        }
    }

    return result;
}

/**
 * Creates a tensor of the given shape.
 * 
 *  - `tensor([]) = 0`
 *  - `tensor([4]) = [0, 0, 0, 0]`
 *  - `tensor([2, 3]) = [[0, 0, 0], [0, 0, 0]]`
 *  - `tensor([4, 7, 6]) = [...]`
 * 
 * @param {number[]} dimensions 
 */
function tensor(dimensions) {
    if (!dimensions.length)
        return 0;

    const result = new Array(dimensions[0]);

    for (let i = 0; i < result.length; i++)
        result[i] = tensor(dimensions.slice(1));

    return result;
}