/**
 * Input: SGF file with tsumego.
 * Output: JSON files with features and config.
 */

const fs = require('fs');
const fspath = require('path');
const fstext = require('./fstext');
const tsumego = require('tsumego.js');

const [, , input, outputDir] = process.argv;

const FI_N = 0; // neutral
const FI_A = 1; // ally
const FI_E = 2; // enemy
const FI_T = 3; // target
const FI_1 = 4; // atari
const FI_S = 5; // size > 1

(function main() {
    const sgf = fstext.read(input);
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const color = tsumego.sign(board.get(solver.target));
    const [x, y] = tsumego.stone.coords(solver.target);
    const feat = features(board, { x, y });

    const config = {
        safe: +(/\bTS\[(\d+)\]/.exec(sgf) || [])[1],
        size: board.size,
        area: +(/\bAS\[(\d+)\]/.exec(sgf) || [])[1],
    };

    fstext.write(
        fspath.join(outputDir, '/features.json'),
        JSON.stringify(feat));

    fstext.write(
        fspath.join(outputDir, '/config.json'),
        JSON.stringify(config));
})();

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
    const result = tensor([6, board.size + 2, board.size + 2]); // +2 to include the wall
    const tblock = board.get(target.x, target.y);
    const color = tsumego.sign(board.get(target.x, target.y));

    for (let x = -1; x < board.size + 1; x++) {
        for (let y = -1; y < board.size + 1; y++) {
            const j = x + 1;
            const i = y + 1;

            if (!board.inBounds(x, y)) {
                result[FI_A][i][j] = 0;
                result[FI_E][i][j] = 0;
                result[FI_T][i][j] = 0;
                result[FI_N][i][j] = 1;
                result[FI_1][i][j] = 0;
                result[FI_S][i][j] = 0;
            } else {
                const block = board.get(x, y);
                const nlibs = tsumego.block.libs(block);
                const nsize = tsumego.block.size(block);

                result[FI_A][i][j] = block * color > 0 ? 1 : 0;
                result[FI_E][i][j] = block * color < 0 ? 1 : 0;
                result[FI_T][i][j] = block == tblock ? 1 : 0;
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