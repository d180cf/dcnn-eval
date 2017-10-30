/**
 * Input: SGF file with tsumego.
 * Output: JSON files with features.
 */

const fs = require('fs');
const fspath = require('path');
const fstext = require('./fstext');
const tsumego = require('tsumego.js');

exports.compute = compute;

const F_WALL = 0;
const F_ALLY = 1;
const F_ENEMY = 2;
const F_VACANT = 3;

const F_LIBS_1 = 4;
const F_LIBS_2 = 5;
const F_LIBS_3 = 6;
const F_LIBS_4 = 7;
const F_LIBS_5 = 8;
const F_LIBS_X = 9;

const F_SIZE_1 = 10;
const F_SIZE_2 = 11;
const F_SIZE_3 = 12;
const F_SIZE_4 = 13;
const F_SIZE_5 = 14;
const F_SIZE_X = 15;

const F_COUNT = 16; // number of features

function compute(input, output) {
    const sgf = fstext.read(input);
    const solver = new tsumego.Solver(sgf);
    const board = solver.board;
    const target = solver.target;
    const tblock = board.get(target);
    const color = tsumego.sign(tblock);
    const [x, y] = tsumego.stone.coords(target);
    const feats = features(board, { x, y });

    const config = {
        safe: +(/\bTS\[(\d+)\]/.exec(sgf) || [])[1], // the label
        size: board.size,
        target: [...board.stones(tblock)].map(s => tsumego.stone.coords(s)),
        area: +(/\bAS\[(\d+)\]/.exec(sgf) || [])[1],
        features: feats
    };

    fstext.write(output, JSON.stringify(config));
}

/**
 * Computes features of the given board
 * and returns them as a list of feature
 * planes where each number is in `0..1` range.
 * 
 * @param {tsumego.Board} board
 * @param {{x: number, y: number}} target
 * @returns {number[][][]} shape = [board.size + 2, board.size + 2, F_COUNT]
 */
function features(board, target) {
    const result = tensor([board.size + 2, board.size + 2, F_COUNT]); // +2 to include the wall
    const color = tsumego.sign(board.get(target.x, target.y));

    for (let x = -1; x < board.size + 1; x++) {
        for (let y = -1; y < board.size + 1; y++) {
            const j = x + 1;
            const i = y + 1;

            if (!board.inBounds(x, y)) {
                result[i][j][F_WALL] = 1;
            } else {
                const block = board.get(x, y);
                const nlibs = tsumego.block.libs(block);
                const nsize = tsumego.block.size(block);

                result[i][j][F_ALLY] = block * color > 0 ? 1 : 0;
                result[i][j][F_ENEMY] = block * color < 0 ? 1 : 0;
                result[i][j][F_VACANT] = block ? 0 : 1;

                result[i][j][F_LIBS_1] = nlibs == 1 ? 1 : 0;
                result[i][j][F_LIBS_2] = nlibs == 2 ? 1 : 0;
                result[i][j][F_LIBS_3] = nlibs == 3 ? 1 : 0;
                result[i][j][F_LIBS_4] = nlibs == 4 ? 1 : 0;
                result[i][j][F_LIBS_5] = nlibs == 5 ? 1 : 0;
                result[i][j][F_LIBS_X] = nlibs >= 6 ? 1 : 0;

                result[i][j][F_SIZE_1] = nsize == 1 ? 1 : 0;
                result[i][j][F_SIZE_2] = nsize == 2 ? 1 : 0;
                result[i][j][F_SIZE_3] = nsize == 3 ? 1 : 0;
                result[i][j][F_SIZE_4] = nsize == 4 ? 1 : 0;
                result[i][j][F_SIZE_5] = nsize == 5 ? 1 : 0;
                result[i][j][F_SIZE_X] = nsize >= 6 ? 1 : 0;
            }
        }
    }

    return result;
}

/**
 * Creates a zero-initialized tensor of the given shape.
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