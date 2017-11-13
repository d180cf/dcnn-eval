const tsumego = require('tsumego.js');

const F_WALL = 0;
const F_ALLY = 1;
const F_ENEMY = 2;
const F_ATARI = 3;
const F_SIZE_1 = 4;

const F_COUNT = 5; // number of features

/**
 * Computes features of the given board
 * and returns them as a list of feature
 * planes where each number is in `0..1` range.
 * 
 * @param {tsumego.Board} board
 * @param {{x: number, y: number}} target
 * @returns {number[][][]} shape = [board.size + 2, board.size + 2, F_COUNT]
 */
exports.features = function features(board, target) {
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
                result[i][j][F_ATARI] = nlibs == 1 ? 1 : 0;
                result[i][j][F_SIZE_1] = nsize == 1 ? 1 : 0;
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
