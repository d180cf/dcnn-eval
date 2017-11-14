const tsumego = require('tsumego.js');

const F_WALL = 0;
const F_ALLY = 1;
const F_ENEMY = 2;
const F_ATARI = 3;
const F_SIZE_1 = 4;

const F_COUNT = 5; // number of features

exports.F_COUNT = F_COUNT;

/**
 * Computes features of the given board
 * and returns them as a list of feature
 * planes where each number is in `0..1` range.
 * 
 * @param {number[]} result NHWC format;
 *      shape = `[n + 2, n + 2, F_COUNT]`, n = `board.size`;
 *      index = `[y + 1, x + 1, f]` to account for walls;
 *      x, y, f are all zero based
 * @param {tsumego.Board} board
 * @param {{x: number, y: number}} target
 */
exports.features = function features(result, board, target) {
    const size = board.size;
    const tblock = board.get(target.x, target.y);
    const color = Math.sign(tblock);
    const offset = (x, y) => (y + 1) * (size + 2) * F_COUNT + (x + 1) * F_COUNT;

    if ((size + 2) * (size + 2) * F_COUNT > result.length)
        throw Error('The output array is too small: ' + result.length);

    for (let i = 0; i < result.length; i++)
        result[i] = 0;

    for (let x = -1; x < size + 1; x++) {
        for (let y = -1; y < size + 1; y++) {
            const base = offset(x, y);

            if (!board.inBounds(x, y)) {
                result[base + F_WALL] = 1;
            } else {
                const block = board.get(x, y);
                const nlibs = tsumego.block.libs(block);
                const nsize = tsumego.block.size(block);

                result[base + F_ALLY] = block * color > 0 ? 1 : 0;
                result[base + F_ENEMY] = block * color < 0 ? 1 : 0;
                result[base + F_ATARI] = nlibs == 1 ? 1 : 0;
                result[base + F_SIZE_1] = nsize == 1 ? 1 : 0;
            }
        }
    }
}
