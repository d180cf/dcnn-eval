const F_COUNT = 13; // the number of features

const [

    F_WALL,
    F_ALLY,
    F_ENEMY,
    F_TARGET,
    F_SURE_EYE,

    F_LIBS_1,
    F_LIBS_2,
    F_LIBS_3,
    F_LIBS_4, // 4+ libs

    F_SIZE_1,
    F_SIZE_2,
    F_SIZE_3,
    F_SIZE_4, // 4+ stones

] = Array.from(Array(F_COUNT).keys());

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
                const { libs: nlibs, size: nsize } = board.getBlockInfo(x, y);

                result[base + F_ALLY] = block * color > 0 ? 1 : 0;
                result[base + F_ENEMY] = block * color < 0 ? 1 : 0;
                result[base + F_TARGET] = block == tblock ? 1 : 0;
                result[base + F_SURE_EYE] = isSureEye(board, +1, x, y) || isSureEye(board, -1, x, y) ? 1 : 0;

                result[base + F_LIBS_1] = nlibs == 1 ? 1 : 0;
                result[base + F_LIBS_2] = nlibs == 2 ? 1 : 0;
                result[base + F_LIBS_3] = nlibs == 3 ? 1 : 0;
                result[base + F_LIBS_4] = nlibs >= 4 ? 1 : 0;

                result[base + F_SIZE_1] = nsize == 1 ? 1 : 0;
                result[base + F_SIZE_2] = nsize == 2 ? 1 : 0;
                result[base + F_SIZE_3] = nsize == 3 ? 1 : 0;
                result[base + F_SIZE_4] = nsize >= 4 ? 1 : 0;
            }
        }
    }
}

/**
 * Detects 1-point sure eyes.
 * 
 * @param {tsumego.Board} board 
 * @param {number} color 
 * @param {number} x 
 * @param {number} y 
 */
function isSureEye(board, color, x, y) {
    const n = board.size;

    const get = (dx, dy) => board.get(x + dx, y + dy);
    const dist = (x, y) => Math.min(x, n - 1 - x) + Math.min(y, n - 1 - y);
    const isWall = (dx, dy) => !board.inBounds(x + dx, y + dy);
    const isCorner = (dx, dy) => dist(x + dx, y + dy) == 0;

    if (get(0, 0))
        return false;

    let count = 0;
    let ndiag = 0;
    let nwall = 0;
    let necrn = 0;

    for (let dx = -1; dx <= +1; dx++) {
        for (let dy = -1; dy <= +1; dy++) {
            if (isWall(dx, dy)) {
                nwall++;
            } else {
                const x = get(dx, dy);

                if (x * color > 0) {
                    count++;

                    if (dx && dy)
                        ndiag++;
                } else if (!x && isCorner(dx, dy)) {
                    necrn++;
                }
            }
        }
    }

    switch (count) {
        case 8:
            // X X X
            // X - X
            // X X X
            return true;

        case 7:
            // X X X
            // X - X
            // X X -
            return ndiag == 3;

        case 6:
            // X X -
            // X - X
            // X X -
            return necrn == 1;

        case 5:
            // X X - //  X - -
            // X - - //  X - -
            // X X - //  X X X
            return nwall == 3;

        case 4:
            // X - -
            // X - -
            // X X -        
            return nwall == 4;

        case 3:
            // - - -
            // X - -
            // X X -        
            return nwall == 5;
    }

    return false;
}
