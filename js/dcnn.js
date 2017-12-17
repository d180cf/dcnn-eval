const nn = require('./nn');
const { features, F_COUNT } = require('./features');

const WINDOW_SIZE = 15; // must match the DCNN
const WINDOW_HALF = WINDOW_SIZE / 2 | 0;

module.exports = class DCNN {
    /**
     * Reconstructs the DCNN that correponds
     * to the weights file. The result is a 
     * function that takes a tensor as input
     * and returns a single value in the 0..1
     * range.
     * 
     * @param {JSON} json description of the NN
     */
    constructor(json) {
        this._nnfn = reconstructDCNN(json);
        this._planes = new Float32Array(20 ** 2 * F_COUNT); // enough for any board size
        this._fslice = new Float32Array(WINDOW_SIZE ** 2 * F_COUNT); // no need to recreate it    
    }

    eval(board, [x, y], defender) {
        this._planes.fill(0); // just in case

        features(this._planes, board, { x, y }, defender);

        slice(this._fslice, this._planes, [board.size, board.size, F_COUNT],
            [y - WINDOW_HALF, y + WINDOW_HALF],
            [x - WINDOW_HALF, x + WINDOW_HALF],
            [0, F_COUNT - 1]);

        return this._nnfn(this._fslice);
    }
}

function offsetFn(x_size, y_size, z_size) {
    return (x, y, z) => (x * y_size + y) * z_size + z;
}

function slice(res, src, [x_size, y_size, z_size], [xmin, xmax], [ymin, ymax], [zmin, zmax]) {
    const ires = offsetFn(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const isrc = offsetFn(x_size, y_size, z_size);

    for (let x = xmin; x <= xmax; x++) {
        for (let y = ymin; y <= ymax; y++) {
            for (let z = zmin; z <= zmax; z++) {
                const ri = ires(x - xmin, y - ymin, z - zmin);
                const si = isrc(x, y, z);
                const outside = x < 0 || x >= x_size || y < 0 || y >= y_size || z < 0 || z >= z_size;

                res[ri] = outside ? 0 : src[si];
            }
        }
    }
}

function reconstructDCNN(json) {
    const input = nn.value([WINDOW_SIZE ** 2 * F_COUNT]);

    /*

    (3150, 256) align/dense/weights:0
         (256,) align/dense/bias:0
     (256, 256) resb1/1/dense/weights:0
         (256,) resb1/1/dense/bias:0
     (256, 256) resb1/2/dense/weights:0
         (256,) resb1/2/dense/bias:0
     (256, 256) eval/dense/weights:0
         (256,) eval/dense/bias:0
       (256, 1) eval/dense_1/weights:0
           (1,) eval/dense_1/bias:0
     (256, 256) move/dense/weights:0
         (256,) move/dense/bias:0
     (256, 225) move/dense_1/weights:0
         (225,) move/dense_1/bias:0

    */

    function get(name) {
        const v = json.vars[name];
        const w = v && v.data;
        return w;
    }

    function fconn(x, w, b) {
        x = nn.mul(x, w);
        x = nn.add(x, b);
        return x;
    }

    let x = input;

    // the alignment layer
    x = fconn(x,
        get('align/dense/weights:0'),
        get('align/dense/bias:0'));
    x = nn.relu(x);

    // the residual tower
    for (let i = 1; ; i++) {
        const w_name = k => `resb${i}/${k}/dense/weights:0`;
        const b_name = k => `resb${i}/${k}/dense/bias:0`;

        const w1 = get(w_name(1));
        const b1 = get(b_name(1));

        const w2 = get(w_name(2));
        const b2 = get(b_name(2));

        if (!w1) break;

        const y = x;

        x = fconn(x, w1, b1);
        x = nn.relu(x);

        x = fconn(x, w2, b2);
        x = nn.add(x, y);
        x = nn.relu(x);
    }

    // the "value" head to predict the outcome
    let y = fconn(x,
        get('eval/dense/weights:0'),
        get('eval/dense/bias:0'));
    y = nn.relu(y);
    y = fconn(y,
        get('eval/dense_1/weights:0'),
        get('eval/dense_1/bias:0'));
    y = nn.sigmoid(y);

    // the "policy" head to select moves
    let z = fconn(x,
        get('move/dense/weights:0'),
        get('move/dense/bias:0'));
    z = nn.relu(z);
    z = fconn(z,
        get('move/dense_1/weights:0'),
        get('move/dense_1/bias:0'));
    z = nn.softmax(z);

    if (y.size != 1)
        throw Error('Invalid value output: ' + y.shape);

    if (z.size != WINDOW_SIZE ** 2)
        throw Error('Invalid policy output: ' + z.shape);

    return data => {
        input.set(data);
        y.eval();
        z.eval();
        return [y.value[0], z.value];
    };
}

function print(name, size, depth, data) {
    console.log(name + ' = [ // ' + size + 'x' + size + 'x' + depth);
    for (let d = 0; d < depth; d++) {
        console.log('  [ // feature = ' + d);
        for (let y = 0; y < size; y++) {
            let s = '';
            for (let x = 0; x < size; x++)
                s += (data[y * size * depth + x * depth + d] ? '+' : '-') + ' ';
            console.log('    ' + s.slice(0, -1));
        }
        console.log('  ],');
    }
    console.log(']');
}
