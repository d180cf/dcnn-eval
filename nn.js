function operation(shape, deps, eval) {
    const size = shape.reduce((p, n) => p * n, 1);
    const value = new Float32Array(size);
    const op = {};
    op.eval = () => {
        for (const dep of deps)
            dep.eval();
        eval(value, deps.map(dep => dep.value));
        //console.log(moments(value));
    };
    op.value = value;
    op.shape = shape;
    op.size = size;
    return op;
}

function moments(a) {
    let n = a.length;
    let s = 0;

    for (let i = 0; i < n; i++)
        s += a[i];

    let m = s / n;
    let v = 0;

    for (let i = 0; i < n; i++)
        v += (a[i] - m) * (a[i] - m);

    return [m, Math.sqrt(v / n)];
}

const nn = {};

/**
 * @param {number[]} array an array-like object
 * @param {number[]} shape
 */
nn.value = function value(tensor, shape = [tensor.length]) {
    for (const x of shape)
        if (x % 1)
            throw Error('Invalid shape: [' + shape + ']');

    return operation(shape, [], y => {
        for (let i = 0; i < y.length; y++)
            y[i] = tensor[i];
    });
};

/**
 * Multiples a `[n]` vector by a `[n, m]` matrix: `y = x * w`
 * 
 * @param x shape = [n]
 * @param w shape = [n, m]
 * @returns shape = [m] - the result
 */
nn.mul = function mul(x, w) {
    const [n] = x.shape;

    if (w.length) // array-like
        w = nn.value(w, [n, w.length / n]);

    if (w.shape[0] != n)
        throw Error('Incompatible x and w shapes: [' + x.shape + '] and [' + w.shape + ']');

    const [, m] = w.shape;

    return operation([m], [x, w], (y, [x, w]) => {
        for (let i = 0; i < m; i++) {
            let s = 0;

            for (let j = 0; j < n; j++)
                s += x[j] * w[j * m + i];

            y[i] = s;
        }
    });
};

/**
 * Element-wise `z = x + y`.
 */
nn.add = function add(x, y) {
    const n = x.size;

    if (y.length) // array-like
        y = nn.value(y)

    if (n != y.size)
        throw Error('Incompatible x and y shapes: [' + x.shape + '] and [' + y.shape + ']');

    return operation(x.shape, [x, y], (z, [x, y]) => {
        for (let i = 0; i < n; i++)
            z[i] = x[i] + y[i];
    });
};

/**
 * Element-wise `y = f(x)`
 */
nn.map = function map(x, f) {
    const n = x.size;

    return operation(x.shape, [x], (y, [x]) => {
        for (let i = 0; i < n; i++)
            y[i] = f(x[i]);
    });
};

/**
 * Element-wise `y = max(0, x)`
 */
nn.relu = function relu(x) {
    return nn.map(x, x => Math.max(0, x));
};

/**
 * Element-wise `y = 1/(1 + exp(-x))`
 */
nn.sigmoid = function sigmoid(x) {
    return nn.map(x, x => 1 / (1 + Math.exp(-x)));
};

module.exports = nn;
