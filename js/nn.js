function operation(shape, deps, eval, init) {
    for (const x of shape)
        if (x % 1)
            throw Error('Invalid shape: [' + shape + ']');    

    const size = shape.reduce((p, n) => p * n, 1);
    const value = new Float32Array(size);
    const op = {};    
    op.eval = () => {
        for (const dep of deps)
            dep.eval();
        eval(value, deps.map(dep => dep.value));
        //console.log(moments(value));
    };
    op.set = init => {
        if (init.length != size)
            throw Error('Invalid initializer size: ' + init.length);
        value.set(init);
    };
    op.value = value;
    op.shape = shape;
    op.size = size;
    init && op.set(init);
    return op;
}

const nn = {};

/**
 * Computes `[mean, variance]`.
 */
nn.moments = function moments(a) {
    let n = a.size;

    return operation([2], [a], (r, [a]) => {
        let s = 0;

        for (let i = 0; i < n; i++)
            s += a[i];

        let m = s / n;
        let v = 0;

        for (let i = 0; i < n; i++)
            v += (a[i] - m) * (a[i] - m);

        r[0] = m;
        r[1] = Math.sqrt(v / n);
    });
};

/**
 * @param {number[]} shape
 * @param {number[]} input optional array-like initial value
 */
nn.value = function value(shape, input) {
    return operation(shape, [], y => { }, input);
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
        w = nn.value([n, w.length / n], w);

    if (w.shape[0] != n)
        throw Error('Incompatible x and w shapes: [' + x.shape + '] and [' + w.shape + ']');

    const [, m] = w.shape;

    return operation([m], [x, w], (y, [x, w]) => {
        for (let i = 0; i < m; i++) {
            y[i] = 0;

            for (let j = 0; j < n; j++)
                y[i] += x[j] * w[j * m + i];
        }
    });
};

/**
 * Element-wise `z = x + y`.
 */
nn.add = function add(x, y) {
    const n = x.size;

    if (y.length) // array-like
        y = nn.value(x.shape, y);

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

/**
 * Element-wise `y = exp(x) / sum exp(x)`
 */
nn.softmax = function softmax(x) {
    const n = x.size;
    const e = nn.map(x, x => Math.exp(x));
    const s = nn.moments(e);

    return operation(x.shape, [e, s], (y, [e, s]) => {
        const sum = s[0] * n; // mean * size

        for (let i = 0; i < n; i++)
            y[i] = e[i] / sum;
    });
};

module.exports = nn;
