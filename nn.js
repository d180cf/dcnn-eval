function tensor(n) {
    return new Float32Array(n);
}

/**
 * Multiples a vector by a matrix.
 * 
 * @param x shape = [n, 1]
 * @param w shape = [n, m]
 * @returns shape = [m, 1]
 */
exports.mul = function mul(x, w) {
    const n = x.length;
    const m = w.length / n;

    if (m % 1)
        throw Error('Incompatible x and w shapes: ' + x.length + ' and ' + w.length);

    const y = tensor(m);

    // for (...)

    return y;
};

exports.add = function add(x, y) {
    const n = x.length;

    if (n != y.length)
        throw Error('Incompatible x and y shapes: ' + x.length + ' and ' + y.length);

    const z = tensor(n);

    for (let i = 0; i < n; i++)
        z[i] = x[i] + y[i];

    return z;
};

exports.relu = function relu(x) {
    const n = x.length;
    const y = tensor(n);

    for (let i = 0; i < n; i++)
        y[i] = Math.max(0, x[i]);

    return y;
};

exports.sigmoid = function sigmoid(x) {
    const n = x.length;
    const y = tensor(n);

    for (let i = 0; i < n; i++)
        y[i] = 1 / (1 + Math.exp(-x[i]));

    return y;
};

