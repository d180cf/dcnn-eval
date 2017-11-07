// tests how fast JS V8 does multiplications-additions
// so far the results are impressive: 48 G ops per second
// where 1 op = 1 mul + 1 add

const k = 500;
const n = 1 << 20; // 1 M elements
const a = new Array(10);

for (let i = 0; i < a.length; i++) {
    a[i] = new Float32Array(n);
    for (let j = 0; j < n; j++)
        a[i][j] = Math.random() - 0.5;
}

function mul(x, y) {
    let n = x.length;
    let s = 0.0;

    for (let i = 0; i < n; i += 8) {
        let s0 = x[i + 0] * y[i + 0];
        let s1 = x[i + 1] * y[i + 1];
        let s2 = x[i + 2] * y[i + 2];
        let s3 = x[i + 3] * y[i + 3];
        let s4 = x[i + 4] * y[i + 4];
        let s5 = x[i + 5] * y[i + 5];
        let s6 = x[i + 6] * y[i + 6];
        let s7 = x[i + 7] * y[i + 7];

        s += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    }

    return s;
}

let p_min = +Infinity;
let p_max = -Infinity;
let p_sum = 0; // for p_avg

for (let i = 0; i < k; i++) {
    const x = a[Math.random() * a.length | 0];
    const y = a[Math.random() * a.length | 0];
    const t = Date.now();
    const s = mul(x, y);
    const p = n / (Date.now() - t) / 1000;

    p_min = Math.min(p_min, p);
    p_max = Math.max(p_max, p);
    p_sum += p;

    if (i % 50 == 0)
        console.log(`min = ${p_min.toFixed(1)} M/s; max = ${p_max.toFixed(1)} M/s; avg = ${(p_sum / (i + 1)).toFixed(1)} M/s`);
}
