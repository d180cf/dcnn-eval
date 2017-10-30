/**
 * Implements Python style format function:
 * https://pyformat.info/
 * 
 * Supported formats:
 * 
 *  - format("{0} is {1}", 234, "even") = "234 is even"
 *  - format("{0:5} is {1}", 234, "even") = "234   is even"
 *  - format("{0:>5} is {1}", 234, "even") = "  234 is even"
 */

module.exports = function format(text, ...args) {
    return text

        // regular format: "{0}"
        .replace(/{(\d+)}/gm, (s, i) => args[+i])

        // padding on the right: {0:5}
        .replace(/{(\d+):(\d+)}/gm, (s, i, n) => (args[+i] + ' '.repeat(+n)).slice(0, +n))

        // padding on the left: {0:>5}
        .replace(/{(\d+):>(\d+)}/gm, (s, i, n) => (' '.repeat(+n) + args[+i]).slice(-+n));
};
