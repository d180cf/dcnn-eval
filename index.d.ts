/**
 * @param {tsumego.Board} board
 * @param coords The [x, y] coords of the target.
 * @param defstarts 1 if the target starts; 0 otherwise.
 * @returns Prediction of the outcome and prediction of best moves.
 */
declare function evaldcnn(board, coords: [number, number], defstarts: number): [number, number[]];