// One colour and one name per board, the same in every view. The name uses the index the result
// files use (board_idxs). Green and red are left out: on a photo they mean a corner's error.
const COLORS = ["#38bdf8", "#fb923c", "#c084fc", "#f472b6", "#facc15", "#e2e8f0", "#818cf8", "#d6a77a"];

export const boardColor = (id: number) => COLORS[((id % COLORS.length) + COLORS.length) % COLORS.length];
export const boardName = (id: number) => `Board ${id}`;
