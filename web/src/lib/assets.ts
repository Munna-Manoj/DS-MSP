// Resolve bundled public assets against the Vite base URL, so they load both in
// dev ("/") and under the GitHub Pages project subpath ("/DS-MSP/studio/").
export const ENV_URL = import.meta.env.BASE_URL + "env.jpg";
