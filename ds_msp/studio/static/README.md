# Bundled Studio interface

`npm run build:studio` in `web/` writes the built interface here. The built files are
git-ignored and are added to the wheel by the release build. Without them,
`ds-msp studio` serves a page that says how to build them, and the JSON API still works.
