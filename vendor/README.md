# Vendored dependencies

## mediabunny

- Package: `mediabunny@1.65.2` (MIT-licensed peer: MPL-2.0, see
  https://github.com/Vanilagy/mediabunny)
- Files: `mediabunny.mjs` (minified ESM bundle) and `mediabunny.d.ts` (types,
  used for editor and `tsc --checkJs` support).
- Why vendored: the app is bundler-free; a pinned local copy keeps runs
  reproducible and offline-capable (same reasoning as the committed `pkg/`).

Upgrade:

```sh
npm install mediabunny@<new-version>
npm run vendor
```

The vendored file must not be edited in place (MPL-2.0 requires modifications
to the library itself to be published; we only consume it).
