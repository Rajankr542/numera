Per-milestone API docs (D-056). Each `pNN.mjs` exports `categories` (same
shape as `docs/site/api.mjs`); entries in a category whose `id` matches an
existing one are appended to it, new ids become new categories.
