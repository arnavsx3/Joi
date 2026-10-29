# Working on Joi

- Commit and push each completed, coherent chunk of changes to `origin`, as
  requested by the project owner. Check the remote state before pushing.
- Use the project's uv environment and CPU defaults. Keep training runs small
  enough for the development laptop unless a larger experiment is requested.
- Verify model or training changes with relevant tests and record measured
  results honestly. Documentation-only changes need a formatting/diff check.
- Keep environments, downloaded data, and checkpoints out of Git as specified
  in `.gitignore`. Small reviewed experiment summaries and figures can be saved
  under `evaluation/` and `docs/`.
