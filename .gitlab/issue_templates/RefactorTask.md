## Summary

Describe the purpose of this refactor/feature (one idea per issue).  
Link to relevant section of the Wiki roadmap:

👉 [[PyFuga Refactor Roadmap]]  


## Branch

Create the branch from `release/beta`:

```bash
git checkout release/beta
git checkout -b <branch-name>
```

Suggested branch name:
`<feature|refactor>/<short-name>`

## Scope of work

- [ ] Extract only the relevant changes from `refactor-and-docs`
- [ ] Adapt the code to the current structure on `release/beta`
- [ ] Keep changes focused and behaviour-preserving unless otherwise agreed
- [ ] Add or update unit tests as needed
- [ ] Ensure CI passes
- [ ] Open MR and request feedback from Leonardo
- [ ] Decide whether it belongs in:
    - `release/beta`, or 
    - `release/v1.0.0` (future public release)
- [ ] Merge or park

## Notes / Design considerations

(Add context, blockers, open questions, or decisions needed from Leonardo/Elvira)

## Links

- Roadmap: [[PyFuga Refactor Roadmap]]
- MR: (add when created)
- Related issues: (add as needed)
