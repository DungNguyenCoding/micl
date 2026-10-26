# Historical documents — preserved, not current defaults

`pre_refresh_20261026/` contains exact byte copies of all nine supplied root files and all five Markdown files from docs.zip. It also keeps the older command guide and multicast-transition guideline available in the conversation. No historical content has been silently corrected in those copies.

The original files can contain old “official/current” language, unicast equations, MLP/N100 or ResNet profiles, old accuracy/test counts, and code-preservation claims that were true only of their release context. They are not authority over the current [START_HERE](../START_HERE.md), [locks](../LOCKED_EXPERIMENTS.md) or latest user request.

Root CHANGE_MANIFEST.json and PACKAGE_MANIFEST.txt remain untouched because their hashes and September package listing are historical provenance. Dependency files also remain untouched; they are executable packaging/environment declarations, not prose to rewrite during a documentation cleanup.

The installer separately backs up the actual live files it replaces, outside the project. Those live backups may differ in newline encoding from the uploaded copies; both are preserved. This history directory is not a model/dataset backup.
