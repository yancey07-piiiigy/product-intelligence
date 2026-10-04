# Similar-shoe user study materials

Eight frozen target photos are in `images/`. Their filenames are task IDs (`C_A.jpg`, `F_B.jpg`, etc.); none reveals the correct catalog ID. The participant-facing task order is generated with `python user_study.py --schedule P01` (replace `P01` for each anonymous participant). The output gives the image path, shared catalog clues and the dedicated manual-browsing URL when applicable.

Run the local ShoeLens server before opening a manual URL. For an assistant trial, open the regular ShoeLens homepage and let the participant upload the scheduled image. For a manual trial, open the scheduled `?study=manual&task=...` URL; the page shows the target and 20 fixed similar candidates without name/ID search.

Record real outcomes using a copy of `artifacts/user-study-template.csv`. The detailed timing, counterbalancing, correctness and analysis rules are in `docs/USER_STUDY.md`. The answer key is `data/user-study-tasks.json`; keep it away from participants. These images come from transformed catalog photos and should not be described as independent real-world photographs.
