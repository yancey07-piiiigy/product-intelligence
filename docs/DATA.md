# Data explainer

## Submitted product data

The project uses a fixed catalog of **2,000 footwear products**. Each record has one catalog image and the fields required by the application: product ID, article type, base colour, catalog audience (`gender` in the source schema), season, and usage.

The submitted catalog was selected deterministically with seed 6201 and proportional stratification by article type and catalog audience. Known products required by the frozen evaluation were retained, while products assigned to the unknown-footwear condition were excluded from the searchable catalog. The selection record is `data/gallery-subset.json`; the application-ready records are `artifacts/index.json`, and the corresponding images are in `data/gallery-images/`.

## Source and inclusion policy

The catalog was derived from Param Aggarwal's **Fashion Product Images Dataset** distributed through Kaggle: <https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset>. The original Kaggle archive is not included in this project submission. Only the 2,000-item course subset needed to run ShoeLens is checked in. Source import metadata and integrity information are retained in:

- `data/DATASET_IMPORT.json`
- `data/import-report.json`
- `data/manifest.json`
- `data/gallery-subset.json`

The upstream dataset's redistribution terms should be checked before publishing the images outside the course submission. The data is treated as catalog evidence, not as proof of material, authenticity, stock, price, or specifications that are not present in its fields.

## Evaluation data

The frozen robustness protocol contains 100 queries:

| Split | Known catalog shoes | Unknown footwear | Non-footwear | Total |
|---|---:|---:|---:|---:|
| Calibration and test combined | 70 | 15 | 15 | 100 |
| Held-out test | 42 | 9 | 9 | 60 |

The remaining 40 queries form the calibration set. Exact membership and hashes are recorded in `artifacts/protocol.json` and `artifacts/manifest.json`; images are stored in `artifacts/queries/`.

Known-item queries are deterministic crops, rotations, lighting changes, and colour changes derived from catalog-origin images. Unknown footwear products are excluded from the 2,000-item gallery, and non-footwear images test local refusal. Development image IDs and hashes were excluded from the final query list.

This design supports a repeatable closed-gallery robustness test. It does **not** measure performance on independent phone photographs, new viewpoints, cluttered real backgrounds, or shoes from a different source distribution.

## User-study data

`artifacts/user-study.csv` contains verified participant observations from five anonymized participant codes and 20 complete paired tasks. Each pair compares manual browsing of 20 fixed, visually similar candidates with ShoeLens search over the 2,000-product catalog.

Success requires the correct product, season, and usage. Incorrect answers receive the prespecified 180-second penalty in the paired-time analysis. `artifacts/user-study-provenance.json` links the project owner's provenance confirmation to the analyzed CSV hash. No participant names, contact details, or other direct identifiers are stored.

The study task images are also transformed catalog photos. They are not independent real-world phone photos. Administration and analysis rules are in `study/README.md` and `data/user-study-tasks.json`.

## AI-description review data

`artifacts/visual-human-review.csv` records the project owner's review of nine descriptions for uncertain footwear images. Four visible fields were checked per image: shoe type, colour, closure, and visible details. `artifacts/visual-human-review-provenance.json` records the scope and CSV hash.

The review contains 36 field judgments and reports all 36 as correct, with no unsupported claim reported. It is a small owner-attested review of transformed catalog photos, not an independent blinded study.

## Reproducibility and integrity

Key manifests contain fixed random seeds, selected IDs, file hashes, and model/index signatures. These make accidental changes detectable and keep calibration separate from the held-out test. Generated API caches and local runtime logs are excluded from submission because they are not source data and may contain environment-specific information.
