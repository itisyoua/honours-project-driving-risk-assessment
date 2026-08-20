# Data sources and storage

## comma2k19

comma2k19 is a public driving dataset released by comma.ai. It contains 2,019
one-minute segments (more than 33 hours of driving) and is approximately 100 GB
in total. The upstream project divides the data into ten chunks of roughly
10 GB each.

- Project and documentation: <https://github.com/commaai/comma2k19>
- Dataset download: <http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb>
- Paper: <https://arxiv.org/abs/1812.05752>

After downloading and extracting all chunks, use this local layout:

```text
honours-project-driving-risk-assessment/
├── comma2k19/
│   ├── Chunk_1/
│   ├── Chunk_2/
│   ├── ...
│   └── Chunk_10/
└── comma2k19_data_preparation/
```

The raw `comma2k19/` directory is ignored by Git because GitHub enforces a
100 MiB limit for ordinary Git objects, a 2 GB push limit, and recommends that
repositories remain below 5 GB. Keeping the source data outside Git also avoids
duplicating the authoritative public distribution. The derived manifests,
splits, validation reports and preview assets remain versioned in
`comma2k19_data_preparation/`.

## Citation

If this dataset is used in a publication, cite the upstream work:

```bibtex
@misc{1812.05752,
  author = {Harald Schafer and Eder Santana and Andrew Haden and Riccardo Biasini},
  title = {A Commute in Data: The comma2k19 Dataset},
  year = {2018},
  eprint = {arXiv:1812.05752}
}
```

## Other raw trajectory data

The local `raw_data/` directory is also excluded because it is approximately
6 GB and contains third-party source datasets. The small scan summaries in
`processed_data/` are tracked, while each source dataset must be obtained under
its own licence and terms.
