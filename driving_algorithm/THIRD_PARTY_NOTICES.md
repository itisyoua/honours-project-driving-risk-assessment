# Third-Party Notices

## Waymo Open Dataset Protobuf Definitions

- Source: <https://github.com/waymo-research/waymo-open-dataset>
- Revision: `99a4cb3ff07e2fe06c2ce73da001f850f628e45a`
- License: Apache License 2.0
- License text: <https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/LICENSE>

The `waymo_open_dataset/**/*_pb2.py` files are generated from the official
protobuf sources by:

```bash
.venv/bin/python scripts/bootstrap_waymo_protos.py --output-root .
```

Only the entry definition
`waymo_open_dataset/protos/end_to_end_driving_data.proto` and its recursively
imported dependencies are compiled. The generated modules are used solely to
read the non-commercial research dataset under the Waymo Dataset License.
