# Flask inference demo

This local website calls the shared inference module in `src/melanoma_classifier`. Install the package from the repository root, put a trained checkpoint at `artifacts/best_model.pth`, then run `python web/app.py`. Set `MELANOMA_CHECKPOINT` to use a different checkpoint path.
