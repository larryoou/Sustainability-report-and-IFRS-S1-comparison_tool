# IFRS S1 Dual-Model Accelerated Package

This package bundles the IFRS S1 ESG analysis tool with the accelerated dual-model backend.

It runs entirely on your local machine:
- Backend (accelerated dual-model retrieval) on port 8004
- Optional local semantic proxy on port 8000
- Frontend served on port 9000

## Quick Start

1) Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

2) Start all services and the frontend (recommended)

```bash
bash scripts/start.sh
```

Then open:

- Frontend: http://localhost:9000/ifrs_s1_auto_keywords_tool.html
- Backend health: http://localhost:8004/health
- Local proxy health: http://localhost:8000/health

To stop all services:

```bash
bash scripts/stop.sh
```

## Package Structure

- `backend/accelerated_dual_model_service.py`
  Accelerated hybrid retrieval service (FAISS + Sentence-Transformers) exposing endpoints:
  - POST `/accelerated_similarity`
  - POST `/accelerated_batch_analysis`
  - POST `/analyze_specific_article`
  - GET `/health`, `/models`, `/performance`

- `backend/faiss_vector_service.py`
  FAISS-based retrieval core, MPS/GPU-optimized Sentence-Transformers embedding with intelligent dimension adaptation.

- `backend/ifrs_s1_articles_data.py`
  Full IFRS S1 articles dataset used to initialize the accelerated service (excludes IFRS-S1-20 as required).

- `backend/local_semantic_service.py`
  Optional FastAPI-based local semantic proxy on port 8000. Configured in proxy mode to prefer the accelerated backend.

- `frontend/ifrs_s1_auto_keywords_tool.html`
  Main UI. Endpoints are already set to use port 8004 for dual-model analysis and recommendation generation.

- `frontend/ifrs_s1_complete_articles.js`
  Frontend data source of IFRS S1 articles for rendering and reporting.

- `requirements.txt`
  Python dependencies for both services.

- `scripts/start.sh`, `scripts/stop.sh`
  Convenience scripts to start/stop services and the local HTTP server.

- `logs/`, `vector_cache/`
  Runtime logs and FAISS/vector caches.

## Notes & Tips

- Apple Silicon (MPS) and NVIDIA GPUs are supported via PyTorch when available; otherwise CPU is used. The FAISS index uses inner-product with normalized vectors.
- If FAISS or Sentence-Transformers cannot initialize, the service degrades gracefully to TF‑IDF fallback methods.
- The frontend will fetch PDF.js from CDN if local assets are not present. For fully offline usage, place `pdf.min.js` and `pdf.worker.min.js` under `frontend/assets/pdfjs/` and the page will use those automatically.
- Known dependency note: to avoid compatibility issues, NumPy is pinned to `< 2.0.0`.

## Ports

- 8004: Accelerated backend (required)
- 8000: Local semantic proxy (optional)
- 9000: Frontend HTTP server

## Troubleshooting

- If you see FAISS import issues on macOS/Apple Silicon, try installing `faiss-cpu` again or use Conda:
  - `python3 -m pip install faiss-cpu`
  - or `conda install -c pytorch faiss-cpu`
- If you see Torch/MPS issues, ensure a recent PyTorch is installed that supports MPS.
- Check `logs/accelerated_service.log` and `logs/local_service.log` for detailed errors.
