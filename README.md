# Tomato disease burden under INT8 quantization: dashboard

DAP391m Final Project, class AI2002, Group 1 (FPT University, Fall 2026).
Static version of the project dashboard, served by GitHub Pages; it needs no server.

* **Results explorer**: stored test results of MNv4-FPN and five comparison configurations (176 PlantSeg tomato test
  images, three seeds, FP32 and INT8). Four filters, six KPI cards, eight linked views grouped by research question,
  one insight sentence per view, a heatmap click that filters the page, and an image drill-down with the reference,
  FP32 and INT8 masks.
* **Burden estimator**: upload a tomato leaf or fruit photo or pick a validation sample. MNv4-FPN (seed 20260924) runs
  in the browser with ONNX Runtime Web in INT8 and FP32 and returns the masks, the burden, the FP32 to INT8 drift and
  alerts. The first run downloads the models (about 38 MB).

Images: PlantSeg (Wei et al., 2024), tomato subset. Code and paper: https://github.com/Chonf-pkz/MNv4-FPN
