# Training

This folder contains the reproducible training, export, and static gallery generation tools.

Expected local data layout:

```text
train_validation/
  train/<class>/*.jpg
  validation/<class>/*.jpg
```

The raw dataset is intentionally ignored by git.

Smoke test:

```powershell
.\.venv\Scripts\python.exe training\train.py --data-dir ..\train_validation --mode smoke --output-dir runs\smoke
```

Search:

```powershell
.\.venv\Scripts\python.exe training\train.py --data-dir ..\train_validation --mode search --output-dir runs\search --batch-size 40 --num-workers 4 --search-epochs 1
```

Final:

```powershell
.\.venv\Scripts\python.exe training\train.py --data-dir ..\train_validation --mode final --output-dir runs\final --final-model convnext_tiny --final-img-size 288 --final-epochs 10 --batch-size 40 --num-workers 4
```

