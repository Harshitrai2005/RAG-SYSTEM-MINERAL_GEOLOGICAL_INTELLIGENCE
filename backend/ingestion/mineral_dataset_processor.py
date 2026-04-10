"""
Mineral Dataset Processor
Handles structured geochemical and mineral composition datasets.
Supports CSV (assay tables, geochemical surveys) and JSON formats.

Geochemical datasets typically contain:
- Sample coordinates (X, Y, Z / Easting, Northing, Elevation)
- Element concentrations (Au, Cu, Zn, Pb, Ag, Mo, As, Sb, etc.)
- Rock/sample codes, lithology labels, alteration codes
- Downhole assay intervals from drill programs
"""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config import settings
from utils.logger import setup_logger
from utils.text_chunker import TextChunker

logger = setup_logger(__name__)


# ── Geochemical Thresholds ────────────────────────────────────────────────────
# Background vs anomalous concentrations (ppm unless noted)
# Values are indicative — always calibrate against regional background

GEOCHEMICAL_THRESHOLDS = {
    "Au": {"anomalous": 0.05, "high": 0.5, "unit": "ppm"},        # Gold
    "Cu": {"anomalous": 100, "high": 1000, "unit": "ppm"},         # Copper
    "Zn": {"anomalous": 200, "high": 1000, "unit": "ppm"},         # Zinc
    "Pb": {"anomalous": 50, "high": 500, "unit": "ppm"},           # Lead
    "Ag": {"anomalous": 1, "high": 10, "unit": "ppm"},             # Silver
    "Mo": {"anomalous": 10, "high": 100, "unit": "ppm"},           # Molybdenum
    "As": {"anomalous": 20, "high": 200, "unit": "ppm"},           # Arsenic (pathfinder)
    "Sb": {"anomalous": 5, "high": 50, "unit": "ppm"},             # Antimony (pathfinder)
    "Te": {"anomalous": 0.5, "high": 5, "unit": "ppm"},            # Tellurium (pathfinder)
    "W": {"anomalous": 5, "high": 50, "unit": "ppm"},              # Tungsten
    "Bi": {"anomalous": 1, "high": 20, "unit": "ppm"},             # Bismuth (pathfinder)
    "Fe": {"anomalous": 5, "high": 15, "unit": "%"},               # Iron
    "Mn": {"anomalous": 500, "high": 2000, "unit": "ppm"},         # Manganese
}

# Classic pathfinder element suites by deposit type
DEPOSIT_PATHFINDERS = {
    "epithermal_Au": ["Au", "Ag", "As", "Sb", "Te", "Tl", "Hg"],
    "porphyry_Cu": ["Cu", "Mo", "Au", "Re", "W"],
    "VMS": ["Cu", "Zn", "Pb", "Ag", "Ba", "S"],
    "SEDEX": ["Zn", "Pb", "Ag", "Ba", "Tl"],
    "IOCG": ["Cu", "Au", "Fe", "U", "Bi", "Co"],
    "pegmatite": ["Li", "Be", "Cs", "Ta", "Nb", "Sn"],
}


class MineralDatasetProcessor:
    """
    Processes geochemical datasets (CSV/JSON) into enriched text summaries
    suitable for embedding and semantic retrieval.

    Each dataset is summarized at multiple granularities:
    - Dataset-level overview (statistics, anomaly counts)
    - Sample group summaries (grouped by lithology/zone)
    - Individual high-priority sample descriptions
    """

    def __init__(self):
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def process_file(self, file_path: str | Path) -> list[dict]:
        """Process a single CSV or JSON geochemical dataset."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".json":
            df = self._load_json_dataset(file_path)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

        logger.info(f"Loaded dataset: {file_path.name} — {len(df)} rows × {len(df.columns)} cols")

        file_hash = self._hash_path(file_path)
        chunks = []

        # 1. Overall dataset summary
        overview = self._generate_dataset_overview(df, file_path.name)
        chunks.append(
            {
                "id": f"{file_hash}_overview",
                "text": overview,
                "metadata": {
                    "source": file_path.name,
                    "doc_type": "geochemical_dataset",
                    "file_type": suffix[1:],
                    "rows": len(df),
                    "section": "Dataset Overview",
                },
            }
        )

        # 2. Anomaly summary
        anomaly_text = self._identify_geochemical_anomalies(df)
        if anomaly_text:
            chunks.append(
                {
                    "id": f"{file_hash}_anomalies",
                    "text": anomaly_text,
                    "metadata": {
                        "source": file_path.name,
                        "doc_type": "geochemical_dataset",
                        "file_type": suffix[1:],
                        "section": "Geochemical Anomalies",
                    },
                }
            )

        # 3. Deposit model interpretation
        model_text = self._interpret_deposit_model(df)
        if model_text:
            chunks.append(
                {
                    "id": f"{file_hash}_deposit_model",
                    "text": model_text,
                    "metadata": {
                        "source": file_path.name,
                        "doc_type": "geochemical_dataset",
                        "file_type": suffix[1:],
                        "section": "Deposit Model Interpretation",
                    },
                }
            )

        # 4. High-grade sample table
        high_grade_text = self._summarize_high_grade_samples(df, file_path.name)
        if high_grade_text:
            chunks.append(
                {
                    "id": f"{file_hash}_high_grade",
                    "text": high_grade_text,
                    "metadata": {
                        "source": file_path.name,
                        "doc_type": "geochemical_dataset",
                        "file_type": suffix[1:],
                        "section": "High-Grade Samples",
                    },
                }
            )

        logger.info(f"Generated {len(chunks)} chunks from dataset {file_path.name}")
        return chunks

    def _load_json_dataset(self, file_path: Path) -> pd.DataFrame:
        """Load JSON — handles both array-of-objects and nested formats."""
        with open(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try common nested keys
            for key in ["samples", "data", "records", "assays", "results"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
        return pd.DataFrame()

    def _generate_dataset_overview(self, df: pd.DataFrame, filename: str) -> str:
        """Generate a summary of the full dataset."""
        element_cols = [c for c in df.columns if c.upper() in GEOCHEMICAL_THRESHOLDS]
        coord_cols = [c for c in df.columns if c.lower() in {"x", "y", "z", "easting", "northing", "elevation", "depth"}]

        lines = [
            f"GEOCHEMICAL DATASET: {filename}",
            f"Total samples: {len(df)}",
            f"Columns: {', '.join(df.columns.tolist())}",
        ]

        if coord_cols:
            lines.append(f"Coordinate columns detected: {', '.join(coord_cols)}")

        if element_cols:
            lines.append(f"\nGEOCHEMICAL ELEMENTS PRESENT: {', '.join(element_cols)}")
            lines.append("\nELEMENT STATISTICS:")
            for elem in element_cols:
                col = pd.to_numeric(df[elem], errors="coerce")
                col = col.dropna()
                if len(col) > 0:
                    lines.append(
                        f"  {elem}: mean={col.mean():.3f}, "
                        f"median={col.median():.3f}, "
                        f"max={col.max():.3f}, "
                        f"std={col.std():.3f} "
                        f"({GEOCHEMICAL_THRESHOLDS.get(elem.upper(), {}).get('unit', 'ppm')})"
                    )

        if "lithology" in [c.lower() for c in df.columns]:
            lith_col = next(c for c in df.columns if c.lower() == "lithology")
            lith_counts = df[lith_col].value_counts().head(10)
            lines.append(f"\nLITHOLOGY DISTRIBUTION:")
            for lith, count in lith_counts.items():
                lines.append(f"  {lith}: {count} samples ({100*count/len(df):.1f}%)")

        return "\n".join(lines)

    def _identify_geochemical_anomalies(self, df: pd.DataFrame) -> str:
        """Flag samples exceeding anomalous thresholds for each element."""
        element_cols = [c for c in df.columns if c.upper() in GEOCHEMICAL_THRESHOLDS]
        if not element_cols:
            return ""

        lines = ["GEOCHEMICAL ANOMALY ANALYSIS:"]
        found_anomalies = False

        for col in element_cols:
            elem_upper = col.upper()
            thresholds = GEOCHEMICAL_THRESHOLDS.get(elem_upper, {})
            if not thresholds:
                continue

            numeric_col = pd.to_numeric(df[col], errors="coerce")
            anomalous = numeric_col[numeric_col >= thresholds["anomalous"]]
            high = numeric_col[numeric_col >= thresholds["high"]]

            if len(anomalous) > 0:
                found_anomalies = True
                lines.append(
                    f"  {elem_upper}: {len(anomalous)} anomalous samples "
                    f"(>{thresholds['anomalous']} {thresholds['unit']}), "
                    f"{len(high)} high-grade (>{thresholds['high']} {thresholds['unit']}), "
                    f"peak = {numeric_col.max():.3f} {thresholds['unit']}"
                )

        if not found_anomalies:
            return ""

        return "\n".join(lines)

    def _interpret_deposit_model(self, df: pd.DataFrame) -> str:
        """
        Score dataset against known deposit type pathfinder suites.
        Returns the most likely deposit interpretation.
        """
        element_cols = {c.upper() for c in df.columns if c.upper() in GEOCHEMICAL_THRESHOLDS}
        scores = {}

        for deposit_type, pathfinders in DEPOSIT_PATHFINDERS.items():
            matches = [p for p in pathfinders if p in element_cols]
            if len(matches) >= 2:
                scores[deposit_type] = len(matches) / len(pathfinders)

        if not scores:
            return ""

        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        lines = ["DEPOSIT MODEL INTERPRETATION (based on element suite):"]
        for model, score in sorted_models[:3]:
            confidence = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            pathfinders = DEPOSIT_PATHFINDERS[model]
            present = [p for p in pathfinders if p in element_cols]
            lines.append(
                f"  {model.replace('_', ' ')}: {confidence} confidence ({score:.0%} pathfinder match) "
                f"— elements present: {', '.join(present)}"
            )

        return "\n".join(lines)

    def _summarize_high_grade_samples(
        self, df: pd.DataFrame, filename: str, max_samples: int = 20
    ) -> str:
        """Extract and describe the highest-grade samples in the dataset."""
        precious_metals = [c for c in df.columns if c.upper() in {"AU", "AG", "PT", "PD"}]
        base_metals = [c for c in df.columns if c.upper() in {"CU", "ZN", "PB", "NI", "CO"}]
        target_cols = precious_metals + base_metals

        if not target_cols:
            return ""

        # Find the primary value column (most likely Au or Cu)
        primary_col = precious_metals[0] if precious_metals else base_metals[0]
        numeric_primary = pd.to_numeric(df[primary_col], errors="coerce")

        top_samples = df[numeric_primary.notna()].copy()
        top_samples[primary_col] = pd.to_numeric(top_samples[primary_col], errors="coerce")
        top_samples = top_samples.nlargest(max_samples, primary_col)

        lines = [f"TOP {len(top_samples)} HIGHEST-GRADE SAMPLES (ranked by {primary_col}):"]
        for _, row in top_samples.iterrows():
            sample_vals = []
            for col in target_cols:
                val = pd.to_numeric(row.get(col, None), errors="coerce")
                if pd.notna(val) and val > 0:
                    unit = GEOCHEMICAL_THRESHOLDS.get(col.upper(), {}).get("unit", "ppm")
                    sample_vals.append(f"{col}={val:.3f}{unit}")
            if sample_vals:
                lines.append(f"  Sample: {', '.join(sample_vals)}")

        return "\n".join(lines)

    def _hash_path(self, file_path: Path) -> str:
        return hashlib.md5(str(file_path).encode()).hexdigest()[:12]
