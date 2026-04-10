"""
Image & Hyperspectral Processor
Handles satellite imagery and hyperspectral data for mineral mapping.

Hyperspectral data contains hundreds of narrow spectral bands — each pixel
is effectively a spectral "fingerprint" that can identify mineral assemblages
based on absorption features.

This processor:
- Extracts spectral statistics for known mineral absorption bands
- Identifies spectral anomalies suggestive of alteration zones
- Generates textual summaries ready for embedding in the vector store
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import spectral
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False

from core.config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Mineral Spectral Library ──────────────────────────────────────────────────
# Key absorption wavelengths (in nm) for common hydrothermal minerals
# Source: USGS Spectral Library, Clark et al. 2007

MINERAL_SPECTRAL_SIGNATURES = {
    "kaolinite": {
        "absorption_bands": [1395, 1415, 2160, 2205],
        "diagnostic_band": 2205,
        "alteration_type": "argillic",
        "deposit_association": ["epithermal", "porphyry_Au_Cu"],
    },
    "alunite": {
        "absorption_bands": [1480, 2165, 2320],
        "diagnostic_band": 2165,
        "alteration_type": "advanced_argillic",
        "deposit_association": ["high_sulfidation_epithermal"],
    },
    "illite_muscovite": {
        "absorption_bands": [1400, 2200, 2350],
        "diagnostic_band": 2200,
        "alteration_type": "phyllic",
        "deposit_association": ["porphyry_Cu", "mesothermal_Au"],
    },
    "chlorite": {
        "absorption_bands": [1395, 2250, 2335],
        "diagnostic_band": 2335,
        "alteration_type": "propylitic",
        "deposit_association": ["porphyry_Cu", "VMS"],
    },
    "calcite_dolomite": {
        "absorption_bands": [2330, 2500],
        "diagnostic_band": 2330,
        "alteration_type": "carbonate",
        "deposit_association": ["skarn", "MVT", "SEDEX"],
    },
    "iron_oxide_goethite": {
        "absorption_bands": [490, 650, 900],
        "diagnostic_band": 900,
        "alteration_type": "iron_oxide",
        "deposit_association": ["IOCG", "gossanous_zones"],
    },
    "jarosite": {
        "absorption_bands": [430, 900, 2265],
        "diagnostic_band": 2265,
        "alteration_type": "supergene",
        "deposit_association": ["high_sulfidation_epithermal", "gossans"],
    },
}


class HyperspectralProcessor:
    """
    Processes hyperspectral and satellite imagery into mineral intelligence summaries.

    For true hyperspectral files (ENVI format via spectral library), performs
    spectral angle mapping (SAM) against the mineral signature library.

    For standard images (RGB/multispectral), extracts band ratios and color
    statistics commonly used in mineral exploration.
    """

    def process_file(self, file_path: str | Path) -> list[dict]:
        """
        Process a hyperspectral or image file.

        Returns document chunks with textual mineral analysis summaries
        ready for vector store embedding.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in [".hdr"] and SPECTRAL_AVAILABLE:
            return self._process_envi_hyperspectral(file_path)
        elif suffix in [".tif", ".tiff"]:
            return self._process_geotiff(file_path)
        elif suffix in [".png", ".jpg", ".jpeg"] and PIL_AVAILABLE:
            return self._process_standard_image(file_path)
        elif suffix == ".json":
            return self._process_spectral_metadata_json(file_path)
        else:
            logger.warning(f"Unsupported image format: {suffix}")
            return []

    def _process_envi_hyperspectral(self, file_path: Path) -> list[dict]:
        """
        Process ENVI-format hyperspectral data using spectral angle mapping.
        Identifies mineral assemblages pixel-by-pixel.
        """
        try:
            img = spectral.open_image(str(file_path))
            data = img.load()
            wavelengths = img.bands.centers

            logger.info(
                f"Hyperspectral cube: {data.shape} — "
                f"{len(wavelengths)} bands, "
                f"{wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm"
            )

            detected_minerals = self._run_spectral_angle_mapping(data, wavelengths)
            summary = self._generate_hyperspectral_summary(
                file_path, data.shape, wavelengths, detected_minerals
            )

            file_hash = self._hash_path(file_path)
            return [
                {
                    "id": f"{file_hash}_hyperspectral_analysis",
                    "text": summary,
                    "metadata": {
                        "source": file_path.name,
                        "doc_type": "hyperspectral_analysis",
                        "file_type": "envi_hyperspectral",
                        "bands": len(wavelengths),
                        "rows": data.shape[0],
                        "cols": data.shape[1],
                        "wavelength_range": f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm",
                        "minerals_detected": json.dumps(list(detected_minerals.keys())),
                    },
                }
            ]

        except Exception as e:
            logger.error(f"ENVI processing failed: {e}")
            return self._create_fallback_chunk(file_path, "envi_hyperspectral", str(e))

    def _process_geotiff(self, file_path: Path) -> list[dict]:
        """
        Process GeoTIFF — commonly used for multispectral satellite bands
        (Landsat, Sentinel-2, ASTER).

        Band ratio analysis for mineral discrimination:
        - ASTER band ratios for clay minerals, iron oxides, carbonates
        """
        try:
            if not PIL_AVAILABLE:
                return self._create_fallback_chunk(file_path, "geotiff", "PIL not available")

            img = Image.open(str(file_path))
            img_array = np.array(img).astype(float)

            analysis = self._analyze_band_statistics(img_array, file_path.name)
            file_hash = self._hash_path(file_path)

            return [
                {
                    "id": f"{file_hash}_geotiff_analysis",
                    "text": analysis,
                    "metadata": {
                        "source": file_path.name,
                        "doc_type": "satellite_imagery",
                        "file_type": "geotiff",
                        "image_size": f"{img.size[0]}x{img.size[1]}",
                        "bands": img_array.shape[2] if img_array.ndim == 3 else 1,
                    },
                }
            ]
        except Exception as e:
            logger.error(f"GeoTIFF processing failed: {e}")
            return self._create_fallback_chunk(file_path, "geotiff", str(e))

    def _process_standard_image(self, file_path: Path) -> list[dict]:
        """Process RGB images — extract color statistics for mineralogical hints."""
        try:
            img = Image.open(str(file_path)).convert("RGB")
            img_array = np.array(img).astype(float)

            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            iron_ratio = np.mean(r) / (np.mean(g) + 1)
            summary_lines = [
                f"Image file: {file_path.name}",
                f"Dimensions: {img.size[0]} x {img.size[1]} pixels",
                f"Mean RGB: R={np.mean(r):.1f}, G={np.mean(g):.1f}, B={np.mean(b):.1f}",
                f"Iron oxide proxy (R/G ratio): {iron_ratio:.2f}",
            ]
            if iron_ratio > 1.5:
                summary_lines.append(
                    "Interpretation: Elevated R/G ratio suggests iron oxide alteration "
                    "(gossan, limonite, hematite). Potential IOCG or gossanous zone."
                )
            elif iron_ratio < 0.8:
                summary_lines.append(
                    "Interpretation: Low R/G ratio consistent with chloritic or "
                    "propylitic alteration assemblages."
                )
            else:
                summary_lines.append(
                    "Interpretation: Moderate R/G ratio. No strong iron oxide signature detected."
                )

            file_hash = self._hash_path(file_path)
            return [
                {
                    "id": f"{file_hash}_image_analysis",
                    "text": "\n".join(summary_lines),
                    "metadata": {
                        "source": file_path.name,
                        "doc_type": "field_image",
                        "file_type": "rgb_image",
                    },
                }
            ]
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return []

    def _process_spectral_metadata_json(self, file_path: Path) -> list[dict]:
        """
        Process JSON files containing spectral metadata or pre-computed results.
        Supports output from ENVI, QGIS spectral plugins, or custom field instruments.
        """
        try:
            with open(file_path) as f:
                data = json.load(f)

            text_parts = [f"Spectral metadata file: {file_path.name}"]

            if "minerals_detected" in data:
                text_parts.append(f"Minerals detected: {', '.join(data['minerals_detected'])}")
            if "location" in data:
                text_parts.append(f"Location: {data['location']}")
            if "survey_date" in data:
                text_parts.append(f"Survey date: {data['survey_date']}")
            if "notes" in data:
                text_parts.append(f"Notes: {data['notes']}")

            # Flatten all remaining fields
            for key, val in data.items():
                if key not in {"minerals_detected", "location", "survey_date", "notes"}:
                    text_parts.append(f"{key}: {val}")

            file_hash = self._hash_path(file_path)
            return [
                {
                    "id": f"{file_hash}_spectral_meta",
                    "text": "\n".join(text_parts),
                    "metadata": {
                        "source": file_path.name,
                        "doc_type": "spectral_metadata",
                        "file_type": "json",
                    },
                }
            ]
        except Exception as e:
            logger.error(f"JSON spectral metadata processing failed: {e}")
            return []

    def _run_spectral_angle_mapping(
        self, data: np.ndarray, wavelengths: list
    ) -> dict:
        """
        Simplified spectral angle mapping against mineral library.
        Returns detected minerals with coverage percentages.
        """
        detected = {}
        wl_array = np.array(wavelengths)

        for mineral, signature in MINERAL_SPECTRAL_SIGNATURES.items():
            diag_band = signature["diagnostic_band"]
            band_idx = np.argmin(np.abs(wl_array - diag_band))

            if band_idx < data.shape[2]:
                band_data = data[:, :, band_idx].flatten()
                # Pixels significantly below mean indicate absorption
                threshold = np.mean(band_data) - 0.5 * np.std(band_data)
                anomaly_pixels = np.sum(band_data < threshold)
                coverage = (anomaly_pixels / len(band_data)) * 100

                if coverage > 5.0:  # >5% of image shows absorption at this band
                    detected[mineral] = {
                        "coverage_pct": round(coverage, 1),
                        "alteration_type": signature["alteration_type"],
                        "deposit_association": signature["deposit_association"],
                    }

        return detected

    def _generate_hyperspectral_summary(
        self,
        file_path: Path,
        shape: tuple,
        wavelengths: list,
        detected_minerals: dict,
    ) -> str:
        """
        Generate a human-readable, embeddable summary from hyperspectral analysis.
        """
        lines = [
            f"HYPERSPECTRAL ANALYSIS REPORT",
            f"File: {file_path.name}",
            f"Scene dimensions: {shape[0]} rows × {shape[1]} columns × {shape[2]} bands",
            f"Spectral range: {wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm",
            "",
            "DETECTED MINERAL ASSEMBLAGES:",
        ]

        if not detected_minerals:
            lines.append("No significant mineral spectral signatures detected above threshold.")
        else:
            for mineral, info in detected_minerals.items():
                lines.append(
                    f"  • {mineral.replace('_', ' ').title()}: "
                    f"{info['coverage_pct']}% areal coverage | "
                    f"Alteration: {info['alteration_type'].replace('_', ' ')} | "
                    f"Associated deposits: {', '.join(info['deposit_association'])}"
                )

        # Infer alteration zoning
        alteration_types = {v["alteration_type"] for v in detected_minerals.values()}
        if "advanced_argillic" in alteration_types and "argillic" in alteration_types:
            lines.append(
                "\nALTERATION ZONATION: Advanced argillic + argillic assemblage detected. "
                "Pattern consistent with high-sulfidation epithermal system. "
                "Recommend detailed surface sampling and induced polarization survey."
            )
        elif "potassic" in alteration_types and "phyllic" in alteration_types:
            lines.append(
                "\nALTERATION ZONATION: Potassic + phyllic zoning detected. "
                "Classic porphyry Cu-Mo-Au alteration pattern. "
                "High-priority drill target."
            )
        elif "iron_oxide" in alteration_types:
            lines.append(
                "\nALTERATION ZONATION: Iron oxide assemblage dominant. "
                "May represent gossan or IOCG mineralization. "
                "Recommend geochemical sampling for Cu, Au, U."
            )

        return "\n".join(lines)

    def _analyze_band_statistics(self, img_array: np.ndarray, filename: str) -> str:
        """Generate band statistics summary for multispectral imagery."""
        n_bands = img_array.shape[2] if img_array.ndim == 3 else 1
        lines = [
            f"SATELLITE/MULTISPECTRAL IMAGE ANALYSIS",
            f"File: {filename}",
            f"Dimensions: {img_array.shape[0]} × {img_array.shape[1]} pixels, {n_bands} bands",
            "",
            "BAND STATISTICS:",
        ]
        if img_array.ndim == 3:
            band_names = ["Band 1 (Blue)", "Band 2 (Green)", "Band 3 (Red)"]
            if n_bands > 3:
                band_names.extend([f"Band {i+1}" for i in range(3, n_bands)])
            for i in range(min(n_bands, 6)):
                band = img_array[:, :, i]
                name = band_names[i] if i < len(band_names) else f"Band {i+1}"
                lines.append(
                    f"  {name}: mean={np.mean(band):.1f}, "
                    f"std={np.std(band):.1f}, "
                    f"min={np.min(band):.1f}, max={np.max(band):.1f}"
                )
        return "\n".join(lines)

    def _create_fallback_chunk(
        self, file_path: Path, file_type: str, error: str
    ) -> list[dict]:
        """Create a minimal metadata chunk when processing fails."""
        file_hash = self._hash_path(file_path)
        return [
            {
                "id": f"{file_hash}_fallback",
                "text": (
                    f"Image file registered: {file_path.name}\n"
                    f"File type: {file_type}\n"
                    f"Processing note: {error}\n"
                    f"File size: {file_path.stat().st_size / 1024:.1f} KB"
                ),
                "metadata": {
                    "source": file_path.name,
                    "doc_type": "image_metadata",
                    "file_type": file_type,
                    "processing_error": error,
                },
            }
        ]

    def _hash_path(self, file_path: Path) -> str:
        return hashlib.md5(str(file_path).encode()).hexdigest()[:12]
