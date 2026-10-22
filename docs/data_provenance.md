# Data provenance and acquisition

## Initial workspace and source check

- Primary public source: [NOAA Marine Cadastre AIS](https://marinecadastre.gov/ais/) (redirects to the Marine Cadastre Vessel Traffic page). The raw-message ordering interface is [AccessAIS](https://marinecadastre.gov/accessais/).
- Initial workspace inspection found no `data/` directory, raw AIS files, processed observations, or existing source manifest.
- AccessAIS was inspected directly on `2026-10-19`. Its public ordering API reported available observation dates `2020-01-01` through `2025-12-31`, a 2 GB per-order size limit, and a 30-second estimate time limit. An estimate for the project-area bbox (longitude -118.49 to -118.31, latitude 33.68 to 33.82; 145.07 sq mi), from `2023-01-01` through `2023-03-31`, returned 43,004 records / 4,379,174 bytes and `limit: false`. The service status endpoint returned `AIS API is running`.
- This is evidence that a small, relevant raw-AIS request is technically within the advertised limit, not proof that data was downloaded. No order was submitted, no output filename/download date/hash exists, and no raw AIS file or processing result is claimed. AccessAIS data orders are asynchronous and involve an email contact for order retrieval/status; no user contact address was available to complete that handoff. Accordingly, this workspace continues to use only the explicitly synthetic fixture.
- Marine Cadastre also lists public [AIS Vessel Transit Counts 2023](https://www.arcgis.com/home/item.html?id=141d03bf16ec4d8585cac7726ed04283), credited to NOAA Office for Coastal Management. Its item metadata says it is an annual 100 m × 100 m transit-count summary, not raw vessel positions. The listed [direct product](https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais/aistransit/ais-transit-count-2023.tif) is a GeoTIFF; the item links to [NOAA InPort item 73063](https://www.fisheries.noaa.gov/inport/item/73063). The item license statement says U.S. government material is public domain within the United States unless otherwise noted and foreign copyrights may apply; attribution is NOAA Office for Coastal Management. This aggregate raster was not substituted for AIS position records and was not downloaded or used by this pipeline.
- A deterministic synthetic development AIS-like CSV is generated on demand by `scripts/generate_synthetic_data.py`. Its manifest marks it `synthetic`, records the generation timestamp, seed, filename, row count, and SHA-256 checksum. It is not a Marine Cadastre export and not real AIS.

## Optional acquisition workflow

1. Use AccessAIS to define an area and date interval, check its estimated record count and byte size, and review the current order/use terms before submission. The API endpoint used for the documented non-ordering estimate is `https://marinecadastre.gov/accessais/api/v1/search/limit`.
2. AccessAIS supplies selected-area/time AIS data through its order workflow rather than a direct static raw-CSV URL. Complete the portal's order/retrieval process and record the exact provider title, returned filename, download date, source coverage, and current terms before placing the file in `data/raw/`. Use `scripts/download_data.py` only if the portal supplies an actual direct file URL.
3. Run `scripts/validate_data.py --input data/raw/<file>.csv`, then `scripts/build_dataset.py --input ... --source-name "<exact title>" --source-url "<source URL>"`.
4. Retain the generated manifest and quality report. Raw input is intentionally excluded from Git.

The workflow records the acquisition date locally at download/import, input filename/size/hash, observed input columns, actual timestamp and spatial coverage, filtering counts, missingness, and duplicates. URL queries, fragments, and embedded user information are removed from stored URLs to avoid persisting credentials; a manifest flag notes when sensitive URL parts were redacted. Downloaded file dates do not prove source publication date; both should be kept distinct.

## Expected columns

The processor requires `MMSI`, `BaseDateTime`, `LAT`, `LON`, and `SOG` (common Marine Cadastre AIS field names). It retains/report additional original columns without using them in model features. A different export schema must be explicitly mapped rather than silently treated as equivalent.
