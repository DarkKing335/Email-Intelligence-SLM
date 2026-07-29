# Epic 1: Data Engineering & Dataset Management

This document details the architecture, design patterns, schema structures, and technical implementations of the **Data Engineering and Dataset Management** module (Epic 1) of the AI Email Assistant.

---

## 1. Architectural Overview

The module follows **Clean Architecture** principles to separate the domain logic (the core email schema, label taxonomy rules, and pipeline states) from operational execution (Click CLI, Loguru logging, and raw CSV loaders).

```
 ┌───────────────────────────────────────────────────────────┐
 │                        cli.main                           │ (Click CLI Entry Point)
 └─────────────────────────────┬─────────────────────────────┘
                               │ calls
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │           email_data_engineering.pipeline                 │ (Pipeline Orchestrator)
 └───────────────┬─────────────────────────────┬─────────────┘
                 │ uses                        │ uses
                 ▼                             ▼
 ┌─────────────────────────────┐ ┌───────────────────────────┐
 │         application         │ │      infrastructure       │
 │  (Importer, Normalizer,     │ │ (CSVEmailLoader,          │
 │   Cleaner, Augmentor,       │ │  DatasetStorage,          │
 │   Splitter, Versioner)      │ │  SchemaValidator)         │
 └───────────────┬─────────────┘ └─────────────┬─────────────┘
                 │                             │
                 │ references                  │ references
                 ▼                             ▼
 ┌───────────────────────────────────────────────────────────┐
 │                          domain                           │
 │        (EmailRecord, LabelTaxonomy, DatasetVersion)        │ (Core models & business rules)
 └───────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Execution Stages

The dataset lifecycle consists of **8 separated, linear stages**:

```
[ Raw CSV / JSONL ]
        │
        ▼
1. Import      : Read local source memory-efficiently using CSVEmailLoader (chunked).
        │
        ▼
2. Validate    : Schema checks (presence, length bounds) via SchemaValidator.
        │
        ▼
3. Normalize   : Map raw label strings to canonical project taxonomy using LabelTaxonomy.
        │
        ▼
4. Clean       : Strip HTML tags, signatures, thread history; redact PII; run cosine similarity dedup.
        │
        ▼
5. Augment     : Synthesize rule-based formatting, whitespace, synonym, and case variations.
        │
        ▼
6. Split       : Partition dataset into train, validation, and test splits (stratified by label).
        │
        ▼
7. Version     : Generate a unique semantic version and compile version manifest metrics.
        │
        ▼
8. Export      : Atomically write split JSONL files and version.json metadata to data/processed/.
```

---

## 3. Data Schema & Models

### Core record schema: `EmailRecord`

Every email sample inside the processing stream maps directly to this schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `email_id` | `str` | Unique record UUID |
| `thread_id` | `str` | Thread conversation identifier |
| `sender_domain` | `str` | Domain parsed from sender address |
| `subject` | `str` | Normalized subject line |
| `body_text` | `str` | Cleaned and redacted body text |
| `label` | `str` | Canonical taxonomy label |
| `original_label` | `str` | Raw label string before mapping |
| `priority` | `str` | Priority level (`high`, `medium`, `low`) |
| `response_required` | `bool` | True if this label class requires a reply draft |
| `source` | `str` | Raw file path origins |
| `split` | `str` | Train/Val/Test designation |
| `version` | `str` | Semantic dataset version string |
| `is_augmented` | `bool` | True if this is a synthetic variant |
| `augmentation_type` | `str` \| `null` | Strategy name used for synthesis |
| `created_at` | `str` | UTC timestamp string |

### Version Manifest: `DatasetVersion`

Written as `version.json` alongside the split files. Contains full audit metadata:

```json
{
  "version": "v0.1.0",
  "created_at": "2026-07-29T12:00:00Z",
  "source_path": "data/raw/archive/emails.csv",
  "source_hash": "a4d3f2b87c6e...",
  "split_counts": {
    "train": 80,
    "val": 10,
    "test": 10,
    "total": 100
  },
  "label_distribution": {
    "train": { "security": 8, "unsubscribe": 16, "fyi": 56 },
    "val": { "security": 1, "unsubscribe": 2, "fyi": 7 },
    "test": { "security": 1, "unsubscribe": 2, "fyi": 7 }
  },
  "preprocessing_steps": ["import", "validate", "normalize", "clean", "augment", "split", "version", "export"],
  "random_seed": 42,
  "config_snapshot": { ... }
}
```

---

## 4. Pipeline Stages In-Depth

### US-1.1 Import & Validate
Loads raw records chunk by chunk using a generator. The `SchemaValidator` enforces non-empty requirements for subject/body/category and filters out records violating constraints. Rejected rows are recorded in `reports/rejection_report.json` detailing line-level issues.

### US-1.2 Label Normalization
Ensures that all downstream steps run on consistent label targets. Raw categories such as `verify_code` or `promotions` map to canonical definitions (like `security` and `unsubscribe` respectively) via `configs/label_taxonomy.yaml`.

### US-1.3 Cleaning
- **Deduplication**: Drops exact duplicates (matching subject + body hashes) and near-duplicates (cosine similarity on TF-IDF word vectors matching a user-specified threshold like `0.95`).
- **Signature & Reply Stripping**: Strips greetings/signature blocks ("Best regards", "Sincerely") and reply headers (`----- Original Message -----`) using regex anchors.
- **PII Redaction**: Regular expressions match and redact email addresses, phone numbers, credit card combinations, SSN patterns, and IP addresses to prevent private data leakages into the fine-tuning training dataset.
- **Clipping**: Long email bodies exceeding maximum token lengths are clipped rather than rejected to retain partial email contexts.

### US-1.4 Augmentation
Augmentation runs deterministically utilizing a configured random seed. Original records remain untouched while synthetic records are generated and appended. Four main strategies are executed:
1. **Synonym Substitution**: Swaps words matching a local, context-appropriate dictionary (e.g. `urgent` -> `critical`).
2. **Formatting Variation**: Prepends random greetings and appends closing tags to alter structure styles.
3. **Case Variation**: Transforms subjects to Title Case, Sentence case, or UPPERCASE.
4. **Whitespace Variation**: Randomly expands line spacing between paragraphs.

### US-1.5 Splits, Versioning, & Exporting
Splits partitions the inputs into Train, Val, and Test subsets using label-based stratified splitting to preserve class balances. Versioner increments semantic tags and builds version manifests. Exporter writes the files atomically to the destination disk.

---

## 5. Directory Reports layout (`reports/`)

Every pipeline run writes dedicated summaries of each processing stage:

- **`reports/import_report.json`**: Shows ingest rates, read stats, and source tracks.
- **`reports/rejection_report.json`**: Details rejection counts and lists actual failed entries.
- **`reports/normalization_report.json`**: Tracks label map counts and logs unmapped items.
- **`reports/cleaning_report.json`**: Summary counts of PII hits, exact/near dups, and clips.
- **`reports/augmentation_report.json`**: Statistics of synthesised counts across strategy keys.
- **`reports/split_report.json`**: Verification report demonstrating stratified class distributions.
- **`reports/version_report.json`**: Build audit report including files hashes.
