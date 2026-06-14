
# ML
Data Sources
    ↓
Data Ingestion
    ↓
Data Validation
    ↓
Cleaning & Preprocessing
    ↓
Time Alignment & Resampling
    ↓
Missing Value Handling
    ↓
Outlier Detection
    ↓
Feature Store
    ↓
Feature Engineering
    ↓
External Data Enrichment
    ↓
Stationarity Analysis
    ↓
Feature Selection
    ↓
Dataset Versioning
    ↓
Train / Validation / Test Split
    ↓
Model Training
    ↓
Hyperparameter Tuning
    ↓
Ensemble Learning
    ↓
Forecast Generation
    ↓
Prediction Intervals
    ↓
Explainability
    ↓
Model Registry
    ↓
Deployment
    ↓
Real-Time Inference
    ↓
Monitoring & Drift Detection
    ↓
Automated Retraining
    ↓
Dashboard & Alerting



# RAG

Data Sources
    ↓
Data Ingestion
    ↓
Document Parsing & OCR
    ↓
Cleaning & Preprocessing
    ↓
Metadata Enrichment
    ↓
Document Classification
    ↓
PII Redaction
    ↓
Quality Validation
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Vector Database
    ↔
Knowledge Graph
    ↓
Indexing
────────────────────────────────────
    ↓
Query Understanding
    ↓
Query Rewriting
    ↓
Intent Detection
    ↓
Hybrid Retrieval
    ↓
Graph Retrieval
    ↓
Multi-Hop Retrieval
    ↓
Reranking
    ↓
Context Compression
    ↓
Prompt Engineering
    ↓
LLM Orchestration
    ↓
Tool Calling
    ↓
Agent Routing
    ↓
Generation
    ↓
Grounding & Citations
    ↓
Guardrails
    ↓
Response Validation
────────────────────────────────────
    ↓
Caching
    ↓
Monitoring
    ↓
Feedback
    ↓
Evaluation
    ↓
Continuous Learning
    ↓
LLMOps / MLOps
    ↓
Production Application





# Advanced Production RAG Pipeline

A comprehensive reference for building and operating a production-grade Retrieval-Augmented Generation (RAG) system, covering all stages from raw data ingestion through continuous improvement.

---

## Table of Contents

- [Overview](#overview)
- [Offline Pipeline](#offline-pipeline)
- [Online Pipeline](#online-pipeline)
- [Continuous Improvement Loop](#continuous-improvement-loop)
- [Architecture Diagram](#architecture-diagram)

---

## Overview

This pipeline is divided into three phases:

| Phase | Layers | Purpose |
|---|---|---|
| **Offline** | L1 – L11 | Ingest, parse, enrich, embed, and index documents |
| **Online** | L12 – L29 | Receive queries, retrieve context, generate responses |
| **Improvement** | L30 – L35 | Evaluate, monitor, collect feedback, retrain |

---

## Offline Pipeline

The offline pipeline runs asynchronously to prepare a searchable knowledge base.

### L1 · Data Sources
Raw inputs to the system — structured databases, file systems, web crawlers, APIs, data streams, and third-party integrations.

### L2 · Data Ingestion
Connectors and loaders that pull data from sources into the processing pipeline. Handles scheduling, deduplication of already-ingested documents, and error recovery.

### L3 · Document Parsing & OCR
Converts raw files into clean text. Handles PDF extraction, HTML stripping, table parsing, image OCR, and format-specific parsers (DOCX, PPTX, JSON, etc.).

### L4 · Cleaning & Preprocessing
Removes noise: boilerplate headers/footers, duplicate paragraphs, encoding artifacts, stopword-heavy passages. Normalises whitespace, Unicode, and language encoding.

### L5 · Metadata Enrichment
Attaches structured metadata to each document: author, publication date, source URL, language, document type, and custom business attributes.

### L6 · Document Classification & Tagging
Classifies documents by domain, topic, sensitivity level, and content type using zero-shot classifiers or fine-tuned models. Tags drive downstream filtering.

### L7 · Chunking Pipeline
Splits documents into retrievable units. Strategies include fixed-size sliding windows with overlap, sentence-boundary chunking, semantic chunking, and hierarchical parent-child chunking for context preservation.

### L8 · Embedding Generation
Encodes chunks into vector representations. Supports dense embeddings (bi-encoders), sparse embeddings (SPLADE, BM25 vectors), and multi-modal embeddings for mixed content.

### L9 · Vector Database Storage
Persists embeddings in an ANN index (HNSW, IVF). Organises data by namespace or tenant for multi-tenancy. Stores payload metadata alongside vectors for filtered retrieval.

### L10 · Knowledge Graph Integration
Extracts entities and relations from documents and stores them as a graph (nodes = entities, edges = relations). Bidirectionally linked with the vector store to support graph-augmented retrieval.

### L11 · Indexing Pipeline
Builds and updates BM25 / inverted indexes for keyword search. Synchronises with the vector store so hybrid retrieval can query both simultaneously.

---

## Online Pipeline

The online pipeline executes at query time, typically within a few hundred milliseconds.

### L12 · User Query Interface
Entry point for user input — chat UI, REST API, voice, or search bar. Handles authentication, session management, and rate limiting.

### L13 · Query Understanding
Parses the raw query: extracts named entities, detects language, identifies time references, and flags ambiguous or underspecified intent.

### L14 · Query Rewriting & Expansion
Improves retrieval recall by rewriting the query. Techniques include HyDE (generating a hypothetical answer to embed), synonym expansion, and decomposing complex questions into sub-queries.

### L15 · Intent Detection
Routes the query to the appropriate handler: standard RAG retrieval, direct LLM answer (no retrieval needed), tool/API call, or clarification request back to the user.

### L16 · Hybrid Retrieval
Executes dense vector search and sparse BM25 search in parallel, then fuses scores (e.g. RRF — Reciprocal Rank Fusion) for a combined candidate set.

### L17 · Multi-Query Retrieval
Runs parallel retrieval for each sub-query generated in L14. Results are merged and deduplicated before re-ranking.

### L18 · Knowledge Graph Retrieval
Traverses the knowledge graph for queries that require relational reasoning — finding paths between entities, aggregating facts, or following multi-hop chains.

### L19 · Metadata Filtering
Applies hard filters on the candidate set using metadata: date ranges, document source, access control lists, language, topic tags. Reduces the set before expensive re-ranking.

### L20 · Re-Ranking
Scores the filtered candidates with a cross-encoder or listwise ranker for relevance to the original query. Applies diversity constraints (MMR) to avoid redundant context.

### L21 · Context Compression
Reduces token count in the selected chunks without losing key information. Techniques include extractive summarisation, sentence scoring, and LLM-based compression.

### L22 · Context Assembly
Orders the final context chunks, deduplicates overlapping passages, and formats them with source attribution markers ready for prompt insertion.

### L23 · Prompt Engineering
Constructs the full prompt: system instructions, retrieved context, conversation history, few-shot examples, output format requirements, and citation instructions.

### L24 · LLM Inference
Calls the generation model. Configures sampling parameters (temperature, top-p), controls streaming, and manages token budgets across context and generation.

### L25 · Tool Calling / Agent Execution
If the LLM decides to invoke a tool (function call, code interpreter, web search, API), this layer executes the call, collects the result, and feeds it back into the generation loop.

### L26 · Response Generation
Assembles the final response text with inline citations, confidence indicators, and structured output fields as required by the application.

### L27 · Post Processing
Scrubs PII from the response, detects output language, trims excessive length, enforces format constraints, and adds response metadata.

### L28 · Guardrails & Safety Layer
Runs the response through toxicity classifiers, factual consistency checks (grounding to retrieved context), policy filters, and hallucination detection. Blocks or rewrites non-compliant outputs.

### L29 · Response Delivery
Streams the final response to the client. Caches the query–response pair where appropriate. Logs the full request/response trace for observability.

---

## Continuous Improvement Loop

### L30 · Evaluation Framework
Automated offline and online evaluation: retrieval recall@k, MRR, answer faithfulness, relevance scores (RAGAS, TruLens, custom evals). Tracks regressions across pipeline versions.

### L31 · Monitoring & Observability
Real-time dashboards for latency, error rates, retrieval quality, and LLM cost. Distributed tracing across all pipeline layers. Alerting on degradation.

### L32 · Human Feedback Collection
Thumbs up/down, correction annotations, and expert review queues. Feeds labelled data back into evaluation and fine-tuning datasets.

### L33 · Continuous Improvement Loop
Aggregates signals from evaluation, monitoring, and human feedback. Triggers retraining or prompt tuning when quality drops below thresholds.

### L34 · Embedding Re-training
Fine-tunes embedding models on domain-specific data and hard negatives mined from retrieval failures. Re-indexes the vector store after model updates.

### L35 · Knowledge Base Update Pipeline
Detects stale or outdated documents, triggers re-ingestion, and manages incremental index updates without full rebuilds.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ADVANCED PRODUCTION RAG PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘

─── OFFLINE PIPELINE ────────────────────────────────────────────────

  [L1]  Data Sources
          ↓
  [L2]  Data Ingestion
          ↓
  [L3]  Document Parsing & OCR
          ↓
  [L4]  Cleaning & Preprocessing
          ↓
  [L5]  Metadata Enrichment
          ↓
  [L6]  Document Classification & Tagging
          ↓
  [L7]  Chunking Pipeline
          ↓
  [L8]  Embedding Generation
          ↓
  [L9]  Vector Database Storage
          ↕  (bidirectional)
  [L10] Knowledge Graph Integration
          ↓
  [L11] Indexing Pipeline

─── ONLINE PIPELINE ─────────────────────────────────────────────────

  [L12] User Query Interface
          ↓
  [L13] Query Understanding
          ↓
  [L14] Query Rewriting & Expansion
          ↓
  [L15] Intent Detection
          ↓
  [L16] Hybrid Retrieval
          ↓
  [L17] Multi-Query Retrieval
          ↓
  [L18] Knowledge Graph Retrieval
          ↓
  [L19] Metadata Filtering
          ↓
  [L20] Re-Ranking
          ↓
  [L21] Context Compression
          ↓
  [L22] Context Assembly
          ↓
  [L23] Prompt Engineering
          ↓
  [L24] LLM Inference
          ↓
  [L25] Tool Calling / Agent Execution
          ↓
  [L26] Response Generation
          ↓
  [L27] Post Processing
          ↓
  [L28] Guardrails & Safety Layer
          ↓
  [L29] Response Delivery

─── CONTINUOUS IMPROVEMENT ──────────────────────────────────────────

  [L30] Evaluation Framework
          ↓
  [L31] Monitoring & Observability
          ↓
  [L32] Human Feedback Collection
          ↓
  [L33] Continuous Improvement Loop
          ↓
  [L34] Embedding Re-training
          ↓
  [L35] Knowledge Base Update Pipeline
          ↑
          └──────────────────────── back to L1
```

---

## Contributing

When adding or modifying a pipeline layer:

1. Document the layer's inputs, outputs, and configuration options.
2. Add unit tests for any new processing logic.
3. Update the evaluation suite to cover the new layer's quality signals.
4. Record latency and cost impact in the PR description.

## License

MIT



┌─────────────────────────────────────────────────────────────────────────┐
│              ADVANCED TIME SERIES DATA PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

═══ INGESTION LAYER ══════════════════════════════════════════════════════

  [L1]  Data Sources
        │  IoT sensors · financial ticks · application metrics
        │  log streams · weather feeds · clickstreams
        ↓
  [L2]  Transport & Message Broker
        │  Kafka / Redpanda / Kinesis / Pub-Sub
        │  Topics partitioned by entity_id for ordering guarantees
        ↓
  [L3]  Schema Validation & Ingestion Gateway
        │  Schema registry (Avro / Protobuf)
        │  Timestamp extraction · null checks · type coercion
        │  Dead-letter queue for malformed events

═══ STREAM PROCESSING LAYER ══════════════════════════════════════════════

        ↓
  [L4]  Event Time Alignment & Watermarking
        │  Extract event_time vs. processing_time
        │  Set watermark tolerance for late arrivals
        │  Reorder out-of-sequence events within watermark window
        ↓
  [L5]  Real-Time Cleaning
        │  Spike / outlier detection (z-score, IQR, isolation forest)
        │  Missing value imputation (forward-fill, linear interpolation)
        │  Unit normalisation · timezone standardisation
        ↓
  [L6]  Windowing & Aggregation
        │  Tumbling windows  (non-overlapping fixed intervals)
        │  Sliding windows   (overlapping, e.g. 5 min every 1 min)
        │  Session windows   (gap-based, activity-driven)
        │  Per-window: min, max, mean, stddev, percentiles, count
        ↓
  [L7]  Stream Enrichment & Joins
        │  Enrich with entity metadata (device info, geo, config)
        │  Stream-to-stream joins with bounded time tolerance
        │  Broadcast joins for slowly-changing reference tables

═══ STORAGE LAYER ════════════════════════════════════════════════════════

        ↓
  [L8]  Raw Time Series Store  (immutable, append-only)
        │  Time series DB: InfluxDB · TimescaleDB · QuestDB · Druid
        │  Retention policy: hot (SSD) → warm (HDD) → cold (object store)
        │  Partitioned by time bucket + entity_id
        ↓
  [L9]  Processed & Aggregated Store
        │  Pre-aggregated rollups (1m, 5m, 1h, 1d)
        │  Materialised views for common query patterns
        │  Columnar format (Parquet / ORC) on object storage for analytics
        ↓
  [L10] Feature Store
        │  Online store  (Redis / DynamoDB) — low-latency serving
        │  Offline store (S3 / GCS + Delta Lake) — training datasets
        │  Point-in-time correct snapshots to prevent data leakage

═══ BATCH PROCESSING LAYER ═══════════════════════════════════════════════

        ↓
  [L11] Historical Backfill & Reprocessing
        │  Replay raw events through updated logic
        │  Backfill feature store after schema / algorithm changes
        │  Spark / Dask / Ray for distributed processing
        ↓
  [L12] Data Quality & Completeness Checks
        │  Gap detection: expected vs. actual event counts per interval
        │  Flatline detection: series stuck at constant value
        │  Cross-series consistency checks (e.g. power = V × I)
        ↓
  [L13] Segmentation & Labelling
        │  Label known anomaly windows from incident reports
        │  Segment series by regime (weekday/weekend, on-peak/off-peak)
        │  Attach ground-truth forecast targets for supervised training

═══ FEATURE ENGINEERING LAYER ════════════════════════════════════════════

        ↓
  [L14] Lag & Difference Features
        │  Lag-1, lag-k, lag-seasonal (e.g. lag-168 for weekly)
        │  First-order differences (velocity)
        │  Second-order differences (acceleration)
        ↓
  [L15] Rolling Statistical Features
        │  Rolling mean, std, min, max, median
        │  Exponentially weighted moving average (EWMA)
        │  Rolling autocorrelation · rolling quantiles
        ↓
  [L16] Temporal & Calendar Features
        │  Hour of day · day of week · month · quarter
        │  Is-holiday · days-since-last-event
        │  Cyclical encoding: sin/cos transforms for periodic features
        ↓
  [L17] Frequency Domain Features
        │  FFT — dominant frequencies and amplitudes
        │  Wavelet decomposition (multi-resolution analysis)
        │  STL decomposition: trend + seasonality + residual components

═══ MODELING LAYER ═══════════════════════════════════════════════════════

        ↓
  [L18] Model Selection & Experiment Tracking
        │  Classical:  ARIMA · SARIMA · ETS · Prophet · Theta
        │  ML:         XGBoost · LightGBM · Random Forest on lag features
        │  Deep:       LSTM · TCN · N-BEATS · PatchTST · TimesNet
        │  Experiment tracking: MLflow · Weights & Biases
        ↓
  [L19] Cross-Validation & Backtesting
        │  Walk-forward validation (expanding or rolling window)
        │  Embargo gap between train and validation to prevent leakage
        │  Metrics: MAE, RMSE, MAPE, SMAPE, wQL (weighted quantile loss)
        ↓
  [L20] Probabilistic Forecasting
        │  Quantile regression · conformal prediction intervals
        │  Monte Carlo dropout · deep ensembles
        │  Calibration: reliability diagrams, coverage checks
        ↓
  [L21] Anomaly Detection
        │  Statistical: STL residuals, CUSUM, control charts
        │  ML: Isolation Forest, One-Class SVM, Autoencoder residuals
        │  Deep: LSTM reconstruction error, TranAD, USAD
        │  Alert deduplication and severity scoring
        ↓
  [L22] Model Registry & Versioning
        │  Register champion model per entity/series group
        │  Stage: Staging → Shadow → Production → Archived
        │  Artifact storage: weights, hyperparameters, feature schema

═══ SERVING LAYER ════════════════════════════════════════════════════════

        ↓
  [L23] Batch Inference Pipeline
        │  Scheduled forecasts (hourly, daily) for all series
        │  Writes predictions to forecast store with horizon metadata
        │  Supports multi-step, multi-horizon, multi-series outputs
        ↓
  [L24] Real-Time Prediction API
        │  REST / gRPC endpoint: POST /forecast
        │  Sub-100ms p99 latency via online feature store + cached model
        │  Handles single series and batch-of-series requests
        ↓
  [L25] Alerting & Notification Engine
        │  Threshold alerts (static and dynamic/adaptive)
        │  Anomaly alerts from L21 with cooldown and dedup
        │  Routing: PagerDuty · Slack · email · webhook
        ↓
  [L26] Visualisation & Dashboard Layer
        │  Grafana · Superset · custom React dashboards
        │  Forecast overlays with confidence intervals
        │  Anomaly annotations on time series charts

═══ OBSERVABILITY LAYER ══════════════════════════════════════════════════

        ↓
  [L27] Data Quality Monitoring
        │  Schema drift detection (new/missing fields)
        │  Distribution shift: PSI, KL divergence on value distributions
        │  Freshness SLA tracking: alert if series goes stale
        ↓
  [L28] Pipeline Health & SLOs
        │  End-to-end latency tracking (event_time → prediction_available)
        │  Throughput, error rate, consumer lag per Kafka topic
        │  SLO dashboards with error budget burn rates
        ↓
  [L29] Model Performance Monitoring
        │  Compare live predictions to actuals as ground truth arrives
        │  Sliding-window metrics: rolling MAE, rolling coverage
        │  Trigger retraining when metric degrades past threshold

═══ CONTINUOUS IMPROVEMENT LAYER ═════════════════════════════════════════

        ↓
  [L30] Retraining Trigger System
        │  Scheduled retraining (daily, weekly)
        │  Performance-triggered retraining from L29 signals
        │  Data-volume-triggered retraining when new labels accumulate
        ↓
  [L31] Automated Hyperparameter Optimisation
        │  Optuna · Ray Tune · Bayesian optimisation
        │  Search over model architecture, window sizes, feature sets
        │  Budget-aware: early stopping, successive halving
        ↓
  [L32] Champion / Challenger A/B Testing
        │  Shadow mode: new model runs alongside champion silently
        │  Canary rollout: route N% of traffic to challenger
        │  Promote challenger when statistically better over eval window
        ↓
  [L33] Feedback & Label Collection
        │  Operator-confirmed anomaly labels from alert responses
        │  Forecast correction annotations from domain experts
        │  Auto-label from delayed ground truth (actuals feed)
        │
        └──────────────────────────────────────────► back to L18 (retrain)



