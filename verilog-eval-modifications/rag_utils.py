import os
import re
import hashlib
import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

#PARQUET FOR RTL-Coder Vector
PARQUET_PATH_OLD = "/mnt/d/mikel/Documents/ESCUELA/MAESTRÍA/McGill_MEng_ECE/Term_4/Circuitos/FinalProject/rag-for-verilog-gen/vector_parquet/part-00000-tid-5189121977065850180-3b867b6d-c7f2-4576-b96b-f7dd87bf9dfd-126-1.c000.snappy.parquet"

#PARQUET FOR VERIRAG VECTOR
PARQUET_PATH = "/mnt/d/mikel/Documents/ESCUELA/MAESTRÍA/McGill_MEng_ECE/Term_4/Circuitos/FinalProject/rag-for-verilog-gen/vector_parquet/verirag_clean.parquet"

#PARQUET FOR RTL-Coder KG
KG_PARQUET_PATH_OLD = "/mnt/d/mikel/Documents/ESCUELA/MAESTRÍA/McGill_MEng_ECE/Term_4/Circuitos/FinalProject/rag-for-verilog-gen/kg_features_parquet/part-00000-tid-5475169348745911287-c2bd5a7c-598b-492d-a9d9-f3537b852815-141-1.c000.snappy.parquet"

#PARQUET FOR VERIRAG KG
KG_PARQUET_PATH = "/mnt/d/mikel/Documents/ESCUELA/MAESTRÍA/McGill_MEng_ECE/Term_4/Circuitos/FinalProject/rag-for-verilog-gen/kg_features_parquet/verirag_kg_features.parquet"



TOP_K = 2  # NUMBER OF EXAMPLES
MAX_INSTR_CHARS = 600
MAX_CODE_CHARS = 800
MAX_HYBRID_KG_CANDIDATES = 50
RETRIEVAL_TEXT_VERSION = "instruction_only_v2"

MIN_RERANK_SCORE = 0.30
MIN_MARGIN = 0.05 
FIRST_STAGE_K = 10 # FIRST STAGE
FINAL_K = 2   # FINAL STAGE

MIN_RERANK_SCORE_HYBRID = 0.00
MIN_MARGIN_HYBRID = 0.00


# Real embedding model
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

# Globals kept lazy because this file is called by other scripts
_VECTOR_DF = None
_KG_DF = None
_VECTOR_MODEL = None
_CORPUS_EMBEDDINGS = None

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
_RERANKER = None

def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANK_MODEL_NAME, device=_get_device())
    return _RERANKER


def _dense_retrieve_ids(query_text: str, top_k: int, allowed_doc_ids=None):
    df = load_vector_db()
    model = _get_vector_model()
    corpus_embeddings = _load_or_build_corpus_embeddings()

    if len(df) == 0:
        return []

    query_embedding = model.encode_query(
        query_text,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    if allowed_doc_ids is None:
        search_embeddings = corpus_embeddings
        index_map = None
        top_k = min(top_k, len(df))
    else:
        allowed_doc_ids = [doc_id for doc_id in allowed_doc_ids if 0 <= doc_id < len(df)]
        if not allowed_doc_ids:
            return []

        search_embeddings = corpus_embeddings[allowed_doc_ids]
        index_map = allowed_doc_ids
        top_k = min(top_k, len(allowed_doc_ids))

    if top_k == 0:
        return []

    hits = util.semantic_search(query_embedding, search_embeddings, top_k=top_k)[0]

    if index_map is None:
        return [(int(hit["corpus_id"]), float(hit["score"])) for hit in hits]

    # Map subset indices back to original doc_ids
    return [(int(index_map[int(hit["corpus_id"])]), float(hit["score"])) for hit in hits]

def _rerank_ids(query_text: str, candidate_ids):
    df = load_vector_db()
    reranker = _get_reranker()

    pairs = [(query_text, df.iloc[idx]["retrieval_text"]) for idx in candidate_ids]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )
    return [(idx, float(score)) for idx, score in ranked]


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_vector_model():
    global _VECTOR_MODEL
    if _VECTOR_MODEL is None:
        _VECTOR_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_get_device())
    return _VECTOR_MODEL


def _get_embedding_cache_path() -> str:
    parquet_abs = os.path.abspath(PARQUET_PATH)
    parquet_mtime = str(os.path.getmtime(PARQUET_PATH)) if os.path.exists(PARQUET_PATH) else "missing"

    key = f"{parquet_abs}|{parquet_mtime}|{EMBEDDING_MODEL_NAME}|{RETRIEVAL_TEXT_VERSION}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()

    cache_dir = os.path.join(os.path.dirname(parquet_abs), ".embedding_cache")
    os.makedirs(cache_dir, exist_ok=True)

    return os.path.join(cache_dir, f"vector_embeddings_{digest}.npy")


def load_vector_db():
    global _VECTOR_DF
    if _VECTOR_DF is None:
        _VECTOR_DF = pd.read_parquet(PARQUET_PATH)

        _VECTOR_DF["Instruction"] = _VECTOR_DF["Instruction"].fillna("")
        _VECTOR_DF["VerilogCode"] = _VECTOR_DF["VerilogCode"].fillna("")

        # Separate field used only for retrieval
        _VECTOR_DF["retrieval_text"] = _VECTOR_DF["Instruction"].str.strip()

    return _VECTOR_DF


def load_kg_db():
    global _KG_DF
    if _KG_DF is None:
        _KG_DF = pd.read_parquet(KG_PARQUET_PATH)

        if "module_name" in _KG_DF.columns:
            _KG_DF["module_name"] = _KG_DF["module_name"].fillna("")

        # Ensure all feature columns are numeric ints
        feature_cols = [
            # original structural features
            "has_always", "has_assign", "has_case", "has_if", "has_for", "has_generate",
            "has_posedge", "has_negedge", "has_clk_signal", "has_reset_signal",
            "tag_async_reset", "tag_fsm", "tag_counter", "tag_mux", "tag_alu",
            "tag_shift", "tag_mem_array", "tag_ram", "tag_fifo",

            # new low-level operator features
            "has_not_op", "has_and_op", "has_or_op", "has_xor_op",
            "has_add_op", "has_sub_op", "has_lshift_op", "has_rshift_op",
            "has_concat", "has_compare_eq", "has_compare_lt", "has_compare_gt",

            # new functional tags
            "tag_constant_output", "tag_not_gate", "tag_encoder", "tag_decoder",
            "tag_comparator", "tag_edge_detector", "tag_byte_reorder",
            "has_instantiation", "tag_wrapper_module",

            # interface / shape
            "num_inputs", "num_outputs", "max_decl_width",
            "has_wide_port_8plus", "has_wide_port_32plus", "is_single_output",
            "is_assign_only"
        ]

        for col in feature_cols:
            if col in _KG_DF.columns:
                _KG_DF[col] = _KG_DF[col].fillna(0)

        # explicit int casting for binary/count fields commonly used in scoring
        int_cols = [
            c for c in feature_cols
            if c not in ["num_inputs", "num_outputs", "max_decl_width"]
        ]
        for col in int_cols:
            if col in _KG_DF.columns:
                _KG_DF[col] = _KG_DF[col].astype(int)

        for col in ["num_inputs", "num_outputs", "max_decl_width"]:
            if col in _KG_DF.columns:
                _KG_DF[col] = _KG_DF[col].astype(int)

    return _KG_DF


def _load_or_build_corpus_embeddings():
    global _CORPUS_EMBEDDINGS

    if _CORPUS_EMBEDDINGS is not None:
        return _CORPUS_EMBEDDINGS

    df = load_vector_db()
    model = _get_vector_model()
    cache_path = _get_embedding_cache_path()

    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        _CORPUS_EMBEDDINGS = torch.from_numpy(arr).to(_get_device())
        return _CORPUS_EMBEDDINGS

    corpus_texts = df["retrieval_text"].tolist()

    embeddings = model.encode_document(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    _CORPUS_EMBEDDINGS = embeddings
    np.save(cache_path, embeddings.cpu().numpy())

    return _CORPUS_EMBEDDINGS

def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[_\-\/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_any(text: str, patterns) -> bool:
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def _add_positive(features: dict, name: str):
    features["positive"][name] = 1


def _add_negative(features: dict, name: str):
    features["negative"][name] = 0


def _extract_kg_query_features(prompt: str) -> dict:
    """
    Rule-based prompt-to-feature extractor.
    Returns:
        {
            "positive": {feature_name: 1, ...},
            "negative": {feature_name: 0, ...}
        }
    """
    text = _normalize_text(prompt)

    features = {
        "positive": {},
        "negative": {}
    }

    if not text:
        return features

    # ------------------------------------------------------------------
    # High-confidence functional tags
    # ------------------------------------------------------------------
    if _contains_any(text, [
        r"\bfifo\b", r"\bqueue\b", r"\bread pointer\b", r"\bwrite pointer\b",
        r"\brd_ptr\b", r"\bwr_ptr\b", r"\bread_ptr\b", r"\bwrite_ptr\b"
    ]):
        _add_positive(features, "tag_fifo")
        _add_positive(features, "tag_ram")
        _add_positive(features, "tag_mem_array")
        _add_positive(features, "has_always")
        _add_positive(features, "has_clk_signal")

    if _contains_any(text, [
        r"\bfsm\b", r"\bfinite state machine\b", r"\bstate machine\b",
        r"\bstate transition\b", r"\bnext state\b", r"\bcurrent state\b"
    ]):
        _add_positive(features, "tag_fsm")
        _add_positive(features, "has_case")
        _add_positive(features, "has_always")

    if _contains_any(text, [
        r"\bcounter\b", r"\bup counter\b", r"\bdown counter\b",
        r"\bincrement\b", r"\bdecrement\b", r"\bcounts?\b"
    ]):
        _add_positive(features, "tag_counter")
        _add_positive(features, "has_always")
        _add_positive(features, "has_clk_signal")
        _add_positive(features, "has_posedge")

    if _contains_any(text, [
        r"\bmux\b", r"\bmultiplexer\b", r"\bselector\b", r"\bselect one of\b",
        r"\b4 to 1\b", r"\b2 to 1\b", r"\b8 to 1\b"
    ]):
        _add_positive(features, "tag_mux")

    if _contains_any(text, [
        r"\balu\b", r"\barithmetic logic unit\b", r"\bopcode\b",
        r"\badd/subtract\b", r"\badd and subtract\b"
    ]):
        _add_positive(features, "tag_alu")
        _add_positive(features, "has_case")

    if _contains_any(text, [
        r"\bshift register\b", r"\bshift left\b", r"\bshift right\b",
        r"\bbarrel shifter\b", r"\bshifter\b", r"\bshift\b"
    ]):
        _add_positive(features, "tag_shift")

    if _contains_any(text, [
        r"\bram\b", r"\bmemory\b", r"\bmem\b", r"\bstorage array\b"
    ]):
        _add_positive(features, "tag_ram")
        _add_positive(features, "tag_mem_array")

    if _contains_any(text, [
        r"\balways output[s]?\s+(a\s+)?low\b",
        r"\balways output[s]?\s+(a\s+)?high\b",
        r"\balways drive[s]?\s+0\b",
        r"\balways drive[s]?\s+1\b",
        r"\balways drive[s]?\s+low\b",
        r"\balways drive[s]?\s+high\b",
        r"\bconstant output\b",
        r"\bconstant zero\b",
        r"\bconstant one\b"
    ]):
        _add_positive(features, "tag_constant_output")
        _add_positive(features, "is_assign_only")
        _add_positive(features, "is_single_output")

    if _contains_any(text, [
        r"\bnot gate\b", r"\binverter\b", r"\binvert\b", r"\bbitwise not\b"
    ]):
        _add_positive(features, "tag_not_gate")
        _add_positive(features, "has_not_op")
        _add_positive(features, "is_assign_only")

    if _contains_any(text, [
        r"\bencoder\b", r"\bpriority encoder\b"
    ]):
        _add_positive(features, "tag_encoder")
        _add_positive(features, "has_if")

    if _contains_any(text, [
        r"\bdecoder\b"
    ]):
        _add_positive(features, "tag_decoder")

    if _contains_any(text, [
        r"\bcomparator\b", r"\bcompare\b", r"\bequal\b", r"\bgreater than\b",
        r"\bless than\b", r"\bgreater-than\b", r"\bless-than\b"
    ]):
        _add_positive(features, "tag_comparator")

    if _contains_any(text, [
        r"\bedge detector\b", r"\bdetect any edge\b", r"\brising edge detector\b",
        r"\bfalling edge detector\b", r"\bedge detect\b"
    ]):
        _add_positive(features, "tag_edge_detector")
        _add_positive(features, "has_clk_signal")
        _add_positive(features, "has_posedge")
        _add_positive(features, "has_always")

    if _contains_any(text, [
        r"\breverse the byte order\b", r"\bbyte order\b", r"\bbyte reorder\b",
        r"\bbyte swap\b", r"\bendian\b"
    ]):
        _add_positive(features, "tag_byte_reorder")
        _add_positive(features, "has_concat")
        _add_positive(features, "has_wide_port_32plus")

    if _contains_any(text, [
        r"\binstantiate\b", r"\binstantiates\b", r"\btop level module\b",
        r"\btop-level module\b", r"\bwrapper\b", r"\bsubmodule\b"
    ]):
        _add_positive(features, "has_instantiation")

    # ------------------------------------------------------------------
    # Clock/reset/sequential style
    # ------------------------------------------------------------------
    if _contains_any(text, [
        r"\bclock\b", r"\bclk\b", r"\bsynchronous\b", r"\bposedge\b",
        r"\brising edge\b"
    ]):
        _add_positive(features, "has_clk_signal")
        _add_positive(features, "has_posedge")
        _add_positive(features, "has_always")

    if _contains_any(text, [
        r"\bnegedge\b", r"\bfalling edge\b"
    ]):
        _add_positive(features, "has_negedge")
        _add_positive(features, "has_always")

    if _contains_any(text, [
        r"\breset\b", r"\brst\b"
    ]):
        _add_positive(features, "has_reset_signal")

    if _contains_any(text, [
        r"\basynchronous reset\b", r"\basync reset\b", r"\bactive low reset\b",
        r"\bactive-low reset\b", r"\bnegedge reset\b", r"\bnegedge rst\b"
    ]):
        _add_positive(features, "tag_async_reset")
        _add_positive(features, "has_reset_signal")
        _add_positive(features, "has_negedge")
        _add_positive(features, "has_always")

    # ------------------------------------------------------------------
    # Structural constructs
    # ------------------------------------------------------------------
    if _contains_any(text, [
        r"\bcase\b", r"\bopcode\b", r"\bstate\b", r"\bselect\b"
    ]):
        _add_positive(features, "has_case")

    if _contains_any(text, [
        r"\bconditional\b", r"\benable\b", r"\bif-else\b", r"\bpriority\b"
    ]):
        _add_positive(features, "has_if")

    if _contains_any(text, [
        r"\bfor loop\b", r"\biterate\b", r"\bfor each bit\b", r"\bgenerate bits\b"
    ]):
        _add_positive(features, "has_for")

    if _contains_any(text, [
        r"\bgenerate\b", r"\bparameterized instances\b", r"\bn copies\b",
        r"\binstantiates multiple\b"
    ]):
        _add_positive(features, "has_generate")

    if _contains_any(text, [
        r"\bcontinuous assignment\b", r"\bassign\b", r"\bcombinational\b",
        r"\bpure combinational\b"
    ]):
        _add_positive(features, "has_assign")


    if _contains_any(text, [
        r"\band\b", r"\bbitwise and\b"
    ]):
        _add_positive(features, "has_and_op")

    if _contains_any(text, [
        r"\bor\b", r"\bbitwise or\b"
    ]):
        _add_positive(features, "has_or_op")

    if _contains_any(text, [
        r"\bxor\b", r"\bexclusive or\b", r"\bparity\b"
    ]):
        _add_positive(features, "has_xor_op")

    if _contains_any(text, [
        r"\badd\b", r"\badder\b", r"\bsum\b"
    ]):
        _add_positive(features, "has_add_op")

    if _contains_any(text, [
        r"\bsubtract\b", r"\bsubtractor\b", r"\bdifference\b"
    ]):
        _add_positive(features, "has_sub_op")

    if _contains_any(text, [
        r"\bshift left\b", r"\bleft shift\b"
    ]):
        _add_positive(features, "has_lshift_op")

    if _contains_any(text, [
        r"\bshift right\b", r"\bright shift\b"
    ]):
        _add_positive(features, "has_rshift_op")

    if _contains_any(text, [
        r"\bconcatenate\b", r"\bconcatenation\b"
    ]):
        _add_positive(features, "has_concat")

    if _contains_any(text, [
        r"\b32 bits\b", r"\b32-bit\b", r"\b\[31:0\]\b"
    ]):
        _add_positive(features, "has_wide_port_32plus")

    if _contains_any(text, [
        r"\b8 bits\b", r"\b8-bit\b", r"\b\[7:0\]\b"
    ]):
        _add_positive(features, "has_wide_port_8plus")

    if _contains_any(text, [
        r"\boutput zero\b", r"\boutput one\b", r"\bsingle output\b"
    ]):
        _add_positive(features, "is_single_output")


    # ------------------------------------------------------------------
    # Derived implications
    # ------------------------------------------------------------------
    pos = features["positive"]

    if pos.get("tag_fsm") == 1:
        pos.setdefault("has_case", 1)
        pos.setdefault("has_always", 1)

    if pos.get("tag_counter") == 1:
        pos.setdefault("has_clk_signal", 1)
        pos.setdefault("has_posedge", 1)
        pos.setdefault("has_always", 1)

    if pos.get("tag_fifo") == 1:
        pos.setdefault("tag_ram", 1)
        pos.setdefault("tag_mem_array", 1)
        pos.setdefault("has_clk_signal", 1)
        pos.setdefault("has_always", 1)

    if pos.get("tag_async_reset") == 1:
        pos.setdefault("has_reset_signal", 1)
        pos.setdefault("has_negedge", 1)
        pos.setdefault("has_always", 1)

    if pos.get("tag_constant_output") == 1:
        pos.setdefault("is_assign_only", 1)
        pos.setdefault("is_single_output", 1)
        pos.setdefault("has_assign", 1)

    if pos.get("tag_not_gate") == 1:
        pos.setdefault("has_not_op", 1)
        pos.setdefault("is_assign_only", 1)
        pos.setdefault("has_assign", 1)

    if pos.get("tag_byte_reorder") == 1:
        pos.setdefault("has_concat", 1)
        pos.setdefault("has_wide_port_32plus", 1)
        pos.setdefault("has_assign", 1)

    if pos.get("tag_edge_detector") == 1:
        pos.setdefault("has_clk_signal", 1)
        pos.setdefault("has_posedge", 1)
        pos.setdefault("has_always", 1)

    if pos.get("has_instantiation") == 1:
        pos.setdefault("tag_wrapper_module", 1)


    # ------------------------------------------------------------------
    # Explicit negative cues
    # ------------------------------------------------------------------
    if _contains_any(text, [
        r"\bwithout clock\b", r"\bno clock\b", r"\bclockless\b"
    ]):
        _add_negative(features, "has_clk_signal")
        _add_negative(features, "has_posedge")
        _add_negative(features, "has_negedge")

    if _contains_any(text, [
        r"\bno reset\b", r"\bwithout reset\b"
    ]):
        _add_negative(features, "has_reset_signal")
        _add_negative(features, "tag_async_reset")

    if _contains_any(text, [
        r"\bcombinational\b", r"\bpure combinational\b"
    ]):
        _add_negative(features, "has_posedge")
        _add_negative(features, "has_negedge")
        _add_negative(features, "has_clk_signal")

    return features


def _get_kg_feature_weights() -> dict:
    return {
        # original high-level tags
        "tag_fsm": 5.0,
        "tag_counter": 5.0,
        "tag_mux": 4.0,
        "tag_alu": 5.0,
        "tag_shift": 4.0,
        "tag_mem_array": 4.0,
        "tag_ram": 5.0,
        "tag_fifo": 6.0,
        "tag_async_reset": 4.0,

        # new functional tags
        "tag_constant_output": 5.0,
        "tag_not_gate": 5.0,
        "tag_encoder": 5.0,
        "tag_decoder": 5.0,
        "tag_comparator": 5.0,
        "tag_edge_detector": 5.5,
        "tag_byte_reorder": 6.0,
        "tag_wrapper_module": 4.0,

        # operator features
        "has_not_op": 3.0,
        "has_and_op": 2.0,
        "has_or_op": 2.0,
        "has_xor_op": 2.5,
        "has_add_op": 2.5,
        "has_sub_op": 2.5,
        "has_lshift_op": 2.5,
        "has_rshift_op": 2.5,
        "has_concat": 3.0,
        "has_compare_eq": 2.5,
        "has_compare_lt": 2.5,
        "has_compare_gt": 2.5,

        # sequential / clocking
        "has_posedge": 3.0,
        "has_negedge": 3.0,
        "has_clk_signal": 3.0,
        "has_reset_signal": 2.5,

        # generic structure
        "has_always": 2.0,
        "has_assign": 2.0,
        "has_case": 2.0,
        "has_if": 1.5,
        "has_for": 1.0,
        "has_generate": 1.5,
        "has_instantiation": 2.5,

        # interface / shape
        "has_wide_port_8plus": 2.0,
        "has_wide_port_32plus": 3.0,
        "is_single_output": 2.5,
        "is_assign_only": 3.0,
    }


def _score_kg_row(row, positive: dict, negative: dict, weights: dict) -> float:
    score = 0.0

    # Reward expected positive features, penalize their absence
    for feat in positive:
        w = weights.get(feat, 1.0)
        val = int(getattr(row, feat, 0))
        if val == 1:
            score += w
        else:
            score -= 0.5 * w

    # Reward explicit negatives being absent, penalize if present
    for feat in negative:
        w = weights.get(feat, 1.0)
        val = int(getattr(row, feat, 0))
        if val == 0:
            score += 0.5 * w
        else:
            score -= w

    return score


def _format_example_chunk(example_idx: int, instruction: str, verilog: str, score_label: str, score_value: float) -> str:
    instruction = instruction[:MAX_INSTR_CHARS ]
    verilog = verilog[:MAX_CODE_CHARS ]

    chunk = f"""I will provide a reference example below. Use it only if it is helpful for the current task. If it is not relevant, ignore it. Do not copy it blindly.
    
Reference Example {example_idx}:

Instruction:
{instruction}

Verilog:
{verilog}
"""
    return chunk



def _get_kg_filtered_doc_ids(prompt: str):
    """
    Returns a list of doc_ids that satisfy the KG-based structural filter,
    ordered by KG score descending.
    """
    kg_df = load_kg_db()

    query_text = (prompt or "").strip()
    if not query_text or len(kg_df) == 0:
        return []

    query_features = _extract_kg_query_features(query_text)
    positive = query_features["positive"]
    negative = query_features["negative"]

    # If nothing meaningful was extracted, do not filter.
    if not positive and not negative:
        return []

    weights = _get_kg_feature_weights()

    scored = []
    for row in kg_df.itertuples(index=False):
        score = _score_kg_row(row, positive, negative, weights)
        if score > 0:
            scored.append((int(row.doc_id), float(score)))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep unique doc_ids in KG-score order
    seen = set()
    doc_ids = []
    for doc_id, _ in scored:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)

    return doc_ids[:MAX_HYBRID_KG_CANDIDATES]

def _should_use_retrieval_hybrid(reranked):
    if not reranked:
        print("[HYBRID GATE] no reranked candidates")
        return False

    top_score = float(reranked[0][1])
    second_score = float(reranked[1][1]) if len(reranked) > 1 else None
    margin = (top_score - second_score) if second_score is not None else None

    print(f"[HYBRID GATE] top={top_score:.4f} second={second_score} margin={margin}")

    if top_score < MIN_RERANK_SCORE_HYBRID:
        print("[HYBRID GATE] rejected: top score below threshold")
        return False

    if second_score is not None and (top_score - second_score) < MIN_MARGIN_HYBRID:
        print("[HYBRID GATE] rejected: margin below threshold")
        return False

    print("[HYBRID GATE] accepted")
    return True



def _get_vector_context_from_doc_ids(prompt: str, allowed_doc_ids) -> str:
    df = load_vector_db()

    query_text = (prompt or "").strip()
    if not query_text:
        return ""

    if len(df) == 0:
        return ""

    if not allowed_doc_ids:
        return ""

    # Stage 1: dense retrieval inside KG-filtered subset
    dense_hits = _dense_retrieve_ids(
        query_text,
        top_k=FIRST_STAGE_K,
        allowed_doc_ids=allowed_doc_ids
    )
    if not dense_hits:
        return ""

    candidate_ids = [idx for idx, _ in dense_hits]

    # Stage 2: reranking on the subset candidates
    reranked = _rerank_ids(query_text, candidate_ids)
    if not reranked:
        return ""

    # Stage 3: gate
    if not _should_use_retrieval_hybrid(reranked):
        return ""

    # Stage 4: final selection
    selected = reranked[:min(FINAL_K, len(reranked))]

    chunks = []
    for i, (doc_id, rerank_score) in enumerate(selected, start=1):
        row = df.iloc[doc_id]

        chunk = _format_example_chunk(
            example_idx=i,
            instruction=row["Instruction"],
            verilog=row["VerilogCode"],
            score_label="HybridRerankScore",
            score_value=float(rerank_score)
        )
        chunks.append(chunk)

    return "\n".join(chunks)

def _should_use_retrieval(reranked):
    if not reranked:
        return False

    top_score = float(reranked[0][1])
    second_score = float(reranked[1][1]) if len(reranked) > 1 else None

    if top_score < MIN_RERANK_SCORE:
        return False

    if second_score is not None and (top_score - second_score) < MIN_MARGIN:
        return False

    return True


def get_vector_context(prompt: str) -> str:
    df = load_vector_db()

    query_text = (prompt or "").strip()
    if not query_text:
        return ""

    if len(df) == 0:
        return ""

    # Stage 1: dense retrieval
    dense_hits = _dense_retrieve_ids(query_text, min(FIRST_STAGE_K, len(df)))
    if not dense_hits:
        return ""

    candidate_ids = [idx for idx, _ in dense_hits]

    # Stage 2: reranking
    reranked = _rerank_ids(query_text, candidate_ids)
    if not reranked:
        return ""

    # Gate: if retrieval confidence is low, fall back to vanilla
    if not _should_use_retrieval(reranked):
        return ""

    # Keep only the best final examples
    selected = reranked[:min(FINAL_K, len(reranked))]

    chunks = []
    for i, (doc_id, rerank_score) in enumerate(selected, start=1):
        row = df.iloc[doc_id]

        instruction = row["Instruction"]
        verilog = row["VerilogCode"]

        chunk = _format_example_chunk(
            example_idx=i,
            instruction=instruction,
            verilog=verilog,
            score_label="RerankScore",
            score_value=float(rerank_score)
        )
        chunks.append(chunk)

    return "\n".join(chunks)

def get_kg_context(prompt: str) -> str:
    kg_df = load_kg_db()
    vector_df = load_vector_db()

    query_text = (prompt or "").strip()
    if not query_text or len(kg_df) == 0 or len(vector_df) == 0:
        return ""

    query_features = _extract_kg_query_features(query_text)
    positive = query_features["positive"]
    negative = query_features["negative"]

    print("[KG] positive =", positive)
    print("[KG] negative =", negative)

    # If nothing meaningful was extracted, return empty context
    # so KG retrieval stays conservative.
    if not positive and not negative:
        return ""

    weights = _get_kg_feature_weights()

    scored = []
    for row in kg_df.itertuples(index=False):
        score = _score_kg_row(row, positive, negative, weights)

        # Keep only candidates with some useful structural compatibility
        if score > 0:
            scored.append((int(row.doc_id), float(score)))

    print("[KG] num_scored_candidates =", len(scored))

    if not scored:
        return ""

    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate doc_ids while preserving order
    seen_doc_ids = set()
    top_hits = []
    for doc_id, score in scored:
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        top_hits.append((doc_id, score))
        if len(top_hits) >= TOP_K:
            break

    if not top_hits:
        return ""

    chunks = []
    for i, (doc_id, score) in enumerate(top_hits, start=1):
        # We assume doc_id aligns with the corresponding row in vector_df
        if doc_id < 0 or doc_id >= len(vector_df):
            continue

        row = vector_df.iloc[doc_id]
        instruction = row["Instruction"]
        verilog = row["VerilogCode"]

        chunk = _format_example_chunk(
            example_idx=i,
            instruction=instruction,
            verilog=verilog,
            score_label="KGScore",
            score_value=score
        )
        chunks.append(chunk)

    return "\n".join(chunks)


def get_hybrid_context(prompt: str) -> str:
    kg_doc_ids = _get_kg_filtered_doc_ids(prompt)
    print("[HYBRID] num_kg_doc_ids =", len(kg_doc_ids))

    if not kg_doc_ids:
        print("[HYBRID] fallback to full vector retrieval")
        return get_vector_context(prompt)

    print("[HYBRID] using KG-filtered vector retrieval")
    return _get_vector_context_from_doc_ids(prompt, kg_doc_ids)


def get_rag_context(prompt: str, mode: str) -> str:
    if mode == "vector":
        return get_vector_context(prompt)
    elif mode == "kg":
        return get_kg_context(prompt)
    elif mode == "hybrid":
        return get_hybrid_context(prompt)
    return ""