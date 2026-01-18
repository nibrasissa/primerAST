
from __future__ import annotations

"""
Primer design + feature engineering pipeline for PrimerAST / PrimerSAT.

This version:
- Can design primers from an existing flanking sequence, OR
- Can start from variant info (chrom, pos, assembly), fetch the region from Ensembl,
  and use Primer3 SEQUENCE_TARGET to force the amplicon across the variant locus.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import pandas as pd
import primer3

from variant_fetcher import fetch_flank_from_ensembl


# ==============================
# Helper functions
# ==============================

def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return 100.0 * gc / len(seq)


def gc_at_skew(seq: str) -> Tuple[float, float]:
    seq = seq.upper()
    A = seq.count("A")
    T = seq.count("T")
    G = seq.count("G")
    C = seq.count("C")
    gc_denom = (G + C) or 1
    at_denom = (A + T) or 1
    gc_skew = (G - C) / gc_denom
    at_skew = (A - T) / at_denom
    return gc_skew, at_skew


def longest_homopolymer(seq: str) -> int:
    seq = seq.upper()
    if not seq:
        return 0
    max_run = 1
    curr = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            curr += 1
            max_run = max(max_run, curr)
        else:
            curr = 1
    return max_run


def gc_3prime(seq: str, k: int = 5) -> int:
    seq = seq.upper()
    tail = seq[-k:] if len(seq) >= k else seq
    return tail.count("G") + tail.count("C")


def hairpin_dg(seq: str) -> float:
    hp = primer3.bindings.calcHairpin(seq)
    return hp.dg if hp.dg is not None else 0.0


def homodimer_dg(seq: str) -> float:
    d = primer3.bindings.calcHomodimer(seq)
    return d.dg if d.dg is not None else 0.0


def heterodimer_dg(seq_f: str, seq_r: str) -> float:
    d = primer3.bindings.calcHeterodimer(seq_f, seq_r)
    return d.dg if d.dg is not None else 0.0


def check_snps_placeholder(fwd_seq: str, rev_seq: str) -> Tuple[int, int]:
    # Stub – integrate real SNPCheck later if desired
    return 0, 0


def in_silico_pcr_placeholder(
    template_seq: str,
    fwd_seq: str,
    rev_seq: str,
) -> Tuple[int, int, int, bool]:
    # Stub – integrate UCSC In-Silico PCR later if desired
    return 0, 0, 0, True


# ==============================
# Primer3 configuration + design
# ==============================

@dataclass
class PrimerDesignConfig:
    gc_min: float = 40.0
    gc_max: float = 60.0
    tm_min: float = 58.0
    tm_opt: float = 60.0
    tm_max: float = 62.0
    product_size_ranges: List[Tuple[int, int]] | None = None
    n_rounds: int = 5

    def __post_init__(self) -> None:
        if self.product_size_ranges is None:
            self.product_size_ranges = [
                (90, 130),
                (120, 160),
                (150, 200),
                (180, 230),
                (200, 250),
            ]


def design_primers_with_primer3(
    template_seq: str,
    config: PrimerDesignConfig,
    target_start: int | None = None,  # 0-based
    target_len: int | None = None,    # bp
) -> List[Dict[str, Any]]:
    """
    Design primers using Primer3.

    If target_start/target_len are provided, SEQUENCE_TARGET is used so
    every amplicon is constrained to cover the variant locus.
    """
    template_seq = template_seq.upper()
    results: List[Dict[str, Any]] = []

    profiles = [
        {
            "gc_min": config.gc_min, "gc_max": config.gc_max,
            "tm_min": config.tm_min, "tm_opt": config.tm_opt, "tm_max": config.tm_max,
            "product_ranges": config.product_size_ranges,
        },
        {
            "gc_min": 35.0, "gc_max": 65.0,
            "tm_min": 56.0, "tm_opt": 60.0, "tm_max": 64.0,
            "product_ranges": [(80, 260)],
        },
        {
            "gc_min": 30.0, "gc_max": 70.0,
            "tm_min": 54.0, "tm_opt": 60.0, "tm_max": 66.0,
            "product_ranges": [(70, 300)],
        },
    ]

    for profile_idx, prof in enumerate(profiles):
        results.clear()

        for round_idx in range(config.n_rounds):
            ps_ranges = prof["product_ranges"]
            ps_min, ps_max = ps_ranges[round_idx % len(ps_ranges)]

            primer3_global = {
                "PRIMER_OPT_TM": prof["tm_opt"],
                "PRIMER_MIN_TM": prof["tm_min"],
                "PRIMER_MAX_TM": prof["tm_max"],
                "PRIMER_MIN_GC": prof["gc_min"],
                "PRIMER_MAX_GC": prof["gc_max"],
                "PRIMER_PRODUCT_SIZE_RANGE": [[ps_min, ps_max]],
                "PRIMER_MAX_SELF_ANY": 8.0,
                "PRIMER_MAX_SELF_END": 3.0,
                "PRIMER_MAX_HAIRPIN_TH": 47.0,
                "PRIMER_MAX_POLY_X": 4,
                "PRIMER_SALT_MONOVALENT": 50.0,
                "PRIMER_DNA_CONC": 50.0,
                "PRIMER_NUM_RETURN": 5,
                "PRIMER_TASK": "generic",
                "PRIMER_PICK_LEFT_PRIMER": 1,
                "PRIMER_PICK_RIGHT_PRIMER": 1,
                "PRIMER_PICK_INTERNAL_OLIGO": 0,
            }

            primer3_input = {
                "SEQUENCE_ID": f"target_profile{profile_idx}_round{round_idx}",
                "SEQUENCE_TEMPLATE": template_seq,
            }
            if target_start is not None and target_len is not None:
                primer3_input["SEQUENCE_TARGET"] = [int(target_start), int(target_len)]

            primer3_result = primer3.bindings.designPrimers(
                primer3_input,
                primer3_global,
            )

            pair_index = 0
            while f"PRIMER_PAIR_{pair_index}_PRODUCT_SIZE" in primer3_result:
                fwd_seq = primer3_result[f"PRIMER_LEFT_{pair_index}_SEQUENCE"]
                rev_seq = primer3_result[f"PRIMER_RIGHT_{pair_index}_SEQUENCE"]

                fwd_pos, fwd_len = primer3_result[f"PRIMER_LEFT_{pair_index}"]
                rev_pos, rev_len = primer3_result[f"PRIMER_RIGHT_{pair_index}"]

                product_size = primer3_result[f"PRIMER_PAIR_{pair_index}_PRODUCT_SIZE"]

                results.append(
                    {
                        "primer_id": f"profile{profile_idx}_round{round_idx}_pair{pair_index}",
                        "forward_seq": fwd_seq,
                        "reverse_seq": rev_seq,
                        "forward_start": fwd_pos,
                        "forward_len": fwd_len,
                        "reverse_start": rev_pos,
                        "reverse_len": rev_len,
                        "amplicon_size": product_size,
                    }
                )
                pair_index += 1

        if results:
            break

    return results


# ==============================
# Feature engineering
# ==============================

def build_feature_table(template_seq: str, primer_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in primer_pairs:
        fwd = p["forward_seq"]
        rev = p["reverse_seq"]

        f_len = len(fwd)
        r_len = len(rev)
        gc_f = gc_content(fwd)
        gc_r = gc_content(rev)
        gc_pair = (gc_f + gc_r) / 2.0

        tm_f = primer3.bindings.calcTm(fwd)
        tm_r = primer3.bindings.calcTm(rev)
        delta_tm = abs(tm_f - tm_r)

        gc_sk, at_sk = gc_at_skew(fwd + rev)
        homopolymer = max(longest_homopolymer(fwd), longest_homopolymer(rev))
        gc3_f = gc_3prime(fwd, k=5)
        gc3_r = gc_3prime(rev, k=5)

        hp_f = hairpin_dg(fwd)
        hp_r = hairpin_dg(rev)
        sd_f = homodimer_dg(fwd)
        sd_r = homodimer_dg(rev)
        hd_fr = heterodimer_dg(fwd, rev)

        GC_clamp_code = 0
        if gc3_f < 1 or gc3_r < 1:
            GC_clamp_code = 2
        elif gc3_f == 1 or gc3_r == 1:
            GC_clamp_code = 1

        self_annealing_code = 0
        if sd_f < -8 or sd_r < -8:
            self_annealing_code = 2
        elif sd_f < -5 or sd_r < -5:
            self_annealing_code = 1

        hairpin_code = 0
        if hp_f < -6 or hp_r < -6:
            hairpin_code = 2
        elif hp_f < -4 or hp_r < -4:
            hairpin_code = 1

        snp_count_clean, has_snp_clean = check_snps_placeholder(fwd, rev)
        amp_size, off_targets, mismatches, single_on_target = in_silico_pcr_placeholder(
            template_seq, fwd, rev
        )

        rows.append(
            {
                "primer_id": p["primer_id"],
                "forward_seq": fwd,
                "reverse_seq": rev,
                "amplicon_size": p["amplicon_size"] or amp_size,
                "forward_len": f_len,
                "reverse_len": r_len,
                "gc_content_fix": gc_pair,
                "tm_forward": tm_f,
                "tm_reverse": tm_r,
                "delta_tm": delta_tm,
                "gc_skew": gc_sk,
                "at_skew": at_sk,
                "homopolymer_max": homopolymer,
                "gc_3prime_forward": gc3_f,
                "gc_3prime_reverse": gc3_r,
                "GC clamp_code": GC_clamp_code,
                "Self-annealing_code": self_annealing_code,
                "Hairpin formation_code": hairpin_code,
                "snp_count_clean": snp_count_clean,
                "has_snp_clean": has_snp_clean,
                "off_target_count": off_targets,
                "mismatch_count": mismatches,
                "single_on_target": single_on_target,
            }
        )
    return pd.DataFrame(rows)


# ==============================
# High-level APIs
# ==============================

def run_primer_design_pipeline_from_sequence(target_sequence: str) -> pd.DataFrame:
    """Design primers from an already-provided flanking sequence."""
    target_seq = target_sequence.strip().upper()
    cfg = PrimerDesignConfig()
    primer_pairs = design_primers_with_primer3(target_seq, cfg)
    if not primer_pairs:
        cols = [
            "primer_id", "forward_seq", "reverse_seq", "amplicon_size",
            "forward_len", "reverse_len", "gc_content_fix",
            "tm_forward", "tm_reverse", "delta_tm",
            "gc_skew", "at_skew", "homopolymer_max",
            "gc_3prime_forward", "gc_3prime_reverse",
            "GC clamp_code", "Self-annealing_code",
            "Hairpin formation_code", "snp_count_clean",
            "has_snp_clean",
        ]
        return pd.DataFrame(columns=cols)
    return build_feature_table(target_seq, primer_pairs)


def run_primer_design_pipeline_for_variant(
    chromosome: str,
    position: int,
    flank: int = 500,
    assembly: str = "GRCh38",
) -> pd.DataFrame:
    """
    Variant-aware pipeline.

    - Fetches flanking region around (chromosome, position) via Ensembl REST
    - Computes variant position relative to that region
    - Uses SEQUENCE_TARGET so every amplicon covers the variant locus
    - Then designs primers and builds the feature table.
    """
    flank = int(flank)
    pos = int(position)

    seq = fetch_flank_from_ensembl(
        chromosome=chromosome,
        position=pos,
        flank=flank,
        assembly=assembly,
    )
    seq = seq.strip().upper()
    if not seq:
        return pd.DataFrame()

    # Variant is approximately at the center of the window: local index = flank
    variant_local = flank
    target_start = max(0, variant_local - 10)
    target_len = 20

    cfg = PrimerDesignConfig()
    primer_pairs = design_primers_with_primer3(
        seq,
        cfg,
        target_start=target_start,
        target_len=target_len,
    )
    if not primer_pairs:
        return pd.DataFrame()
    return build_feature_table(seq, primer_pairs)
