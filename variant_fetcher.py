
from __future__ import annotations

"""
Simple helper to fetch flanking genomic sequences around a variant
using the Ensembl REST API.
"""

import requests

ENSEMBL_REST = "https://rest.ensembl.org"


def fetch_flank_from_ensembl(
    chromosome: str,
    position: int,
    flank: int = 500,
    assembly: str = "GRCh38",
    species: str = "homo_sapiens",
) -> str:
    """
    Fetch a flanking genomic sequence around a variant using Ensembl REST.

    Args:
        chromosome: Chromosome name (e.g. "1", "2", "X")
        position: 1-based genomic position of the variant
        flank: number of bases upstream and downstream
        assembly: genome assembly, e.g. "GRCh38" or "GRCh37"
        species: species name for Ensembl, default "homo_sapiens"

    Returns:
        Flanking sequence as an uppercase string (length ~ 2*flank).
    """
    chrom = str(chromosome).replace("chr", "").replace("CHR", "")
    try:
        pos_int = int(position)
    except Exception:
        raise ValueError(f"Invalid position: {position!r}")

    start = max(1, pos_int - flank)
    end = pos_int + flank
    region = f"{chrom}:{start}-{end}"

    url = f"{ENSEMBL_REST}/sequence/region/{species}/{region}"
    params = {"coord_system_version": assembly}
    headers = {"Content-Type": "text/plain"}

    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    seq = resp.text.strip().upper()
    return seq
