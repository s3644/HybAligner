"""Shared pytest fixtures for HybAligner tests."""

import os
import tempfile
import pytest


@pytest.fixture
def temp_dir():
    """Temporary directory that auto-cleans."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_fastq(temp_dir):
    """Create a small FASTQ file."""
    path = os.path.join(temp_dir, "test.fastq")
    with open(path, 'w') as f:
        f.write("@read_0\n")
        f.write("ACGTACGTACGT\n")
        f.write("+\n")
        f.write("IIIIIIIIIIII\n")
        f.write("@read_1\n")
        f.write("TGCATGCATGCA\n")
        f.write("+\n")
        f.write("IIIIIIIIIIII\n")
    return path


@pytest.fixture
def sample_fasta(temp_dir):
    """Create a small FASTA file."""
    path = os.path.join(temp_dir, "test.fasta")
    with open(path, 'w') as f:
        f.write(">ref\n")
        f.write("ACGTACGTACGTACGTACGT\n")
    return path


@pytest.fixture
def dna_ref():
    """A small reference sequence for testing."""
    return "ACGTACGTACGTACGTACGT"


@pytest.fixture
def dna_reads():
    """Small list of reads for testing."""
    return ["ACGTACGT", "TGCATGCA", "AAAAAAAA"]
