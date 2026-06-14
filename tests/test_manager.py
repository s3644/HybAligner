"""Tests for pipeline manager (parsing, pipeline orchestration)."""

import pytest
from runtime.manager import parse_fastq, parse_fasta


class TestParseFastq:
    def test_parse(self, sample_fastq):
        reads = parse_fastq(sample_fastq)
        assert len(reads) == 2
        assert reads[0] == "ACGTACGTACGT"
        assert reads[1] == "TGCATGCATGCA"

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_fastq("/nonexistent/file.fastq")


class TestParseFasta:
    def test_parse(self, sample_fasta):
        seq = parse_fasta(sample_fasta)
        assert seq == "ACGTACGTACGTACGTACGT"

    def test_multiline_fasta(self, temp_dir):
        import os
        path = os.path.join(temp_dir, "multi.fasta")
        with open(path, 'w') as f:
            f.write(">ref\n")
            f.write("ACGT\n")
            f.write("TGCA\n")
        seq = parse_fasta(path)
        assert seq == "ACGTTGCA"

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_fasta("/nonexistent/file.fasta")
