'''
OCR Robustness Scoring

This is a standalone scoring script rather than a pytest test because OCR
robustness is evaluated quantitatively rather than as a pass/fail condition.
It compares the extracted text with manually prepared ground truth, following
the FPR requirement to estimate recognition accuracy across document types.

Each document has a corresponding `*_ground_truth.txt` file containing a
manually transcribed version of its expected text. These files provide the 
reference against which the OCR output is measured.

HOW TO RUN: 
    python tests/score_ocr.py samples/
    
WHAT IT COMPUTES:
    Word Error Rate (WER) and Character Error Rate (CER) for each document
    followed by average WER and CER for each document type.
    
The script uses the existing ``material_processor.process_file()`` pipeline
rather than reimplementing document extraction or OCR, so the measurements
reflect the application's actual processing path.
'''
import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#  Once path is set, can safely import application modules
from material_processor import process_file  # noqa: E402


def _levenshtein(seq_a, seq_b):
    '''
    Return the standard Levenshtein edit distance between two sequences.
    '''
    n, m = len(seq_a), len(seq_b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def word_error_rate(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def char_error_rate(reference, hypothesis):
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(list(reference), list(hypothesis)) / len(reference)


def score_folder(root_dir):
    results = []
    for type_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(type_dir):
            continue
        doc_type = os.path.basename(type_dir)
        gt_files = glob.glob(os.path.join(type_dir, "*_ground_truth.txt"))
        for gt_path in gt_files:
            base = os.path.basename(gt_path).replace("_ground_truth.txt", "")
            # Match each ground-truth file to its source document by filename,
            # allowing the source file to use any supported extension.
            candidates = [
                f for f in glob.glob(os.path.join(type_dir, base + ".*"))
                if not f.endswith("_ground_truth.txt")
            ]
            if not candidates:
                print(f"  [skip] no source file found for {gt_path}")
                continue
            source_path = candidates[0]
            # Ground truth is manually prepared; extracted text comes from 
            # the application's production document-processing pipeline.
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()
            try:
                extracted = process_file(source_path).strip()
            except Exception as e:
                print(f"  [error] {source_path}: {e}")
                continue
            # Lower WER/CER indicates closer agreement with the ground truth.
            wer = word_error_rate(ground_truth, extracted)
            cer = char_error_rate(ground_truth, extracted)
            results.append((doc_type, base, wer, cer))

    print(f"\n{'Type':<15}{'Document':<20}{'WER':>8}{'CER':>8}")
    print("-" * 51)
    
    # Keep per-document scores for the detailed output, then aggregate them 
    # separately by document type to compare OCR performance across categories.
    by_type = {}
    for doc_type, base, wer, cer in results:
        print(f"{doc_type:<15}{base:<20}{wer:>8.2%}{cer:>8.2%}")
        by_type.setdefault(doc_type, []).append((wer, cer))

    print("\nAverages by document type:")
    for doc_type, pairs in by_type.items():
        avg_wer = sum(p[0] for p in pairs) / len(pairs)
        avg_cer = sum(p[1] for p in pairs) / len(pairs)
        print(f"  {doc_type:<15} avg WER={avg_wer:.2%}  avg CER={avg_cer:.2%}  (n={len(pairs)})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python score_ocr.py <samples_root_dir>")
        sys.exit(1)
    score_folder(sys.argv[1])
