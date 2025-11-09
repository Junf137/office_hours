import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple, List


def remap_correct_choice(
    original_choices: Dict[str, str],
    new_choices: Dict[str, str],
    original_correct_letter: str,
) -> str:
    original_correct_letter = (original_correct_letter or "").strip().upper()
    if original_correct_letter == "E":
        return "E"
    if original_correct_letter not in ("A", "B", "C", "D"):
        return original_correct_letter
    original_correct_value = original_choices.get(original_correct_letter)
    if original_correct_value is None:
        return original_correct_letter
    for letter, value in new_choices.items():
        if value == original_correct_value:
            return letter
    return original_correct_letter


def shuffle_question_choices(
    question: Dict, rng: random.Random
) -> Tuple[bool, str]:
    """
    Shuffles A-D in-place for the question's Multiple Choice, leaves E as-is,
    and updates Correct Choice accordingly.
    Returns (changed: bool, new_correct_letter: str)
    """
    mc = question.get("Multiple Choice")
    if not isinstance(mc, dict):
        return False, str(question.get("Correct Choice", ""))

    # Snapshot original choices
    original_choices = dict(mc)

    # Keep E unchanged (both key and value)
    e_value = mc.get("E")

    # Collect A-D that exist
    ad_pairs: List[Tuple[str, str]] = []
    for letter in ("A", "B", "C", "D"):
        if letter in mc:
            ad_pairs.append((letter, mc[letter]))

    if len(ad_pairs) <= 1:
        # Nothing to shuffle
        return False, str(question.get("Correct Choice", ""))

    rng.shuffle(ad_pairs)

    # Rebuild choices: assign shuffled values back to A-D in new order
    new_mc: Dict[str, str] = {}
    for i, letter in enumerate(("A", "B", "C", "D")):
        if i < len(ad_pairs):
            # Only values are permuted; keys A-D stay as A-D positions
            new_mc[letter] = ad_pairs[i][1]
        elif letter in mc:
            # In case some letter was missing, retain original
            new_mc[letter] = mc[letter]

    # Restore E exactly as it was
    if "E" in mc:
        new_mc["E"] = e_value

    # Update question
    question["Multiple Choice"] = new_mc

    # Remap Correct Choice
    new_correct = remap_correct_choice(
        original_choices=original_choices,
        new_choices=new_mc,
        original_correct_letter=str(question.get("Correct Choice", "")),
    )
    question["Correct Choice"] = new_correct

    # Determine if anything actually changed (A-D order or correct letter)
    changed = any(
        original_choices.get(letter) != new_mc.get(letter)
        for letter in ("A", "B", "C", "D")
    ) or (new_correct != original_choices.get("Correct Choice", None))

    return changed, new_correct


def process_file(
    input_path: Path,
    output_path: Path,
    seed: int = None,
    inplace: bool = False,
    indent: int = 2,
) -> None:
    rng = random.Random(seed)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object of questions.")

    correct_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    changed_questions = 0

    for qid, q in data.items():
        if not isinstance(q, dict):
            continue
        changed, new_correct = shuffle_question_choices(q, rng)
        changed_questions += int(changed)
        new_correct_upper = (new_correct or "").strip().upper()
        if new_correct_upper in correct_counts:
            correct_counts[new_correct_upper] += 1

    # If writing in-place, backup first
    if inplace:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(str(input_path), str(backup_path))
        target_path = input_path
    else:
        target_path = output_path

    with target_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")

    # Simple stdout summary
    print(f"Shuffled A-D for {changed_questions} questions.")
    print(
        "Correct Choice distribution after shuffle: "
        + ", ".join(f"{k}:{v}" for k, v in correct_counts.items())
    )
    if inplace:
        print(f"Wrote updated JSON (backup at: {backup_path}) -> {target_path}")
    else:
        print(f"Wrote updated JSON -> {target_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle Multiple Choice answers A-D per question, keep E unchanged, and remap Correct Choice."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/questions/global_changes/.json"),
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (ignored if --inplace).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Edit the input file in place (creates a .bak backup).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation spaces when writing JSON output.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.inplace:
        output_path = input_path  # ignored
    else:
        if args.output is not None:
            output_path = args.output
        else:
            # Default output: add _shuffled before extension
            output_path = input_path.with_name(
                input_path.stem + "_shuffled" + input_path.suffix
            )

    process_file(
        input_path=input_path,
        output_path=output_path,
        seed=args.seed,
        inplace=args.inplace,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()


