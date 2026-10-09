from importlib import import_module


def test_lab7_prior_lab_interfaces_are_available():
    required = {
        "labs.lab3.search": ("build_chunks", "load_corpus", "load_questions"),
        "labs.lab4.evaluate": (
            "build_retriever",
            "judge_correctness",
            "judge_faithfulness",
        ),
        "labs.lab4.rag": ("Answer", "answer_question", "REFUSAL"),
        "labs.lab6.agent": ("make_guard", "set_defense_layers", "run_agent"),
    }
    missing = []

    for module_name, symbols in required.items():
        try:
            module = import_module(module_name)
        except ImportError as exc:
            missing.append(f"{module_name}: {exc}")
            continue
        missing.extend(
            f"{module_name}.{symbol}"
            for symbol in symbols
            if not hasattr(module, symbol)
        )

    assert not missing, (
        "Lab 7 requires these earlier-lab interfaces in a clean checkout: "
        + ", ".join(missing)
    )
