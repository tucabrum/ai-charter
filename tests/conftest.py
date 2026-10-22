import sys
from pathlib import Path

# Permite importar os ajudantes de teste (synth, chart_parser) e o pacote sem instalar.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture(autouse=True)
def _rules_only_for_synthetic_drums(monkeypatch):
    """Os sons de bateria sintéticos dos testes não se parecem com um kit de verdade, e o modelo
    (treinado com charts oficiais) não foi feito para eles: os testes conferem as regras do
    transcritor. O modelo tem testes próprios em test_drums.py."""
    import ai_charter.pipeline as pipeline

    class _NoModel:
        @staticmethod
        def load(*_args, **_kwargs):
            return None

    monkeypatch.setattr(pipeline, "DrumModel", _NoModel)
