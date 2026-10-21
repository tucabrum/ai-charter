import sys
from pathlib import Path

# Permite importar os ajudantes de teste (synth, chart_parser) e o pacote sem instalar.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
