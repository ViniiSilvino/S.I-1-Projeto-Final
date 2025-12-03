"""
Script para rodar o Data Explorer
"""

import sys
import os
from pathlib import Path

# Adiciona o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_explorer import run_interactive_explorer

if __name__ == "__main__":
    # Caminho base dos dados
    BASE_PATH = Path(__file__).parent / "data"
    
    print("\n" + "="*80)
    print("🔍 SOCCER DATA EXPLORER - MODO INTERATIVO")
    print("="*80)
    
    # Rodar em modo interativo
    run_interactive_explorer(BASE_PATH)