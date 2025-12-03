"""
Script para limpar o cache do Data Explorer
"""

import os
import shutil
from pathlib import Path

# Caminho base dos dados
BASE_PATH = Path(__file__).parent / "data"

# Caminho do cache
metadata_path = os.path.join(BASE_PATH, 'metadata')
cache_file = os.path.join(metadata_path, 'explorer_cache.pkl')

print("Limpando cache do Data Explorer...")

# Verificar se o diretório de metadata existe
if os.path.exists(metadata_path):
    print(f"Removendo diretório de metadata: {metadata_path}")
    shutil.rmtree(metadata_path)
    print("Cache limpo com sucesso!")
else:
    print("Diretório de metadata não encontrado.")

# Verificar se existe o cache do processed_data
processed_cache = os.path.join(BASE_PATH, 'processed_data.pkl')
if os.path.exists(processed_cache):
    print(f"Removendo cache processed_data: {processed_cache}")
    os.remove(processed_cache)

print("\nPronto! Execute o Data Explorer novamente.")