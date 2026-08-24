#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VÉRIFICATION INSTALLATION - VERSION WINDOWS
============================================

Script pour vérifier que tout est correctement installé sur Windows.

Auteur: Carnot 
Date: Mars 2026
"""

import sys
import subprocess
from pathlib import Path
import platform


def check_python():
    """Vérifie la version Python"""
    print("\n" + "="*60)
    print("1️⃣  PYTHON")
    print("="*60)
    
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("⚠️  Python 3.9+ recommandé")
        return False
    
    print(f"✅ OS: {platform.system()} {platform.release()}")
    return True


def check_packages():
    """Vérifie les packages Python"""
    print("\n" + "="*60)
    print("2️⃣  PACKAGES PYTHON")
    print("="*60)
    
    packages = {
        'sentence_transformers': 'SentenceTransformers',
        'chromadb': 'ChromaDB',
        'gradio': 'Gradio',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'openpyxl': 'OpenPyXL',
        'tqdm': 'TQDM',
        'ollama': 'Ollama Python'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, '__version__', 'N/A')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name} - MANQUANT")
            print(f"   Installation: pip install {module}")
            all_ok = False
    
    return all_ok


def check_ollama():
    """Vérifie Ollama et Llama 3.2"""
    print("\n" + "="*60)
    print("3️⃣  OLLAMA & LLM")
    print("="*60)
    
    try:
        # Vérifier version Ollama
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            shell=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama installé: {version}")
            
            # Vérifier les modèles
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                shell=True,
                timeout=10
            )
            
            if "llama3.2" in result.stdout:
                print("✅ Llama 3.2 téléchargé")
                
                # Extraire la taille
                for line in result.stdout.split('\n'):
                    if 'llama3.2' in line:
                        print(f"   {line.strip()}")
                
                return True
            else:
                print("❌ Llama 3.2 manquant")
                print("\n📥 Pour télécharger:")
                print("   ollama pull llama3.2")
                print("   (Téléchargement: ~2GB, 2-5 minutes)")
                return False
        else:
            print("❌ Ollama installé mais ne répond pas")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama non installé")
        print("\n📥 Installation Windows:")
        print("   1. Téléchargez: https://ollama.com/download/windows")
        print("   2. Installez OllamaSetup.exe")
        print("   3. Redémarrez VS Code")
        print("   4. Exécutez: ollama pull llama3.2")
        return False
        
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama ne répond pas (timeout)")
        return False
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return False


def check_files():
    """Vérifie les fichiers Excel d'entrée"""
    print("\n" + "="*60)
    print("6  FICHIERS EXCEL")
    print("="*60)
    
    input_dir = Path('../data/input')
    
    # Créer le dossier s'il n'existe pas
    input_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        'eusilc_optimized.xlsx': 'EU-SILC',
        'hfcs_variables.xlsx': 'HFCS',
        'eulfs_variables.xlsx': 'EU-LFS',
        'hbs_variables.xlsx': 'HBS',
        'IPCAL_toutes_variables.xlsx': 'IPCAL',
        'variable_codes.xlsx': 'Demobel'
    }
    
    all_ok = True
    for filename, survey in files.items():
        filepath = input_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {survey}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {survey}: {filename} - MANQUANT")
            all_ok = False
    
    if not all_ok:
        print(f"\n📁 Copiez les fichiers Excel dans:")
        print(f"   {input_dir.absolute()}")
    
    return all_ok


def check_structure():
    """Vérifie la structure du projet"""
    print("\n" + "="*60)
    print("5️⃣  STRUCTURE DU PROJET")
    print("="*60)
    
    dirs = [
        '../data/input',
        '../data/unified',
        '../data/embeddings',
        '../scripts'
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"⚠️  {dir_path} - Sera créé automatiquement")
            path.mkdir(parents=True, exist_ok=True)
    
    return True


def test_ollama_connection():
    """Test rapide de connexion Ollama"""
    print("\n" + "="*60)
    print("6️⃣  TEST OLLAMA")
    print("="*60)
    
    try:
        import ollama
        
        print("🔧 Test de connexion au LLM...")
        
        # Test simple
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'user',
                    'content': 'Réponds juste "OK"'
                }
            ],
            options={
                'num_predict': 10  # Limiter pour aller vite
            }
        )
        
        answer = response['message']['content'].strip()
        print(f"✅ Ollama répond: '{answer}'")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        print("\n💡 Solutions:")
        print("   1. Vérifiez qu'Ollama est démarré:")
        print("      ollama list")
        print("   2. Redémarrez Ollama:")
        print("      - Fermez Ollama (barre des tâches)")
        print("      - Relancez depuis le menu Démarrer")
        return False


def print_summary(results):
    """Affiche le résumé"""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)
    
    all_ok = all(results.values())
    
    for step, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {step}")
    
    print("\n" + "="*60)
    if all_ok:
        print("🎉 TOUT EST PRÊT ! Vous pouvez commencer.")
        print("="*60)
        print("\n📋 Prochaines étapes:")
        print("   1. python scripts\\step1_unify.py")
        print("   2. python scripts\\step2_embeddings.py")
        print("   3. python scripts\\step3_rag_engine.py")
        print("   4. python scripts\\step5_gradio_app.py")
    else:
        print("⚠️  INSTALLATION INCOMPLÈTE")
        print("="*60)
        print("\nCorrigez les éléments ❌ ci-dessus avant de continuer.")
    
    return all_ok


def main():
    """Fonction principale"""
    print("="*60)
    print("VÉRIFICATION INSTALLATION - WINDOWS")
    print("="*60)
    print(f"\nExécuté depuis: {Path.cwd()}")
    
    results = {
        'Python': check_python(),
        'Packages': check_packages(),
        'Ollama': check_ollama(),
        'Fichiers Excel': check_files(),
        'Structure': check_structure()
    }
    
    # Test Ollama seulement si installé
    if results['Ollama']:
        results['Test Ollama'] = test_ollama_connection()
    
    print_summary(results)


if __name__ == '__main__':
    main()