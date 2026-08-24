#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MÉTHODE 1 - STEP 4 : Wrapper LLM Local (Ollama)
================================================

Interface pour utiliser Llama 3.2 (ou autre modèle local) via Ollama
pour valider et expliquer les résultats du RAG.

ENQUÊTES SUPPORTÉES: EU-SILC, HFCS, EU-LFS, HBS, IPCAL, DEMOBEL
(Ce step est générique : aucune modification requise pour les nouvelles enquêtes)

NOUVEAUTÉS :
- Validation basée sur le score cosine brut (sans pondération)
- explain_variable() utilise les descriptions longues du step 3
- explain_from_excel() charge les descriptions depuis les fichiers Excel
- Prompts enrichis pour des explications longues et détaillées

Auteur: Carnot
Date: Mars 2026
"""

from typing import List, Dict, Optional
from pathlib import Path
import json


class LLMValidator:
    """
    Wrapper pour LLM local (Ollama)
    
    Utilise Llama 3.2 pour:
    - Valider la pertinence des variables trouvées
    - Expliquer pourquoi certaines sont pertinentes
    - Identifier les faux positifs
    """
    
    def __init__(self, model: str = "llama3.2", explained_dir: str = None):
        """
        Initialise le wrapper LLM

        Args:
            model        : Nom du modèle Ollama (défaut: llama3.2)
            explained_dir: Dossier contenant les fichiers Excel du step 3
                           (ex: data/explained). Si fourni, les descriptions
                           enrichies sont utilisées automatiquement.
        """
        self.model = model

        # Cache des descriptions enrichies (chargées depuis les Excel du step 3)
        self._explained_cache: Dict[str, Dict] = {}
        if explained_dir:
            self._load_explained_data(Path(explained_dir))

        # Vérifier si Ollama est disponible
        try:
            import ollama
            self.client = ollama.Client()
            self.available = True
            print(f"✅ Ollama disponible (modèle: {model})")
        except ImportError:
            print("⚠️  Ollama non installé")
            print("   Installation: pip install ollama")
            print("   Puis: ollama pull llama3.2")
            self.available = False
        except Exception as e:
            print(f"⚠️  Ollama non accessible: {e}")
            self.available = False

    def _load_explained_data(self, explained_dir: Path):
        """
        Charge les descriptions enrichies depuis les fichiers Excel du step 3.
        Peuple le cache _explained_cache : code → dict de descriptions.
        """
        try:
            import pandas as pd
        except ImportError:
            print("⚠️  pandas non installé — cache Excel non chargé")
            return

        if not explained_dir.exists():
            print(f"⚠️  Dossier explained introuvable: {explained_dir}")
            return

        n_loaded = 0
        for xlsx_path in explained_dir.glob("*_variables_explained.xlsx"):
            try:
                df = pd.read_excel(xlsx_path, sheet_name='Variables')
                for _, row in df.iterrows():
                    code = str(row.get('Code', '')).strip()
                    if not code or code == 'nan':
                        continue
                    self._explained_cache[code.upper()] = {
                        'desc_short_heuristique' : str(row.get('Description courte (heuristique FR)', '') or ''),
                        'desc_long_heuristique'  : str(row.get('Description longue (heuristique FR)', '') or ''),
                        'desc_short_llm_fr'      : str(row.get('Description courte LLM (FR)', '') or ''),
                        'desc_long_llm_fr'       : str(row.get('Description longue LLM (FR)', '') or ''),
                        'type_variable'          : str(row.get('Type variable', '') or ''),
                        'niveau_observation'     : str(row.get('Niveau obs.', '') or ''),
                        'langue_detectee'        : str(row.get('Langue', '') or ''),
                    }
                    n_loaded += 1
            except Exception as e:
                print(f"⚠️  Erreur lecture {xlsx_path.name}: {e}")

        print(f"   📂 {n_loaded} descriptions enrichies chargées depuis {explained_dir}")
    
    def validate_results(
        self,
        question: str,
        candidates: List[Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Valide les résultats avec le LLM
        
        Args:
            question: Question originale
            candidates: Variables candidates
            top_n: Nombre à valider
            
        Returns:
            Candidats avec validation LLM
        """
        if not self.available:
            print("⚠️  LLM non disponible, retour sans validation")
            return candidates[:top_n]
        
        # Préparer le prompt
        prompt = self._create_validation_prompt(question, candidates[:top_n])
        
        # Appeler le LLM
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'Tu es un expert en enquêtes statistiques. Tu aides à identifier les variables pertinentes.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            # Parser la réponse
            llm_text = response['message']['content']
            
            # Ajouter la validation aux candidats
            for candidate in candidates[:top_n]:
                candidate['llm_validation'] = self._extract_validation(
                    llm_text,
                    candidate['metadata']['code']
                )
            
            return candidates[:top_n]
            
        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            return candidates[:top_n]
    
    def _create_validation_prompt(
        self,
        question: str,
        candidates: List[Dict]
    ) -> str:
        """
        Crée le prompt de validation.
        Utilise les descriptions enrichies du step 3 si disponibles,
        et le score cosine brut (sans pondération).
        """
        prompt = f"""Question de l'utilisateur : "{question}"

Variables candidates trouvées par recherche sémantique (score cosine brut, sans pondération) :

"""
        for i, cand in enumerate(candidates, 1):
            meta  = cand['metadata']
            code  = meta['code']
            score = cand.get('score', 0)

            # Préférer la description enrichie du step 3 si disponible
            enriched = self._explained_cache.get(code.upper(), {})
            desc = (
                enriched.get('desc_long_llm_fr') or
                enriched.get('desc_long_heuristique') or
                meta.get('name_en', 'N/A')
            )
            # Tronquer pour ne pas surcharger le prompt
            desc_short = desc[:200] if desc else 'N/A'

            prompt += f"{i}. {code} ({meta['survey']})\n"
            prompt += f"   Description : {desc_short}\n"
            prompt += f"   Score cosine : {score:.3f}\n\n"

        prompt += """
Pour chaque variable, indique :
- ✅ si elle répond EXACTEMENT à la question
- ⚠️ si elle est partiellement pertinente
- ❌ si elle est hors sujet

Explique en 1-2 phrases claires et précises pourquoi.

Format attendu :
Variable CODE: [✅/⚠️/❌] Explication
"""
        return prompt
    
    def _extract_validation(
        self,
        llm_text: str,
        var_code: str
    ) -> Dict:
        """Extrait la validation pour une variable"""
        # Simple pattern matching
        for line in llm_text.split('\n'):
            if var_code in line:
                if '✅' in line:
                    return {'status': 'exact', 'explanation': line}
                elif '⚠️' in line:
                    return {'status': 'partial', 'explanation': line}
                elif '❌' in line:
                    return {'status': 'irrelevant', 'explanation': line}
        
        return {'status': 'unknown', 'explanation': ''}
    
    def explain_variable(self, variable: Dict) -> str:
        """
        Génère une explication longue et détaillée d'une variable.

        Priorité des sources :
        1. Descriptions LLM-FR du step 3 (si chargées via explained_dir)
        2. Descriptions heuristiques du step 3
        3. Génération LLM à la volée via Ollama
        4. Fallback textuel

        Args:
            variable: Variable complète (dict avec code, name_en, survey, etc.)

        Returns:
            Explication détaillée en langage naturel (minimum 6-8 phrases)
        """
        code = variable.get('code', '')

        # ── Priorité 1 : description LLM-FR du step 3 ─────────────────────
        enriched = self._explained_cache.get(code.upper(), {})
        if enriched.get('desc_long_llm_fr') and len(enriched['desc_long_llm_fr']) > 100:
            return enriched['desc_long_llm_fr']

        # ── Priorité 2 : description heuristique du step 3 ────────────────
        if enriched.get('desc_long_heuristique') and len(enriched['desc_long_heuristique']) > 100:
            return enriched['desc_long_heuristique']

        # ── Priorité 3 : génération LLM à la volée ────────────────────────
        if not self.available:
            return self._fallback_explanation(variable)

        survey   = variable.get('survey', '')
        name     = variable.get('name_en', '') or variable.get('name_fr', '')
        cat      = variable.get('category', 'N/A')
        desc_s   = variable.get('description_short', '')
        desc_l   = variable.get('description_long', desc_s)
        vtype    = enriched.get('type_variable', '')
        niveau   = enriched.get('niveau_observation', '')

        prompt = f"""Tu es un expert en statistiques sociales et économiques.

Voici une variable statistique issue de l'enquête {survey} :

- Code : {code}
- Nom : {name}
- Enquête : {survey}
- Catégorie : {cat}
- Type de variable : {vtype if vtype else 'non déterminé'}
- Niveau d'observation : {niveau if niveau else 'non déterminé'}
- Description originale : {desc_l[:500] if desc_l else desc_s[:300]}

Ta tâche : rédiger une explication LONGUE, DÉTAILLÉE et ACCESSIBLE EN FRANÇAIS.
Cette explication doit OBLIGATOIREMENT contenir (minimum 8 phrases) :
1. Ce que mesure précisément cette variable
2. Le contexte de l'enquête {survey} et pourquoi cette variable y figure
3. Le type de valeur attendu et son unité de mesure
4. Un exemple concret et parlant (situation réelle d'un ménage ou d'une personne)
5. Comment utiliser cette variable dans une analyse statistique
6. Les précautions d'interprétation (valeurs manquantes, cas particuliers)
7. Les relations possibles avec d'autres variables de la même enquête
8. L'importance de cette variable pour la recherche socioéconomique

Rédige UNIQUEMENT l'explication, sans introduction ni conclusion."""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role'   : 'system',
                        'content': ('Tu es un expert en statistiques qui explique '
                                    'des variables de manière simple, longue et '
                                    'accessible à un large public non spécialisé.')
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.3, 'num_predict': 800}
            )
            return response['message']['content'].strip()

        except Exception as e:
            return self._fallback_explanation(variable)
    
    def _fallback_explanation(self, variable: Dict) -> str:
        """Explication de secours sans LLM — utilise les descriptions du step 3."""
        code    = variable.get('code', '')
        survey  = variable.get('survey', '')
        name    = variable.get('name_en', '') or variable.get('name_fr', '')
        cat     = variable.get('category', 'Non spécifiée')
        desc_s  = variable.get('description_short', '')
        desc_l  = variable.get('description_long', desc_s)

        # Vérifier le cache enrichi
        enriched = self._explained_cache.get(code.upper(), {})

        long_desc = (
            enriched.get('desc_long_heuristique') or
            enriched.get('desc_long_llm_fr') or
            desc_l or
            desc_s or
            'Aucune description disponible.'
        )

        return (
            f"Variable : {code} — Enquête : {survey}\n\n"
            f"Intitulé : {name}\n"
            f"Catégorie : {cat}\n\n"
            f"{long_desc}"
        )

    def explain_from_excel(self, code: str, prefer: str = 'llm') -> str:
        """
        Retourne directement la description enrichie depuis le cache Excel du step 3.

        Args:
            code  : Code de la variable (insensible à la casse)
            prefer: 'llm' pour préférer la version LLM,
                    'heuristic' pour préférer la version heuristique

        Returns:
            Description longue en français, ou message d'absence
        """
        enriched = self._explained_cache.get(code.upper(), {})
        if not enriched:
            return f"Variable '{code}' non trouvée dans le cache Excel."

        if prefer == 'llm':
            return (
                enriched.get('desc_long_llm_fr') or
                enriched.get('desc_long_heuristique') or
                f"Aucune description disponible pour '{code}'."
            )
        else:
            return (
                enriched.get('desc_long_heuristique') or
                enriched.get('desc_long_llm_fr') or
                f"Aucune description disponible pour '{code}'."
            )


# ============================================
# DÉMONSTRATION
# ============================================

def demo():
    """Démonstration du wrapper LLM"""
    print("="*80)
    print("DÉMONSTRATION LLM VALIDATOR")
    print("="*80)

    # Dossier des fichiers Excel générés par le step 3
    explained_dir = 'data/explained'

    # Initialiser avec le cache Excel du step 3
    llm = LLMValidator("llama3.2", explained_dir=explained_dir)

    if not llm.available:
        print("\n⚠️  Ollama non disponible. Installez-le:")
        print("   curl -fsSL https://ollama.com/install.sh | sh")
        print("   ollama pull llama3.2")
        return

    # ── Candidats simulés (scores cosine bruts, sans pondération) ────────────
    fake_candidates = [
        {
            'metadata': {
                'code'   : 'HY040G',
                'survey' : 'EU-SILC',
                'name_en': 'Income from rental of a property or land (Gross)'
            },
            'score': 0.92
        },
        {
            'metadata': {
                'code'   : 'HY040N',
                'survey' : 'EU-SILC',
                'name_en': 'Income from rental of a property or land (Net)'
            },
            'score': 0.91
        },
        {
            'metadata': {
                'code'   : 'HY090G',
                'survey' : 'EU-SILC',
                'name_en': 'Interest, dividends, etc.'
            },
            'score': 0.68
        },
        {
            'metadata': {
                'code'   : 'A0270',
                'survey' : 'IPCAL',
                'name_en': 'Revenus immobiliers imposables (contribuant B)'
            },
            'score': 0.85
        },
        {
            'metadata': {
                'code'   : 'yhoooir_hyg_sm',
                'survey' : 'DEMOBEL',
                'name_en': 'non-indexed cadastral income'
            },
            'score': 0.81
        },
    ]

    question = "Quelles variables concernent les revenus locatifs?"

    # ── TEST 1 : Validation ───────────────────────────────────────────────────
    print(f"\n❓ Question : {question}\n")
    print("─"*80)
    validated = llm.validate_results(question, fake_candidates)

    for v in validated:
        code = v['metadata']['code']
        print(f"\n{code} ({v['metadata']['survey']}) — Score: {v['score']:.3f}")
        if 'llm_validation' in v:
            print(f"   {v['llm_validation'].get('explanation', 'N/A')}")

    # ── TEST 2 : Explication depuis cache Excel ───────────────────────────────
    print("\n" + "="*80)
    print("TEST : explain_from_excel() — description depuis les fichiers step 3")
    print("="*80)

    for code in ['HY040N', 'A0270', 'abo_h_fm']:
        print(f"\n🔍 Variable : {code}")
        print("─"*60)
        explanation = llm.explain_from_excel(code, prefer='llm')
        print(explanation[:400] + "..." if len(explanation) > 400 else explanation)

    # ── TEST 3 : Explication à la volée ──────────────────────────────────────
    print("\n" + "="*80)
    print("TEST : explain_variable() — explication LLM à la volée")
    print("="*80)

    test_var = {
        'code'             : 'HY040N',
        'survey'           : 'EU-SILC',
        'name_en'          : 'Income from rental of a property or land (Net)',
        'category'         : 'DUAL_N',
        'description_short': 'Net income from rental of property or land',
        'description_long' : ('The net amount received by the household for the '
                               'rental of property or land minus expenses.')
    }
    print(f"\n📋 Variable : {test_var['code']}")
    print("─"*60)
    explanation = llm.explain_variable(test_var)
    print(explanation[:600] + "..." if len(explanation) > 600 else explanation)

    print("\n" + "="*80)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("="*80)
    print("\n🚀 Prochaine étape: python step5_app2.py")


if __name__ == '__main__':
    demo()