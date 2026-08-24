#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÉVALUATION COMPLÈTE DU MOTEUR DE RECHERCHE SÉMANTIQUE
AssociationExplorer — 6 enquêtes : EU-SILC, HFCS, EU-LFS, HBS, IPCAL, DEMOBEL
======================================================================

Usage (depuis le dossier LLM/) :
    python evaluate_6bases.py

Sortie :
    - evaluate_results_en.json   : résultats bruts complets
    - evaluate_report_en.txt     : rapport lisible
    - evaluate_bootstrap_en.json : intervalles de confiance bootstrap

Durée estimée : 3–10 minutes selon que Ollama est actif ou non.
"""

import json, time, sys
from pathlib import Path
from collections import defaultdict

# ─── Jeu de test : 50 requêtes annotées sur les 6 enquêtes ─────────────────
# Format : {id, question, expected_codes, category, difficulty, survey_hint}
# survey_hint = None → recherche transversale, sinon enquête suggérée
# expected_codes : codes de variables jugées pertinentes (au moins 1 attendu)

QUERIES = [
    # ── EU-SILC (10) ──────────────────────────────────────────────────────────
    {"id":"q01","question":"Gross rental income received by the household",
     "expected":["HY040G","HY040N"],"category":"Revenus","difficulty":"facile","survey":"EU-SILC"},
    {"id":"q02","question":"Ability to make ends meet financially",
     "expected":["HS120"],"category":"Pauvreté","difficulty":"facile","survey":"EU-SILC"},
    {"id":"q03","question":"Family or children-related allowances received",
     "expected":["HY050G","HY050N"],"category":"Revenus","difficulty":"facile","survey":"EU-SILC"},
    {"id":"q04","question":"Impact of COVID-19 on mental health",
     "expected":["PMH010"],"category":"COVID-19","difficulty":"moyen","survey":"EU-SILC"},
    {"id":"q05","question":"Arrears on mortgage or rental payments",
     "expected":["HS011"],"category":"Logement","difficulty":"facile","survey":"EU-SILC"},
    {"id":"q06","question":"Gross employee cash or near cash income",
     "expected":["PY010G"],"category":"Revenus","difficulty":"facile","survey":"EU-SILC"},
    {"id":"q07","question":"Inability to keep home adequately warm",
     "expected":["HS060"],"category":"Logement","difficulty":"moyen","survey":"EU-SILC"},
    {"id":"q08","question":"At-risk-of-poverty or social exclusion indicator",
     "expected":["HX080"],"category":"Pauvreté","difficulty":"moyen","survey":"EU-SILC"},
    {"id":"q09","question":"Educational attainment level of the person",
     "expected":["PE040"],"category":"Démographie","difficulty":"facile","survey":"EU-SILC"},
    {"id":"q10","question":"Number of persons in the household",
     "expected":["HX040"],"category":"Démographie","difficulty":"facile","survey":"EU-SILC"},
    # ── HFCS (10) ─────────────────────────────────────────────────────────────
    {"id":"q11","question":"Value of the household main residence property",
     "expected":["DA1110"],"category":"Patrimoine","difficulty":"facile","survey":"HFCS"},
    {"id":"q12","question":"Outstanding balance of mortgage debt",
     "expected":["DL1100"],"category":"Patrimoine","difficulty":"moyen","survey":"HFCS"},
    {"id":"q13","question":"Total value of financial assets held by the household",
     "expected":["DA2100"],"category":"Patrimoine","difficulty":"moyen","survey":"HFCS"},
    {"id":"q14","question":"Household food expenditure at home",
     "expected":["HI0200","HI0100"],"category":"Consommation","difficulty":"facile","survey":"HFCS"},
    {"id":"q15","question":"Value of vehicles owned by the household",
     "expected":["DA1130","DA1130i"],"category":"Patrimoine","difficulty":"facile","survey":"HFCS"},
    {"id":"q16","question":"Mortgage repayment payments made by the household",
     "expected":["DL2100"],"category":"Patrimoine","difficulty":"moyen","survey":"HFCS"},
    {"id":"q17","question":"Total value of mutual funds held",
     "expected":["DA2102"],"category":"Patrimoine","difficulty":"difficile","survey":"HFCS"},
    {"id":"q18","question":"Regular private transfers or inheritances received",
     "expected":["HG0200","HG0210"],"category":"Revenus","difficulty":"difficile","survey":"HFCS"},
    {"id":"q19","question":"Size of the household main residence",
     "expected":["HB0900","HB0100"],"category":"Logement","difficulty":"facile","survey":"HFCS"},
    {"id":"q20","question":"Total net income of the household",
     "expected":["DI2100"],"category":"Revenus","difficulty":"facile","survey":"HFCS"},
    # ── EU-LFS (8) ────────────────────────────────────────────────────────────
    {"id":"q21","question":"Number of hours usually worked per week",
     "expected":["HWUSUAL"],"category":"Travail","difficulty":"facile","survey":"EU-LFS"},
    {"id":"q22","question":"Professional status in main job",
     "expected":["STAPRO"],"category":"Travail","difficulty":"facile","survey":"EU-LFS"},
    {"id":"q23","question":"Highest educational attainment level ISCED",
     "expected":["HATLEVEL"],"category":"Démographie","difficulty":"facile","survey":"EU-LFS"},
    {"id":"q24","question":"Full-time or part-time distinction",
     "expected":["FTPT"],"category":"Travail","difficulty":"facile","survey":"EU-LFS"},
    {"id":"q25","question":"Permanent or temporary employment contract",
     "expected":["TEMP"],"category":"Travail","difficulty":"facile","survey":"EU-LFS"},
    {"id":"q26","question":"Duration of unemployment in weeks",
     "expected":["DURUNE"],"category":"Travail","difficulty":"moyen","survey":"EU-LFS"},
    {"id":"q27","question":"Main reason for working part-time",
     "expected":["FTPTREAS"],"category":"Travail","difficulty":"moyen","survey":"EU-LFS"},
    {"id":"q28","question":"Economic sector of activity NACE classification",
     "expected":["NA11S","NA112JS"],"category":"Travail","difficulty":"moyen","survey":"EU-LFS"},
    # ── HBS (8) ───────────────────────────────────────────────────────────────
    {"id":"q29","question":"Total household expenditure on food and non-alcoholic beverages",
     "expected":["EUR_HE01"],"category":"Consommation","difficulty":"facile","survey":"HBS"},
    {"id":"q30","question":"Household spending on clothing and footwear",
     "expected":["EUR_HE03"],"category":"Consommation","difficulty":"facile","survey":"HBS"},
    {"id":"q31","question":"Health and medical care expenditure",
     "expected":["EUR_HE06"],"category":"Consommation","difficulty":"facile","survey":"HBS"},
    {"id":"q32","question":"Transport and mobility spending",
     "expected":["EUR_HE07"],"category":"Consommation","difficulty":"facile","survey":"HBS"},
    {"id":"q33","question":"Total household consumption expenditure",
     "expected":["EUR_HE00"],"category":"Consommation","difficulty":"facile","survey":"HBS"},
    {"id":"q34","question":"Housing water electricity and energy costs",
     "expected":["EUR_HE04"],"category":"Logement","difficulty":"facile","survey":"HBS"},
    {"id":"q35","question":"Communication telephone and internet expenditure",
     "expected":["EUR_HE08"],"category":"Consommation","difficulty":"moyen","survey":"HBS"},
    {"id":"q36","question":"Recreation leisure and cultural activities spending",
     "expected":["EUR_HE09"],"category":"Consommation","difficulty":"facile","survey":"HBS"},
    # ── IPCAL (8) ─────────────────────────────────────────────────────────────
    {"id":"q37","question":"Declared cadastral real estate income",
     "expected":["A1212","B1212"],"category":"Revenus","difficulty":"moyen","survey":"IPCAL"},
    {"id":"q38","question":"Replacement income unemployment and sickness benefits",
     "expected":["A7374","B7374"],"category":"Revenus","difficulty":"moyen","survey":"IPCAL"},
    {"id":"q39","question":"Deductible professional expenses for employees",
     "expected":["A9540","A9550"],"category":"Revenus","difficulty":"moyen","survey":"IPCAL"},
    {"id":"q40","question":"Deductible loan interest and debt charges",
     "expected":["A1490","B1465"],"category":"Revenus","difficulty":"moyen","survey":"IPCAL"},
    {"id":"q41","question":"Personal social security contributions",
     "expected":["A2230","B2230"],"category":"Revenus","difficulty":"moyen","survey":"IPCAL"},
    {"id":"q42","question":"Income from complementary and sharing economy activities",
     "expected":["A2487","A4603"],"category":"Revenus","difficulty":"moyen","survey":"IPCAL"},
    {"id":"q43","question":"Total taxable income of the taxpayer",
     "expected":["A7555","B7555"],"category":"Revenus","difficulty":"facile","survey":"IPCAL"},
    {"id":"q44","question":"Personal income tax rate",
     "expected":["A8096","B8096"],"category":"Revenus","difficulty":"difficile","survey":"IPCAL"},
    # ── DEMOBEL (6) ───────────────────────────────────────────────────────────
    {"id":"q45","question":"Place of residence of the person in Belgium",
     "expected":["GEO"],"category":"Démographie","difficulty":"facile","survey":"DEMOBEL"},
    {"id":"q46","question":"Age of the person",
     "expected":["AGE","MS_AGE"],"category":"Démographie","difficulty":"facile","survey":"DEMOBEL"},
    {"id":"q47","question":"Civil status or marital status",
     "expected":["LMS","CD_CIV"],"category":"Démographie","difficulty":"facile","survey":"DEMOBEL"},
    {"id":"q48","question":"Labour market situation of the person",
     "expected":["CAS","SIE"],"category":"Travail","difficulty":"facile","survey":"DEMOBEL"},
    {"id":"q49","question":"Educational attainment level according to Eurostat",
     "expected":["EDU","CD_ISCED"],"category":"Démographie","difficulty":"moyen","survey":"DEMOBEL"},
    {"id":"q50","question":"Size of the private household",
     "expected":["SPH","NOC"],"category":"Démographie","difficulty":"facile","survey":"DEMOBEL"},
]



# ─────────────────────────────────────────────────────────────────────────────
# IMPORT DU SYSTÈME
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

try:
    from step3_rag_engine import MultiSurveyRAG
    print("✅ step3_rag_engine importé")
except ImportError as e:
    print(f"❌ Impossible d'importer step3_rag_engine : {e}")
    sys.exit(1)

# Chemins ChromaDB et JSON (même logique que step5_app2.py)
def find_file(name_pattern, start="."):
    """Cherche un fichier/dossier récursivement depuis start."""
    for p in Path(start).rglob(name_pattern):
        return p
    return None

# Diagnostic préalable
print("🔍 Recherche des fichiers nécessaires...")
for f in Path(".").rglob("*.json"):
    if any(k in f.name.lower() for k in ("variable", "unified")):
        print(f"   JSON trouvé : {f}")
for f in Path(".").rglob("chroma.sqlite3"):
    print(f"   ChromaDB trouvé : {f.parent}")

vars_path   = find_file("unified_variables.json")
chroma_path = find_file("chroma.sqlite3")
if chroma_path:
    chroma_path = chroma_path.parent  # on veut le dossier, pas le fichier sqlite

if not vars_path:
    print("❌ unified_variables.json introuvable. Assurez-vous d'avoir lancé step1_unify.py")
    sys.exit(1)

if not chroma_path:
    print("❌ Dossier ChromaDB introuvable. Assurez-vous d'avoir lancé step2_embeddings.py")
    sys.exit(1)

print(f"📂 ChromaDB : {chroma_path}")
print(f"📂 Variables JSON : {vars_path}")
print("🔧 Chargement du moteur RAG...")
rag = MultiSurveyRAG(str(chroma_path), str(vars_path))
print(f"✅ Moteur prêt — {len(rag.variables_by_code)} variables indexées\n")

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS D'ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def search_with_filter(question: str, survey: str, k: int = 10):
    """Recherche en utilisant le filtre natif ChromaDB (correction biais IPCAL)."""
    return rag.search_by_question(question, top_k=k, survey_filter=[survey])

def reciprocal_rank(results, expected_codes: list) -> float:
    """1/rang du premier résultat pertinent, 0 si aucun dans le top-k."""
    expected_upper = {c.upper() for c in expected_codes}
    for rank, r in enumerate(results, 1):
        code = r['metadata'].get('code', '').upper()
        if code in expected_upper:
            return 1.0 / rank
    return 0.0

def recall_at_k(results, expected_codes: list, k: int) -> float:
    """Proportion des codes attendus trouvés dans le top-k."""
    expected_upper = {c.upper() for c in expected_codes}
    found = {r['metadata'].get('code','').upper() for r in results[:k]}
    return len(expected_upper & found) / len(expected_upper)

def precision_at_k(results, expected_codes: list, k: int) -> float:
    """Proportion des résultats dans le top-k qui sont pertinents."""
    if k == 0: return 0.0
    expected_upper = {c.upper() for c in expected_codes}
    found = [r for r in results[:k] if r['metadata'].get('code','').upper() in expected_upper]
    return len(found) / k

def cosine_mean(results) -> float:
    """Score cosinus moyen des résultats."""
    if not results: return 0.0
    return sum(r.get('score', 0) for r in results) / len(results)

# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("ÉVALUATION — 50 requêtes × 6 enquêtes")
print("="*70)

K_VALUES = [1, 3, 5, 10]
all_results = []

for q in QUERIES:
    qid      = q['id']
    question = q['question']
    expected = q['expected']
    survey   = q['survey']
    cat      = q['category']
    diff     = q['difficulty']

    t0 = time.time()
    results = search_with_filter(question, survey, k=max(K_VALUES))
    elapsed = time.time() - t0

    # Vérification : les codes attendus existent-ils dans la collection ?
    existing_expected = [
        c for c in expected
        if c.upper() in {v.upper() for v in rag.variables_by_code.keys()}
    ]
    missing_expected = [c for c in expected if c not in existing_expected]

    rr   = reciprocal_rank(results, expected)
    rank = int(1/rr) if rr > 0 else None

    row = {
        "id": qid, "question": question, "survey": survey,
        "category": cat, "difficulty": diff,
        "expected_codes": expected,
        "existing_in_corpus": existing_expected,
        "missing_from_corpus": missing_expected,
        "rr": rr, "rank_of_first_relevant": rank,
        "cosine_mean": cosine_mean(results),
        "elapsed_s": round(elapsed, 3),
        "top10_codes": [r['metadata'].get('code','') for r in results[:10]],
        "top10_scores": [round(r.get('score',0),4) for r in results[:10]],
    }
    for k in K_VALUES:
        row[f"recall@{k}"]    = recall_at_k(results, expected, k)
        row[f"precision@{k}"] = precision_at_k(results, expected, k)

    all_results.append(row)

    status = "✅" if rr > 0 else "❌"
    rank_str = f"rang {rank}" if rank else "non trouvé"
    miss_str = f" [!{missing_expected}]" if missing_expected else ""
    print(f"  {status} {qid} [{survey}] {question[:45]:<45} → {rank_str}{miss_str}")

# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES MÉTRIQUES AGRÉGÉES
# ─────────────────────────────────────────────────────────────────────────────
def agg(rows):
    n = len(rows)
    if n == 0: return {}
    d = {"n": n,
         "mrr":       round(sum(r['rr'] for r in rows) / n, 4),
         "cos_mean":  round(sum(r['cosine_mean'] for r in rows) / n, 4)}
    for k in K_VALUES:
        d[f"R@{k}"] = round(sum(r[f"recall@{k}"]    for r in rows) / n, 4)
        d[f"P@{k}"] = round(sum(r[f"precision@{k}"] for r in rows) / n, 4)
    return d

global_agg = agg(all_results)

by_survey = {}
for s in ["EU-SILC","HFCS","EU-LFS","HBS","IPCAL","DEMOBEL"]:
    rows = [r for r in all_results if r['survey']==s]
    by_survey[s] = agg(rows)

by_cat = {}
for cat in sorted({r['category'] for r in all_results}):
    rows = [r for r in all_results if r['category']==cat]
    by_cat[cat] = agg(rows)

by_diff = {}
for diff in ["facile","moyen","difficile"]:
    rows = [r for r in all_results if r['difficulty']==diff]
    by_diff[diff] = agg(rows)

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP IC à 95 % sur le MRR global (Efron, 1979)
# ─────────────────────────────────────────────────────────────────────────────
import random, statistics
random.seed(42)
B = 10000
rr_values = [r['rr'] for r in all_results]
n = len(rr_values)
bootstrap_mrrs = []
for _ in range(B):
    sample = random.choices(rr_values, k=n)
    bootstrap_mrrs.append(statistics.mean(sample))
bootstrap_mrrs.sort()
ic_lo = bootstrap_mrrs[int(0.025 * B)]
ic_hi = bootstrap_mrrs[int(0.975 * B)]

bootstrap_info = {
    "n_queries": n,
    "n_bootstrap": B,
    "mrr_observed": global_agg['mrr'],
    "ic95_lo": round(ic_lo, 4),
    "ic95_hi": round(ic_hi, 4),
    "ic95_width": round(ic_hi - ic_lo, 4),
}

# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE JSON
# ─────────────────────────────────────────────────────────────────────────────
output = {
    "global": global_agg,
    "by_survey": by_survey,
    "by_category": by_cat,
    "by_difficulty": by_diff,
    "bootstrap": bootstrap_info,
    "per_query": all_results,
}
Path("evaluate_results_en.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────────────────
sep70 = "="*70
sep50 = "-"*50

lines = [
    sep70,
    "RAPPORT D'ÉVALUATION — AssociationExplorer — 6 enquêtes",
    sep70, "",
    "MÉTRIQUES GLOBALES (50 requêtes)",
    sep50,
    f"  MRR              : {global_agg['mrr']:.4f}",
    f"  IC 95% bootstrap : [{bootstrap_info['ic95_lo']:.4f}, {bootstrap_info['ic95_hi']:.4f}]  (B={B})",
    f"  Recall@1         : {global_agg['R@1']:.4f}",
    f"  Recall@3         : {global_agg['R@3']:.4f}",
    f"  Recall@5         : {global_agg['R@5']:.4f}",
    f"  Recall@10        : {global_agg['R@10']:.4f}",
    f"  Precision@5      : {global_agg['P@5']:.4f}",
    f"  Score cosinus moy: {global_agg['cos_mean']:.4f}",
    "",
    "PAR ENQUÊTE",
    sep50,
    f"  {'Enquête':<10} {'N':>3}  {'MRR':>6}  {'R@5':>5}  {'R@10':>5}  {'P@5':>5}  {'cos':>5}",
    f"  {'-'*10} {'-'*3}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}",
]
for s, v in by_survey.items():
    lines.append(f"  {s:<10} {v['n']:>3}  {v['mrr']:>6.4f}  {v['R@5']:>5.3f}  {v['R@10']:>5.3f}  {v['P@5']:>5.3f}  {v['cos_mean']:>5.3f}")

lines += ["", "PAR CATÉGORIE", sep50,
    f"  {'Catégorie':<15} {'N':>3}  {'MRR':>6}  {'R@5':>5}  {'R@10':>5}  {'P@5':>5}"]
for cat, v in sorted(by_cat.items(), key=lambda x: -x[1]['mrr']):
    lines.append(f"  {cat:<15} {v['n']:>3}  {v['mrr']:>6.4f}  {v['R@5']:>5.3f}  {v['R@10']:>5.3f}  {v['P@5']:>5.3f}")

lines += ["", "PAR DIFFICULTÉ", sep50,
    f"  {'Difficulté':<12} {'N':>3}  {'MRR':>6}  {'R@5':>5}  {'R@10':>5}"]
for diff, v in by_diff.items():
    lines.append(f"  {diff:<12} {v['n']:>3}  {v['mrr']:>6.4f}  {v['R@5']:>5.3f}  {v['R@10']:>5.3f}")

lines += ["", "DÉTAIL PAR REQUÊTE", sep50]
for r in all_results:
    rr_str  = f"{r['rr']:.3f}"
    rank_s  = f"rang {r['rank_of_first_relevant']}" if r['rank_of_first_relevant'] else "non trouvé"
    miss_s  = f"  ⚠️ codes manquants: {r['missing_from_corpus']}" if r['missing_from_corpus'] else ""
    lines.append(f"  {r['id']} [{r['survey']}] {r['question'][:45]:<45}  RR={rr_str}  {rank_s}{miss_s}")

lines += ["", sep70, "Fichiers générés : evaluate_results_en.json / evaluate_bootstrap_en.json", sep70]

report = "\n".join(lines)
Path("evaluate_report_en.txt").write_text(report, encoding="utf-8")
Path("evaluate_bootstrap_en.json").write_text(
    json.dumps(bootstrap_info, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n" + report)
print(f"\n✅ Évaluation terminée.")
print(f"   evaluate_results_en.json  → résultats complets")
print(f"   evaluate_report_en.txt    → rapport lisible")
print(f"   evaluate_bootstrap_en.json → IC bootstrap")
