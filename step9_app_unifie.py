#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 9 - INTERFACE UNIFIÉE AssociationExplorer
================================================

Intègre en une seule application :
  - Partie I  : Recherche sémantique (step5 — RAG + E5-Large + ChromaDB)
  - Partie II : Associations statistiques (step7 — R², FDR, Surprise_B)

NOUVEAUTÉS vs step8 :
  ✅ Une seule instance E5-Large + ChromaDB (chargée une seule fois)
  ✅ Onglets connectés : Tab 1 → codes injectés automatiquement dans Tab 2
  ✅ Mapping Excel intégré : codes BEAMM → descriptions lisibles + IDs ChromaDB
  ✅ Vraies distances cosinus (pas de proxy binaire) via variable_id_map
  ✅ Fallback sémantique : encode la DESCRIPTION (pas le code brut)

FLUX :
  Tab 1 : question → RAG → K variables sémantiques (HTML + codes State)
              ↓ [Bouton → Calculer]
  Tab 2 : K variables → associations + vraies distances → S_découverte / S_attendu
              ↓ [automatique après calcul]
  Tab 3 : volcano plot (nuage dispersé, pas diagonale)

PORT : 7862 (step5=7860, step8=7861 inchangés)

Auteur  : AssociationExplorer — Partie I & II
Date    : 2025-2026
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio")

# ─── Imports du projet ────────────────────────────────────────────────────────
try:
    from step5_app2 import AssociationExplorer as _Step5App
    STEP5_CLASS = _Step5App
except Exception as e:
    STEP5_CLASS = None
    print(f"⚠️  step5_app2 non chargé ({e})")

try:
    from step6_data_loader import DataLoader
    from step7_association  import AssociationEngine, AssociationResults
except ImportError as e:
    raise ImportError(f"step6/step7 introuvable : {e}")

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)s | %(message)s",
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODULE MAPPING EXCEL (step6b intégré, section 3.9.5 du mémoire)
# ══════════════════════════════════════════════════════════════════════════════

_EXCEL_NAMES    = Path("data/Names_mapping_28_04.xlsx")
_EXCEL_VARS     = Path("data/Variable_names_BEAMM.xlsx")
_SHEETS_WITH_DATA = [
    "SILC", "HFCS", "LFS", "HBS (statbel 16)", "HBS (eurostat)",
    "DEMOBEL", "TB_calc", "Vehicles", "Fantasi", "Totals", "Interface",
]


def load_excel_mappings() -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Charge les deux fichiers Excel et construit quatre dictionnaires :

    beamm_to_desc    : {beamm_code → description lisible en anglais}
    beamm_to_orig    : {beamm_code → original_code (HY090G, HWUSUAL...)}
    beamm_to_db      : {beamm_code → base d'origine (SILC, LFS, HBS...)}
    orig_to_beamm    : {original_code → beamm_code} — pivot RAG→BEAMM
                       Construit principalement depuis Names_mapping_28_04.xlsx
                       qui est la source authoritative (722 mappings complets).

    Retourne des dicts vides si les fichiers sont absents (dégradation gracieuse).
    """
    beamm_to_desc : Dict[str, str] = {}
    beamm_to_orig : Dict[str, str] = {}
    beamm_to_db   : Dict[str, str] = {}
    orig_to_beamm : Dict[str, str] = {}

    # ── Source 1 : Names_mapping_28_04.xlsx ──────────────────────────────────
    # Fichier pivot : Original_variable_name → New_variable_name (BEAMM)
    # C'est la source authoritative pour relier les codes RAG aux codes BEAMM
    if _EXCEL_NAMES.exists():
        try:
            df_map = pd.read_excel(str(_EXCEL_NAMES))
            for _, row in df_map.iterrows():
                beamm = str(row.get("New_variable_name",      "")).strip()
                orig  = str(row.get("Original_variable_name", "")).strip()
                db    = str(row.get("Original_database",       "")).strip()
                if not beamm or beamm == "nan":
                    continue
                if orig and orig != "nan":
                    beamm_to_orig[beamm] = orig
                    orig_to_beamm[orig]  = beamm   # clé principale
                if db and db != "nan":
                    beamm_to_db[beamm] = db
            log.info(f"   Names_mapping : {len(orig_to_beamm)} mappings "
                     f"orig→BEAMM chargés")
        except Exception as e:
            log.warning(f"⚠️  Names_mapping_28_04.xlsx : {e}")
    else:
        log.warning(f"⚠️  {_EXCEL_NAMES} introuvable")

    # ── Source 2 : Variable_names_BEAMM.xlsx ─────────────────────────────────
    # Source des descriptions lisibles en anglais (class_full, main_full...)
    # Complète aussi orig_to_beamm pour les variables non couvertes par Source 1
    if _EXCEL_VARS.exists():
        try:
            for sheet in _SHEETS_WITH_DATA:
                df = pd.read_excel(str(_EXCEL_VARS), sheet_name=sheet)
                if "New_variable_name" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    code = str(row.get("New_variable_name", "")).strip()
                    if not code or code == "nan":
                        continue
                    # Description
                    desc = str(row.get("Description", ""))
                    if desc == "nan" or not desc:
                        parts = [
                            str(row.get(c, ""))
                            for c in ["class_full", "main_full", "info1_full"]
                            if str(row.get(c, "")) not in ("nan", "")
                        ]
                        desc = " — ".join(parts) if parts else code
                    beamm_to_desc[code] = desc
                    # Complément orig si absent
                    orig = str(row.get("Original_variable_name", ""))
                    db   = str(row.get("Original_database",       sheet))
                    if orig and orig != "nan":
                        if code not in beamm_to_orig:
                            beamm_to_orig[code] = orig
                        if orig not in orig_to_beamm:
                            orig_to_beamm[orig] = code
                    if db and db != "nan" and code not in beamm_to_db:
                        beamm_to_db[code] = db
            log.info(f"   Variable_names : {len(beamm_to_desc)} descriptions chargées")
        except Exception as e:
            log.warning(f"⚠️  Variable_names_BEAMM.xlsx : {e}")
    else:
        log.warning(f"⚠️  {_EXCEL_VARS} introuvable")

    log.info(f"✅ Excel mappings : {len(orig_to_beamm)} orig→BEAMM | "
             f"{len(beamm_to_desc)} descriptions")
    return beamm_to_desc, beamm_to_orig, beamm_to_db, orig_to_beamm


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISATION AU DÉMARRAGE
# ══════════════════════════════════════════════════════════════════════════════

RDS_DIR   = Path("data/rds")
CACHE_DIR = Path("data/cache")
REAL_CVAE = RDS_DIR / "beamm.brussels-250528-CVAE.rds"
REAL_GAN  = RDS_DIR / "beamm.brussels-250528-GAN.rds"

log.info("=" * 65)
log.info("STEP 9 — Démarrage de l'interface unifiée")
log.info("=" * 65)

# ── 1. Mappings Excel ─────────────────────────────────────────────────────────
log.info("[1/5] Chargement des mappings Excel...")
BEAMM_TO_DESC, BEAMM_TO_ORIG, BEAMM_TO_DB, ORIG_TO_BEAMM = load_excel_mappings()

# ── 2. RAG + ChromaDB + E5-Large (step5) ─────────────────────────────────────
log.info("[2/5] Initialisation du moteur RAG (E5-Large + ChromaDB)...")
_RAG_APP = None
_CHROMA  = None
_MODEL   = None
MSG_RAG  = ""

if STEP5_CLASS is not None:
    try:
        _RAG_APP = STEP5_CLASS()
        _CHROMA  = _RAG_APP.rag.collections.get("unified")
        _MODEL   = _RAG_APP.rag.model
        MSG_RAG  = "✅ RAG opérationnel (E5-Large + ChromaDB)"
        log.info(MSG_RAG)
    except Exception as e:
        MSG_RAG = f"⚠️  RAG non disponible ({e})"
        log.warning(MSG_RAG)
else:
    MSG_RAG = "⚠️  step5_app2.py introuvable — recherche sémantique désactivée"
    log.warning(MSG_RAG)

# ── 3. Mapping BEAMM → variable_id ChromaDB ──────────────────────────────────
log.info("[3/5] Construction du mapping BEAMM → ChromaDB variable_id...")
VARIABLE_ID_MAP: Dict[str, str] = {}
if _RAG_APP is not None:
    rag = _RAG_APP.rag
    # Itérer sur variables_by_code (indexé par codes originaux HY090G, HWUSUAL...)
    for orig_code, var_dict in rag.variables_by_code.items():
        vid = var_dict.get("variable_id", "")
        if not vid:
            continue
        # Lier les codes BEAMM qui pointent vers ce code original
        for beamm_code, orig in BEAMM_TO_ORIG.items():
            if orig == orig_code:
                VARIABLE_ID_MAP[beamm_code] = vid

log.info(f"   {len(VARIABLE_ID_MAP)} codes BEAMM liés à un variable_id ChromaDB")

# ── 4. Datasets CVAE + GAN ────────────────────────────────────────────────────
log.info("[4/5] Chargement des datasets...")


def _load(label: str, cache: Path, rds: Path):
    if cache.exists():
        try:
            loader = DataLoader.from_cache(cache)
            return loader, f"✅ {label} : {loader.df.shape[0]:,} obs × {loader.df.shape[1]} vars (cache)"
        except Exception:
            pass
    if rds.exists():
        try:
            loader = DataLoader.from_single_file(rds, label=label)
            loader.save_cache(cache)
            return loader, f"✅ {label} : {loader.df.shape[0]:,} obs × {loader.df.shape[1]} vars"
        except Exception as e:
            return None, f"❌ {label} : {e}"
    return None, f"❌ {label} : fichier introuvable ({rds.name})"


CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOADER_CVAE, MSG_CVAE = _load("CVAE", CACHE_DIR / "v1.parquet", REAL_CVAE)
LOADER_GAN,  MSG_GAN  = _load("GAN",  CACHE_DIR / "v2.parquet", REAL_GAN)

# ── 5. Statut global ──────────────────────────────────────────────────────────
log.info("[5/5] Initialisation terminée")
STATUT = "\n".join([MSG_RAG, MSG_CVAE, MSG_GAN,
    f"📐 Mapping : {len(VARIABLE_ID_MAP)} distances cosinus réelles disponibles",
    f"📚 Descriptions : {len(BEAMM_TO_DESC)} variables avec libellé lisible",
])

# Variables de démo connues dans les fichiers BEAMM
DEMO_VARS = (
    "yin_hyg_sm ypt_hyg_sm yto_hyg_sm "
    "xadbepi_hm_hsm xadbest_hm_hsm "
    "xhoenel_hm_hsn xhoenga_hm_hsn "
    "dctcd21_h_dmn dctcdnb_h_dmn"
)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_q(q: float) -> str:
    if pd.isna(q): return "—"
    if q < 1e-300: return "< 1e-300"
    if q < 1e-10:  return f"{q:.2e}"
    if q < 0.001:  return f"{q:.4e}"
    return f"{q:.4f}"


def _label(code: str) -> str:
    """Description courte d'un code BEAMM pour l'affichage."""
    desc = BEAMM_TO_DESC.get(code, "")
    if desc and desc != code:
        return f"{code} — {desc[:55]}{'…' if len(desc) > 55 else ''}"
    return code


def _parse_codes(text: Optional[str], ref_loader: Optional[DataLoader]) -> Tuple[List[str], bool]:
    """
    Extrait les codes BEAMM valides depuis un champ texte.
    Approche simple par split — plus robuste que la regex avec \\b et underscores.
    """
    text = text or ""
    if not text.strip():
        return [], True

    # Normaliser les séparateurs : virgules, points-virgules, espaces multiples
    normalized = text.replace(",", " ").replace(";", " ").replace("\n", " ")
    tokens     = [t.strip().lower() for t in normalized.split() if t.strip()]

    # Garder uniquement les tokens qui ressemblent à des codes BEAMM
    # (contiennent un underscore et ne sont pas des mots courants)
    STOP = {"les","des","une","est","par","sur","avec","dans","pour",
            "variable","entre","query","the","and","for","with","this"}
    candidates = [t for t in tokens if "_" in t and t not in STOP and len(t) >= 4]

    if not candidates:
        return [], True

    if ref_loader is None:
        return candidates[:20], False  # pas de validation possible

    # Valider contre les colonnes du DataFrame BEAMM
    # Utiliser un set pour la rapidité et normaliser les deux côtés
    col_set = {str(c).strip().lower() for c in ref_loader.df.columns}
    valid   = [c for c in candidates if c in col_set]

    if valid:
        return valid[:20], False
    return [], True


def _format_df(df: pd.DataFrame, add_desc: bool = True,
               full_df=None) -> pd.DataFrame:
    """Formate un DataFrame de résultats pour Gradio.
    #2 décomposition SB, #5 rang+percentile, #9 badge inter-bases, M9.9 N<100.
    """
    if df.empty:
        return pd.DataFrame()
    cols = ["var_semantique", "var_associee", "type_paire",
            "r2_unifie", "q_value", "surprise_b", "n_obs",
            "dist_sem", "loess_pred", "residuel"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    if "q_value" in out.columns:
        out["q_value"] = out["q_value"].apply(_fmt_q)
    for c in ["r2_unifie", "surprise_b", "dist_sem", "loess_pred", "residuel"]:
        if c in out.columns:
            out[c] = out[c].round(4)
    # #5 — Rang et percentile de Surprise_B
    if "surprise_b" in out.columns and full_df is not None and len(full_df) > 0:
        ref_sb  = full_df["surprise_b"].values
        m_total = len(ref_sb)
        out["rang"]  = out["surprise_b"].apply(lambda v: int((ref_sb >= v).sum()))
        out["top_%"] = out["rang"].apply(lambda r: f"top {r/m_total*100:.1f}%")
    # M9.9 — Flag N < 100
    if "n_obs" in out.columns:
        out.insert(
            out.columns.get_loc("n_obs") + 1,
            "alerte",
            out["n_obs"].apply(lambda n: "⚠️ N<100" if pd.notna(n) and int(n) < 100 else "")
        )
    # #9 — Badge inter-bases
    if "var_semantique" in out.columns and "var_associee" in out.columns:
        def _badge(row):
            db1 = BEAMM_TO_DB.get(row["var_semantique"], "")
            db2 = BEAMM_TO_DB.get(row["var_associee"],   "")
            if db1 and db2 and db1 != db2:
                return f"🔀 {db1}↔{db2}"
            return db1 or ""
        out["enquêtes"] = out.apply(_badge, axis=1)
    # Description
    if add_desc and "var_associee" in out.columns:
        def _get_desc(c):
            desc = BEAMM_TO_DESC.get(c, "")
            if desc and desc != c:
                return desc[:65]
            orig = BEAMM_TO_ORIG.get(c, "")
            db   = BEAMM_TO_DB.get(c, "")
            if orig and orig != "nan":
                return f"({orig} — {db})" if db else f"({orig})"
            return "—"
        out.insert(out.columns.get_loc("var_associee") + 1,
                   "description", out["var_associee"].map(_get_desc))
    # #2 — Décomposition Surprise_B
    if all(c in out.columns for c in ["r2_unifie", "loess_pred", "residuel", "surprise_b"]):
        out["décomposition"] = out.apply(
            lambda r: (f"R²={r['r2_unifie']:.3f} | attendu={r['loess_pred']:.3f}"
                       f" | ε={r['residuel']:+.3f} | SB={r['surprise_b']:.2f}"),
            axis=1
        )
        out = out.drop(columns=["loess_pred", "residuel"], errors="ignore")
    out.columns = [c.replace("_", " ").title()
                   if c not in ["enquêtes","alerte","rang","top_%","décomposition","description"]
                   else c for c in out.columns]
    return out.reset_index(drop=True)

def _describe_anchors(codes: List[str], loader: Optional[DataLoader]) -> str:
    """Affiche les ancres RAG avec type, description et code original."""
    if not codes or not loader:
        return ""
    lines = []
    for c in codes[:20]:
        vtype = loader.types.get(c, "?")
        icon  = "📊" if vtype == "quant" else "🏷️"
        desc  = BEAMM_TO_DESC.get(c, "")
        orig  = BEAMM_TO_ORIG.get(c, "")
        db    = BEAMM_TO_DB.get(c,   "")
        if desc and desc != c:
            label = f" — *{desc[:60]}*"
        elif orig:
            label = f" — *(code original : {orig}, {db})*"
        else:
            label = f" — *(description non disponible dans les fichiers Excel)*"
        lines.append(f"{icon} `{c}` ({vtype}){label}")
    return "\n".join(lines)


def _build_volcano_df(res_v1, res_v2) -> Tuple[pd.DataFrame, float]:
    """Retourne (df_volcano, tau_moyen) pour l'affichage."""
    frames = []
    tauss  = []
    for res, lbl in [(res_v1, "CVAE"), (res_v2, "WGAN")]:
        if res is None or res.all_filtered.empty:
            continue
        df = res.all_filtered.copy()
        if "surprise_b" not in df.columns or "r2_unifie" not in df.columns:
            continue
        df["dataset"]        = lbl
        df["Partition"]     = (df["surprise_b"] >= res.seuil_b).map(
                                    {True: "S_découverte ●", False: "S_attendu ○"})
        if "q_value" in df.columns:
            df["q_fmt"] = df["q_value"].apply(_fmt_q)
        cols = ["var_associee", "type_paire", "r2_unifie", "surprise_b",
                "Partition", "n_obs", "dataset"] + \
               (["q_fmt"] if "q_fmt" in df.columns else [])
        frames.append(df[cols])
        tauss.append(res.seuil_b)
    tau_mean = float(np.mean(tauss)) if tauss else 0.0
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), tau_mean


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS GRADIO
# ══════════════════════════════════════════════════════════════════════════════

def recherche_semantique(
    question    : str,
    top_k       : int,
    use_llm     : bool,
    enquetes    : List[str],
):
    """
    Tab 1 — Recherche sémantique.
    Retourne : (html_cards, codes_state, banner_md)
    """
    EMPTY_STATE = []
    EMPTY_MD    = ""

    if not question.strip():
        return "<p>⚠️ Entrez une question.</p>", EMPTY_STATE, EMPTY_MD

    if _RAG_APP is None:
        return (
            "<p>❌ Moteur RAG non disponible.</p>",
            EMPTY_STATE, EMPTY_MD
        )

    try:
        filters = enquetes if enquetes else None

        # Résultats bruts depuis le RAG
        raw_results = _RAG_APP.rag.search_by_question(
            question, top_k=top_k, survey_filter=filters
        )

        if not raw_results:
            return "<p>❌ Aucun résultat.</p>", EMPTY_STATE, EMPTY_MD

        # Extraire les codes BEAMM
        ref_loader = LOADER_CVAE or LOADER_GAN
        rds_cols   = set(ref_loader.df.columns) if ref_loader else set()
        codes        = []   # codes BEAMM présents dans le .rds
        codes_absent = []   # codes RAG sans équivalent BEAMM dans le .rds

        for r in raw_results:
            meta      = r.get("metadata", {})
            orig_code = meta.get("code", "")
            if not orig_code:
                continue
            # Stratégie 1 : code RAG directement dans le .rds
            if orig_code in rds_cols:
                if orig_code not in codes:
                    codes.append(orig_code)
                continue
            # Stratégie 2 : mapping inverse orig_code → BEAMM
            beamm = ORIG_TO_BEAMM.get(orig_code)
            if beamm and beamm in rds_cols:
                if beamm not in codes:
                    codes.append(beamm)
                continue
            # Stratégie 3 : absent du .rds
            codes_absent.append(orig_code)

        n_absent = len(codes_absent)

        # Analyser les codes absents : mappés correctement mais pas dans le .rds
        codes_mapped_absent = []   # mappés via Names_mapping mais absents du .rds
        codes_unmapped      = []   # pas dans Names_mapping du tout
        for orig in codes_absent:
            beamm = ORIG_TO_BEAMM.get(orig)
            if beamm:
                codes_mapped_absent.append((orig, beamm))  # (orig_code, beamm_code)
            else:
                codes_unmapped.append(orig)

        # HTML depuis step5
        html = _RAG_APP.search_by_question(question, top_k, use_llm, filters)

        # Bandeau de transfert — diagnostic complet
        if codes:
            # Afficher chaque variable injectée avec son nom et sa description
            ref_loader_for_type = LOADER_CVAE or LOADER_GAN
            injected_details = []
            for c in codes:
                vtype = ref_loader_for_type.types.get(c, "?") if ref_loader_for_type else "?"
                desc  = BEAMM_TO_DESC.get(c, "")
                desc_part = f" — *{desc[:60]}*" if desc and desc != c else ""
                icon = "📊" if vtype == "quant" else "🏷️"
                injected_details.append(f"{icon} `{c}` ({vtype}){desc_part}")
            vars_list = "\n".join(injected_details)
            banner = (
                f"✅ **{len(codes)} variable(s) BEAMM injectée(s)** dans l'onglet Associations :\n\n"
                f"{vars_list}"
            )
            if codes_mapped_absent:
                detail = ", ".join(
                    f"`{orig}`→`{beamm}`"
                    for orig, beamm in codes_mapped_absent[:4]
                )
                suf = f"... (+{len(codes_mapped_absent)-4})" if len(codes_mapped_absent) > 4 else ""
                banner += (
                    f"\n\n⚠️ **{len(codes_mapped_absent)} variable(s) correctement mappée(s) "
                    f"via Names_mapping mais absente(s) du dataset BEAMM :**\n"
                    f"{detail}{suf}\n"
                    f"*(Ces codes BEAMM existent dans le fichier de mapping mais ne font pas "
                    f"partie des {518}–{523} variables synthétisées dans les fichiers .rds. "
                    f"La synthèse CVAE/GAN couvre principalement les variables de revenus `y*`, "
                    f"dépenses `x*` et démographie `dct*`.)*"
                )
        else:
            # Diagnostiquer pourquoi rien n'a été trouvé
            if codes_mapped_absent:
                detail = "\n".join(
                    f"- `{orig}` → `{beamm}` (mappé mais absent du .rds)"
                    for orig, beamm in codes_mapped_absent[:6]
                )
                banner = (
                    f"⚠️ **0 variable BEAMM dans le dataset pour cette requête.**\n\n"
                    f"**Diagnostic :** {len(codes_mapped_absent)} variable(s) ont été mappée(s) "
                    f"correctement via Names_mapping_28_04.xlsx, mais leurs codes BEAMM "
                    f"ne sont pas dans les fichiers .rds (CVAE/GAN) :\n\n"
                    f"{detail}\n\n"
                    f"**Cause :** Le dataset BEAMM synthétique (518–523 variables) "
                    f"est un sous-ensemble du dictionnaire complet (1 827 variables). "
                    f"Les variables EU-LFS 'heures de travail' (`l*` prefix) et d'autres "
                    f"catégories n'ont pas été incluses dans cette synthèse.\n\n"
                    f"**Solutions :**\n"
                    f"- Essayez une requête sur les revenus, dépenses ou données démographiques\n"
                    f"- Utilisez **Charger démonstration** pour voir les variables disponibles\n"
                    f"- Filtrez par **EU-SILC** ou **DEMOBEL** qui ont meilleure couverture"
                )
            else:
                abst = ", ".join(f"`{c}`" for c in codes_absent[:5])
                suf  = f"... (+{len(codes_absent)-5})" if len(codes_absent) > 5 else ""
                banner = (
                    f"⚠️ **0 variable BEAMM trouvée** — codes RAG non mappés : {abst}{suf}\n\n"
                    f"Ces codes ne figurent pas dans Names_mapping_28_04.xlsx.\n"
                    f"Essayez une requête sur les revenus, dépenses ou démographie."
                )

        return html, codes, banner

    except Exception as e:
        log.error(f"Recherche sémantique échouée : {e}", exc_info=True)
        return f"<p>❌ Erreur : {e}</p>", EMPTY_STATE, EMPTY_MD


def calculer_associations(
    requete       : str,
    codes_state   : List[str],
    dataset_choix : str,
    top_n         : int,
    alpha_fdr     : float,
    alpha_surprise: float,
):
    """
    Tab 2 — Calcul des associations.
    Utilise codes_state si non vide (depuis Tab 1),
    sinon parse requete pour codes BEAMM manuels.

    Retourne : (statut, disc_cvae, att_cvae, disc_gan, att_gan,
                comparaison, volcano_df, anchors_md)
    """
    EMPTY = pd.DataFrame()
    requete = requete or ""   # Gradio envoie None quand Textbox vide

    ref_loader = LOADER_CVAE or LOADER_GAN
    if ref_loader is None:
        return "❌ Aucun dataset chargé.", EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, "", None, None

    # ── Sélection des variables ancres ───────────────────────────────────────
    # Priorité 1 : codes injectés depuis Tab 1 (recherche sémantique)
    # Protection contre codes_state de type inattendu (pd.Series, None...)
    try:
        _cs_valid = isinstance(codes_state, list) and len(codes_state) > 0
    except Exception:
        _cs_valid = False

    if _cs_valid:
        col_set    = {str(c).strip().lower() for c in ref_loader.df.columns}
        query_vars = [c for c in codes_state if str(c).strip().lower() in col_set]
        source     = f"🔗 {len(query_vars)} variables issues de la recherche sémantique"
    else:
        # Priorité 2 : codes saisis manuellement dans la Textbox
        query_vars, is_fallback = _parse_codes(requete, ref_loader)
        if is_fallback or not query_vars:
            # Construire un message d'aide avec les colonnes disponibles
            sample_cols = list(ref_loader.df.columns)[:5] if ref_loader else []
            hint = f"\n\nExemples de codes valides : `{'`, `'.join(sample_cols)}`" if sample_cols else ""
            return (
                "❌ **Aucun code de variable valide.**\n\n"
                "**Solutions :**\n"
                "- Lancez d'abord une **Recherche sémantique** (onglet 🔎)\n"
                "- Entrez des codes BEAMM séparés par des espaces\n"
                "- Cliquez **Charger démonstration** pour un exemple prêt à l'emploi"
                f"{hint}",
                EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, "", None, None
            )
        source = f"✏️ {len(query_vars)} variables saisies manuellement"

    anchors_md = f"**Variables ancres ({len(query_vars)}) — {source} :**\n\n"
    anchors_md += _describe_anchors(query_vars, ref_loader)

    # ── Calcul des associations ───────────────────────────────────────────────
    res_v1 = res_v2 = None

    try:
        for label, loader, res_ref in [
            ("CVAE", LOADER_CVAE, None),
            ("GAN",  LOADER_GAN,  None),
        ]:
            if dataset_choix not in (label, "Les deux") or loader is None:
                continue
            qv = [v for v in query_vars if v in loader.df.columns]
            eng = AssociationEngine(loader)
            res = eng.run(
                query_variables = qv,
                top_n           = top_n,
                alpha_fdr       = alpha_fdr,
                alpha_surprise  = alpha_surprise,
            )
            # Vraies distances sémantiques via mapping Excel + ChromaDB
            if (_CHROMA or _MODEL) and res and not res.all_filtered.empty:
                res = eng.recompute_serendipity(
                    res,
                    top_n             = top_n,
                    alpha_surprise    = alpha_surprise,
                    chroma_collection = _CHROMA,
                    rag_model         = _MODEL,
                    variable_id_map   = VARIABLE_ID_MAP,
                    text_map          = BEAMM_TO_DESC,
                )
            if label == "CVAE":
                res_v1 = res
            else:
                res_v2 = res

    except Exception as e:
        log.error(f"Calcul associations : {e}", exc_info=True)
        return f"❌ Erreur : {e}", EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, anchors_md, None, None

    # ── Formatage ─────────────────────────────────────────────────────────────
    df_disc_cvae = _format_df(res_v1.discoveries, full_df=res_v1.all_filtered) if res_v1 else EMPTY
    df_att_cvae  = _format_df(res_v1.expected,    full_df=res_v1.all_filtered) if res_v1 else EMPTY
    df_disc_gan  = _format_df(res_v2.discoveries, full_df=res_v2.all_filtered) if res_v2 else EMPTY
    df_att_gan   = _format_df(res_v2.expected,    full_df=res_v2.all_filtered) if res_v2 else EMPTY
    comparaison  = AssociationResults.compare(res_v1, res_v2) \
                   if res_v1 and res_v2 else ""
    volcano_df, tau_moyen = _build_volcano_df(res_v1, res_v2)
    if not volcano_df.empty:
        volcano_df["tau"] = tau_moyen

    # ── Statut ─────────────────────────────────────────────────────────────────
    parties = []
    for res, lbl in [(res_v1, "CVAE"), (res_v2, "WGAN")]:
        if res is None:
            continue
        n_sig  = len(res.all_filtered)
        n_disc = len(res.discoveries)
        loess  = "LOESS ✅" if res.model_fitted else "LOESS ⚠️"
        real_d = res.all_filtered["dist_sem"].nunique() > 3 \
                 if "dist_sem" in res.all_filtered.columns else False
        dist_s = "📐 distances réelles ✅" if real_d else "📐 proxy binaire ⚠️"
        parties.append(
            f"**{lbl}** : {n_sig:,} associations | {n_disc} découvertes | {loess} | {dist_s}"
        )
    if res_v1 and res_v2 and not res_v1.discoveries.empty and not res_v2.discoveries.empty:
        d1 = set(res_v1.discoveries["var_associee"].tolist())
        d2 = set(res_v2.discoveries["var_associee"].tolist())
        n_com = len(d1 & d2)
        parties.append(f"✅ **{n_com} découvertes communes aux deux synthèses** (stables entre CVAE et WGAN)")

    statut = "\n\n".join(parties) if parties else "⚠️ Aucun résultat."

    # M9.8 — Avertissement méthodologique
    statut += (
        "\n\n---\n"
        "> ⚠️ **Interprétation** : analyses réalisées sur données synthétiques "
        "(CVAE / WGAN). Les associations identifiées sont exploratoires et ne "
        "constituent pas une validation externe indépendante."
    )

    # M9.2 — Résumé numérique compact
    if res_v1 or res_v2:
        n_test  = sum(len(r.all_raw)      if r and hasattr(r, "all_raw") else 0
                      for r in [res_v1, res_v2])
        n_sig   = sum(len(r.all_filtered) if r else 0 for r in [res_v1, res_v2])
        n_disc  = sum(len(r.discoveries)  if r else 0 for r in [res_v1, res_v2])
        d1_set  = set(res_v1.discoveries["var_associee"]) if res_v1 and not res_v1.discoveries.empty else set()
        d2_set  = set(res_v2.discoveries["var_associee"]) if res_v2 and not res_v2.discoveries.empty else set()
        n_com_r = len(d1_set & d2_set)
        taux    = f"{n_disc/max(n_sig,1)*100:.1f}%" if n_sig > 0 else "—"
        anchors_md = (
            anchors_md
            + f"\n\n**Résumé :** {n_sig:,} associations significatives"
            f" | {n_disc} découvertes ({taux})"
            f" | {n_com_r} communes CVAE+WGAN"
        )

    # M9.3 + #3 — Comparaison tabulaire enrichie avec SurpriseB
    comparaison_df = pd.DataFrame()
    if res_v1 and res_v2:
        d1 = set(res_v1.discoveries["var_associee"].tolist()) if not res_v1.discoveries.empty else set()
        d2 = set(res_v2.discoveries["var_associee"].tolist()) if not res_v2.discoveries.empty else set()
        communes  = sorted(d1 & d2)
        seul_cvae = sorted(d1 - d2)
        seul_wgan = sorted(d2 - d1)

        # #3 — Stabilité : pour les communes, SurpriseB CVAE vs WGAN côte à côte
        rows_stab = []
        for var in communes:
            row_c = res_v1.discoveries[res_v1.discoveries["var_associee"] == var]
            row_g = res_v2.discoveries[res_v2.discoveries["var_associee"] == var]
            if not row_c.empty and not row_g.empty:
                r2c = float(row_c["r2_unifie"].iloc[0])
                r2g = float(row_g["r2_unifie"].iloc[0])
                sbc = float(row_c["surprise_b"].iloc[0])
                sbg = float(row_g["surprise_b"].iloc[0])
                rows_stab.append({
                    "Variable": var,
                    "R²_CVAE": round(r2c, 4), "R²_WGAN": round(r2g, 4),
                    "ΔR²":     round(r2c - r2g, 4),
                    "SB_CVAE": round(sbc, 3),  "SB_WGAN": round(sbg, 3),
                    "ΔSB":     round(sbc - sbg, 3),
                })
        df_stabilite = pd.DataFrame(rows_stab) if rows_stab else pd.DataFrame()

        # #11 — Résumé complet avec spécifiques
        n_total = len(d1 | d2)
        comparaison_df = pd.DataFrame([
            {"Critère": "Associations significatives |S|",
             "CVAE": len(res_v1.all_filtered), "WGAN": len(res_v2.all_filtered)},
            {"Critère": "S_découverte",
             "CVAE": len(res_v1.discoveries),  "WGAN": len(res_v2.discoveries)},
            {"Critère": "S_attendu",
             "CVAE": len(res_v1.expected),     "WGAN": len(res_v2.expected)},
            {"Critère": "Communes aux deux synthèses",
             "CVAE": len(communes), "WGAN": len(communes)},
            {"Critère": "Spécifiques seulement",
             "CVAE": len(seul_cvae), "WGAN": len(seul_wgan)},
            {"Critère": f"Taux stabilité ({len(communes)}/{n_total} communes)",
             "CVAE": f"{len(communes)/max(n_total,1)*100:.0f}%",
             "WGAN": f"{len(communes)/max(n_total,1)*100:.0f}%"},
            {"Critère": "Variables communes (aperçu)",
             "CVAE": ", ".join(communes[:4]) + ("…" if len(communes) > 4 else ""),
             "WGAN": "← idem"},
            {"Critère": "Spécifiques CVAE (aperçu)",
             "CVAE": ", ".join(seul_cvae[:3]) + ("…" if len(seul_cvae) > 3 else "") or "—",
             "WGAN": "—"},
            {"Critère": "Spécifiques WGAN (aperçu)",
             "CVAE": "—",
             "WGAN": ", ".join(seul_wgan[:3]) + ("…" if len(seul_wgan) > 3 else "") or "—"},
        ])
    else:
        df_stabilite = pd.DataFrame()

    return (statut, df_disc_cvae, df_att_cvae,
            df_disc_gan, df_att_gan, comparaison_df,
            df_stabilite, volcano_df, anchors_md,
            res_v1, res_v2)



# ── B — Mini-graphique LOESS interactif ──────────────────────────────────
def _fiche_loess(idx, dataset, partition, res_v1, res_v2):
    """
    Deux panneaux :
    - Gauche  : positionnement LOESS (pourquoi le score est élevé)
    - Droite  : relation brute Xi vs Yj selon type de paire
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d as sc_interp1d

    res = res_v1 if dataset == "CVAE" else res_v2
    loader = LOADER_CVAE if dataset == "CVAE" else LOADER_GAN

    if res is None or res.all_filtered.empty:
        return None, "*Lancez un calcul d'abord.*"
    pool = res.discoveries if partition == "S_découverte" else res.expected
    if pool.empty or int(idx) >= len(pool):
        return None, f"*Index {int(idx)} hors plage (max {len(pool)-1}).*"

    row    = pool.iloc[int(idx)]
    var1   = row.get("var_semantique", "")
    var2   = row.get("var_associee",   "")
    all_s  = res.all_filtered
    sb     = float(row.get("surprise_b", float("nan")))
    sigma  = getattr(res, "sigma_residual", float("nan"))
    tp     = str(row.get("type_paire", "quant↔quant"))

    # ── Données LOESS depuis all_filtered ───────────────────────
    x_all  = all_s["dist_sem"].values  if "dist_sem"   in all_s.columns else None
    y_all  = all_s["r2_unifie"].values if "r2_unifie"  in all_s.columns else None
    yp_all = all_s["loess_pred"].values if "loess_pred" in all_s.columns else None
    xp     = float(row.get("dist_sem",   float("nan")))
    yp_obs = float(row.get("r2_unifie",  float("nan")))

    # Interpoler yo depuis la courbe LOESS globale si manquant
    yo = float(row.get("loess_pred", float("nan")))
    if pd.isna(yo) and x_all is not None and yp_all is not None and not pd.isna(xp):
        try:
            order = x_all.argsort()
            f_int = sc_interp1d(x_all[order], yp_all[order],
                                bounds_error=False, fill_value="extrapolate")
            yo = float(f_int(xp))
        except Exception:
            yo = float("nan")

    res_val = yp_obs - yo if not pd.isna(yo) and not pd.isna(yp_obs) else float("nan")

    # ── Figure 2 panneaux ────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    # ── Panneau gauche : positionnement LOESS ───────────────────
    ax1.set_facecolor("#FAFAFA")
    if x_all is not None and y_all is not None:
        ax1.scatter(x_all, y_all, c="#AEC6E0", alpha=0.45, s=14,
                    label="$S$ (toutes assoc.)", zorder=2, edgecolors="none")
        if yp_all is not None:
            order  = x_all.argsort()
            ax1.plot(x_all[order], yp_all[order], "#17375E", lw=2.2,
                     label=r"LOESS $\hat{f}(d_{sém})$", zorder=4)

    if not pd.isna(xp) and not pd.isna(yp_obs):
        ax1.scatter([xp], [yp_obs], c="#D62728", s=130, zorder=6,
                    label=f"{var2[:20]}", edgecolors="white", linewidths=0.8)
    if not pd.isna(yo) and not pd.isna(xp):
        ax1.scatter([xp], [yo], c="#17375E", s=70, zorder=5, marker="^",
                    label=f"Attendu = {yo:.3f}", edgecolors="white", linewidths=0.5)
        if not pd.isna(yp_obs):
            ax1.annotate("",
                xy=(xp, yp_obs), xytext=(xp, yo),
                arrowprops=dict(arrowstyle="<->", color="#D62728", lw=1.6))
            mid_y = (yp_obs + yo) / 2
            ax1.text(xp + 0.003, mid_y,
                     f"ε̂={res_val:+.3f}\nSB={sb:.2f}",
                     fontsize=8, color="#D62728", va="center")

    ax1.set_xlabel("$d_{sém}(i,j)$", fontsize=10)
    ax1.set_ylabel("$R^2_{unifié}$", fontsize=10)
    ax1.set_title(f"Positionnement LOESS\n{dataset} · {var1[:18]} ↔ {var2[:18]}", fontsize=9.5)
    ax1.legend(fontsize=8, framealpha=0.95, loc="upper right")
    ax1.grid(True, alpha=0.22, ls="--")
    for s in ax1.spines.values(): s.set_edgecolor("#CCCCCC")

    # ── Panneau droit : relation brute Xi vs Yj ─────────────────
    ax2.set_facecolor("#FAFAFA")
    plotted = False

    if loader is not None and var1 in loader.df.columns and var2 in loader.df.columns:
        df_data = loader.df[[var1, var2]].dropna()
        n_pts   = len(df_data)

        if "quant↔quant" in tp:
            x_d = df_data[var1].astype(float)
            y_d = df_data[var2].astype(float)
            ax2.scatter(x_d, y_d, alpha=0.25, s=8, c="#AEC6E0", edgecolors="none")
            # Droite de régression
            if len(x_d) > 2:
                m, b = np.polyfit(x_d, y_d, 1)
                xr   = np.linspace(x_d.min(), x_d.max(), 100)
                ax2.plot(xr, m*xr + b, "#D62728", lw=2, label=f"R²={float(row.get('r2_unifie',0)):.3f}")
                ax2.legend(fontsize=9)
            ax2.set_xlabel(var1[:30], fontsize=9)
            ax2.set_ylabel(var2[:30], fontsize=9)
            ax2.set_title(f"Relation brute (quant↔quant)\nN={n_pts:,}", fontsize=9.5)
            plotted = True

        elif "quant↔cat" in tp or "cat↔quant" in tp:
            # Identifier quelle variable est quant et quelle est cat
            if loader.types.get(var1) == "quant":
                q_var, c_var = var1, var2
            else:
                q_var, c_var = var2, var1
            q_vals = df_data[q_var].astype(float)
            c_vals = df_data[c_var].astype(str)
            cats   = sorted(c_vals.unique())[:12]  # max 12 modalités
            data_bp = [q_vals[c_vals == cat].values for cat in cats]
            bp = ax2.boxplot(data_bp, patch_artist=True, notch=False)
            for patch in bp["boxes"]:
                patch.set_facecolor("#AEC6E0")
                patch.set_alpha(0.7)
            ax2.set_xticks(range(1, len(cats)+1))
            ax2.set_xticklabels([str(c)[:8] for c in cats], rotation=30, ha="right", fontsize=7)
            ax2.set_xlabel(c_var[:25], fontsize=9)
            ax2.set_ylabel(q_var[:25], fontsize=9)
            ax2.set_title(f"Boxplot (quant↔cat)\nN={n_pts:,}", fontsize=9.5)
            plotted = True

        elif "cat↔cat" in tp:
            c1_vals = df_data[var1].astype(str)
            c2_vals = df_data[var2].astype(str)
            cats1   = sorted(c1_vals.unique())[:10]
            cats2   = sorted(c2_vals.unique())[:10]
            mat = np.zeros((len(cats1), len(cats2)))
            for i, c1 in enumerate(cats1):
                for j, c2 in enumerate(cats2):
                    mat[i, j] = ((c1_vals == c1) & (c2_vals == c2)).sum()
            # Normaliser par ligne
            row_sums = mat.sum(axis=1, keepdims=True)
            mat_norm = np.divide(mat, row_sums, where=row_sums > 0)
            im2 = ax2.imshow(mat_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im2, ax=ax2, label="Proportion")
            ax2.set_xticks(range(len(cats2)))
            ax2.set_xticklabels([str(c)[:8] for c in cats2], rotation=30, ha="right", fontsize=7)
            ax2.set_yticks(range(len(cats1)))
            ax2.set_yticklabels([str(c)[:8] for c in cats1], fontsize=7)
            ax2.set_xlabel(var2[:25], fontsize=9)
            ax2.set_ylabel(var1[:25], fontsize=9)
            ax2.set_title(f"Contingence normalisée (cat↔cat)\nN={n_pts:,}", fontsize=9.5)
            plotted = True

    if not plotted:
        ax2.text(0.5, 0.5, "Données non disponibles\n(variables absentes du dataset)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10, color="#888")
        ax2.set_title("Relation brute", fontsize=9.5)

    for s in ax2.spines.values(): s.set_edgecolor("#CCCCCC")
    ax2.grid(True, alpha=0.22, ls="--")
    fig.suptitle(f"Fiche d'association — {dataset}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    # ── Phrase explicative ───────────────────────────────────────
    desc2 = BEAMM_TO_DESC.get(var2, var2)
    phrase = (
        f"**{var1} ↔ {var2}** — *{desc2[:60]}*\n\n"
        f"| Paramètre | Valeur |\n|---|---|\n"
        f"| d_sém | {xp:.3f} |\n"
        f"| R² attendu LOESS | {yo:.3f} |\n"
        f"| R² observé | {yp_obs:.3f} |\n"
        f"| Résidu ε̂ | {res_val:+.3f} |\n"
        f"| σ̂ résiduel | {sigma:.4f} |\n"
        f"| **Surprise_B** | **{sb:.2f}** |\n\n"
        f"Cette association dépasse de **{abs(sb):.2f} écarts-types** la tendance attendue "
        f"pour une distance sémantique d_sém = {xp:.3f}."
    )
    return fig, phrase


def _heatmap_surprise(res_v1, res_v2, dataset="CVAE"):
    """Heatmap Surprise_B — ne montre que S_découverte + top 30 S_attendu."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = res_v1 if dataset == "CVAE" else res_v2
    if res is None or res.all_filtered.empty or "surprise_b" not in res.all_filtered.columns:
        return None

    df   = res.all_filtered.copy()
    tau  = res.seuil_b if hasattr(res, "seuil_b") else df["surprise_b"].quantile(0.90)

    # Garder S_découverte + top 30 S_attendu (par R²)
    disc = df[df["surprise_b"] >= tau].copy()
    att  = df[df["surprise_b"] <  tau].nlargest(30, "r2_unifie").copy()
    df_plot = pd.concat([disc, att], ignore_index=True)
    df_plot["label"] = df_plot.apply(
        lambda r: ("● " if r["surprise_b"] >= tau else "○ ") +
                  r["var_associee"][:22], axis=1
    )
    df_plot = df_plot.sort_values("surprise_b", ascending=True)

    n_bars = len(df_plot)
    fig, ax = plt.subplots(figsize=(10, max(5, n_bars * 0.32)), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    colors = ["#D62728" if sb >= tau else "#AEC6E0"
              for sb in df_plot["surprise_b"]]
    bars = ax.barh(df_plot["label"], df_plot["surprise_b"],
                   color=colors, edgecolor="white", linewidth=0.4)

    # Ligne seuil τ
    ax.axvline(x=tau, color="#555", lw=1.5, ls="--", label=f"τ = {tau:.3f}")
    ax.set_xlabel("Surprise_B", fontsize=10)
    ax.set_title(
        f"Surprise_B par association — {dataset}\n"
        f"● S_découverte ({len(disc)}) · ○ S_attendu (top 30 affiché)",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.25, ls="--")
    for s in ax.spines.values(): s.set_edgecolor("#CCCCCC")
    plt.tight_layout()
    return fig


def _export_csv(res_v1, res_v2):
    import tempfile
    frames = []
    for res, lbl in [(res_v1, "CVAE"), (res_v2, "WGAN")]:
        if res and not res.discoveries.empty:
            df = res.discoveries.copy(); df["dataset"] = lbl
            frames.append(df)
    if not frames:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(tmp.name, index=False, sep=";")
    return tmp.name

def _export_latex(res_v1, res_v2):
    d1 = set(res_v1.discoveries["var_associee"]) if res_v1 and not res_v1.discoveries.empty else set()
    d2 = set(res_v2.discoveries["var_associee"]) if res_v2 and not res_v2.discoveries.empty else set()
    communes = d1 & d2
    rows = [
        "% Tableau généré par AssociationExplorer",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Var. sémantique & Var. associée & Type & $R^2$ & $q$ & $N$ & $d_{\text{sém}}$ & Surprise$_B$ & Stable \\",
        r"\midrule",
    ]
    for res, _ in [(res_v1, "CVAE"), (res_v2, "WGAN")]:
        if res is None or res.discoveries.empty:
            continue
        for _, row in res.discoveries.iterrows():
            va  = str(row["var_associee"]).replace("_", r"\_")
            vs  = str(row["var_semantique"]).replace("_", r"\_")
            tp  = str(row.get("type_paire","")).replace("↔","$\\leftrightarrow$")
            r2  = f"{float(row['r2_unifie']):.4f}"
            qv  = _fmt_q(row.get("q_value", 1.0))
            n   = str(int(row.get("n_obs", 0)))
            ds  = f"{float(row.get('dist_sem', 0)):.3f}" if "dist_sem" in row else "--"
            sb  = f"{float(row['surprise_b']):.3f}"
            stb = "\\checkmark" if row["var_associee"] in communes else "--"
            rows.append(rf"\textit{{{vs}}} & \textit{{{va}}} & {tp} & {r2} & {qv} & {n} & {ds} & {sb} & {stb} \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


# ── E — Interprétation LLM Ollama ────────────────────────────────────────
def _llm_interpretation(idx, dataset, partition, res_v1, res_v2):
    """Interprétation descriptive via Ollama — streaming, timeout 3 min."""
    import requests as _req, json as _json

    res  = res_v1 if dataset == "CVAE" else res_v2
    if res is None or res.all_filtered.empty:
        return "*Aucun résultat disponible.*"
    pool = res.discoveries if partition == "S_découverte" else res.expected
    if pool.empty or int(idx) >= len(pool):
        return "*Index hors plage.*"

    row   = pool.iloc[int(idx)]
    var1  = row.get("var_semantique", "")
    var2  = row.get("var_associee",   "")
    desc1 = BEAMM_TO_DESC.get(var1, var1)[:80]
    desc2 = BEAMM_TO_DESC.get(var2, var2)[:80]
    r2    = float(row.get("r2_unifie",  0))

    # Prompt court → moins de tokens → plus rapide sur CPU
    prompt = (
        f"Variable 1 : {var1} ({desc1}). "
        f"Variable 2 : {var2} ({desc2}). "
        f"R²={r2:.3f}, données synthétiques BEAMM. "
        f"En UNE phrase sans causalité, décris leur lien thématique. "
        f"Commence par 'Ces deux variables mesurent'."
    )
    try:
        resp = _req.post(
            "http://localhost:11434/api/generate",
            json={
                "model"  : "llama3.2",
                "prompt" : prompt,
                "stream" : True,
                "options": {"num_predict": 80, "temperature": 0.3},
            },
            timeout=180,
            stream=True,
        )
        resp.raise_for_status()
        result = ""
        for line in resp.iter_lines():
            if line:
                chunk = _json.loads(line)
                result += chunk.get("response", "")
                if chunk.get("done", False):
                    break
        return (
            "🤖 *Suggestion LLM — non validée, données synthétiques :*\n\n"
            + result.strip()
        )
    except _req.exceptions.ConnectionError:
        return "*Ollama non démarré. Lancez `ollama serve` dans un terminal.*"
    except _req.exceptions.Timeout:
        return (
            "*Délai dépassé (>3 min). Essayez un modèle plus léger :*\n\n"
            "```\nollama pull llama3.2:1b\n```\n"
            "Puis modifiez `'llama3.2'` → `'llama3.2:1b'` dans le code."
        )
    except Exception as e:
        return f"*Erreur LLM : {e}*"

def update_volcano(df_volcano, ds_filter, type_filter):
    """Filtre et retourne les données pour le volcano plot."""
    EMPTY = pd.DataFrame({
        "r2_unifie":[], "surprise_b":[], "type_paire":[],
        "var_associee":[], "n_obs":[], "dataset":[], "q_fmt":[],
    })
    if not isinstance(df_volcano, pd.DataFrame) or df_volcano.empty:
        return EMPTY, EMPTY
    df = df_volcano.copy()
    # Filtre dataset — "WGAN" correspond directement à la colonne dataset
    if ds_filter not in ("Tous", "") and "dataset" in df.columns:
        df = df[df["dataset"] == ds_filter]
    if type_filter and "type_paire" in df.columns:
        df = df[df["type_paire"].isin(type_filter)]
    return (df, df) if not df.empty else (EMPTY, EMPTY)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE L'INTERFACE
# ══════════════════════════════════════════════════════════════════════════════


# ── Palette UCLouvain ─────────────────────────────────────────────────────
UCL_BLUE_NIGHT = "#00204E"
UCL_BLUE       = "#005EB8"
UCL_BLUE_LIGHT = "#7FB2E5"
UCL_CONTENT_BG = "#F4F6F9"

_ucl_css = (
    "body, .gradio-container {"
    "  font-family: 'Segoe UI', Arial, sans-serif;"
    "  font-size: 15px !important;"
    "  background-color: #F4F6F9;"
    "}"
    ".tab-nav button {"
    "  font-size: 14px !important;"
    "  font-weight: 600;"
    "  color: #00204E !important;"
    "}"
    ".tab-nav button.selected {"
    "  border-bottom: 3px solid #005EB8 !important;"
    "  color: #005EB8 !important;"
    "}"
    "h1 { color: #00204E !important; font-size: 1.6rem !important; }"
    "h2 { color: #005EB8 !important; font-size: 1.25rem !important; }"
    "h3 { color: #005EB8 !important; font-size: 1.1rem !important; }"
    ".primary, button.primary {"
    "  background: #005EB8 !important;"
    "  border: none !important;"
    "}"
    ".primary:hover { background: #00204E !important; }"
    "table th {"
    "  background-color: #00204E !important;"
    "  color: #FFFFFF !important;"
    "  font-size: 14px !important;"
    "}"
    "table td { font-size: 14px !important; }"
    "input[type=range] { accent-color: #005EB8; }"
    ".info { color: #555 !important; font-size: 12px !important; }"
)

def build_interface() -> gr.Blocks:

    with gr.Blocks(title="AssociationExplorer — Interface Unifiée UCLouvain",
                   css=_ucl_css) as app:

        # ── En-tête ───────────────────────────────────────────────────────────
        gr.Markdown("""
<div style="background:#00204E;padding:18px 24px 14px;border-radius:8px;margin-bottom:8px;">
<span style="color:#7FB2E5;font-size:0.85rem;font-weight:600;letter-spacing:0.08em;">
UCLouvain · Institut de Statistique · BEAMM</span><br>
<span style="color:#FFFFFF;font-size:1.5rem;font-weight:700;">
🔍 AssociationExplorer — Interface Unifiée</span><br>
<span style="color:#BDD7F0;font-size:0.88rem;">
Recherche sémantique · Associations statistiques · Distances cosinus réelles ·
EU-SILC · HFCS · EU-LFS · HBS · IPCAL · DEMOBEL</span>
</div>
""")

        with gr.Accordion("📊 Statut du système", open=False):
            gr.Markdown(f"```\n{STATUT}\n```")

        # ── State partagé entre les tabs ──────────────────────────────────────
        codes_state    = gr.State([])            # codes Tab 1 → Tab 2
        viz_data_state = gr.State(pd.DataFrame())
        res_v1_state   = gr.State(None)          # AssociationResults CVAE
        res_v2_state   = gr.State(None)          # AssociationResults WGAN

        sem_calc_btn = None   # initialisé ici pour la connexion cross-tab
        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — RECHERCHE SÉMANTIQUE
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🔎 Recherche sémantique"):

                if _RAG_APP is not None:
                    gr.Markdown("""
                    ### Recherche dans le dictionnaire de 1827 variables
                    Posez votre question — les K variables les plus proches
                    sont automatiquement transmises à l'onglet **Associations**.
                    """)
                    with gr.Row():
                            with gr.Column(scale=3):
                                sem_question = gr.Textbox(
                                    label       = "Votre question",
                                    placeholder = "Ex : heures de travail, revenus immobiliers, précariété énergétique...",
                                    lines       = 2,
                                )
                            with gr.Column(scale=1):
                                sem_top_k   = gr.Slider(3, 30, value=10, step=1,
                                                        label="K variables retournées")
                                with gr.Accordion("⚙️ Options avancées", open=False):
                                    sem_use_llm = gr.Checkbox(
                                        label = "🤖 Validation Ollama/Llama",
                                        value = False,
                                        info  = "Ajoute une explication LLM (lent)"
                                    )
                    sem_enquetes = gr.CheckboxGroup(
                        ["EU-SILC","HFCS","EU-LFS","HBS","IPCAL","DEMOBEL"],
                        label="Filtrer par enquête (vide = toutes)"
                    )

                    with gr.Row():
                        sem_btn     = gr.Button("🔍 Rechercher", variant="primary", scale=2)
                        sem_calc_btn = gr.Button(
                            "🚀 → Calculer les associations",
                            variant="secondary", scale=1
                        )

                    sem_transfer_md = gr.Markdown(
                        "*Lancez une recherche pour obtenir des variables candidates.*"
                    )

                    # M9.4 — Tableau RAG avec scores
                    sem_results_table = gr.DataFrame(
                        interactive=False,
                        wrap=True,
                        visible=False,
                        label="Variables candidates (résultats RAG)",
                    )

                    # M9.11 — Désélection des variables
                    sem_var_selector = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="✅ Variables sélectionnées pour l'analyse — décochez pour exclure",
                        visible=False,
                        info="Seules les variables cochées seront utilisées comme ancres.",
                    )
                    sem_confirm_btn = gr.Button(
                        "✔ Confirmer la sélection → onglet Associations",
                        variant="secondary",
                        visible=False,
                        size="sm",
                    )

                    sem_output = gr.HTML()

                    def _search_and_update(question, top_k, use_llm, enquetes):
                        html, codes, banner = recherche_semantique(
                            question, top_k, use_llm, enquetes
                        )
                        if not codes:
                            # Aucune variable trouvée — masquer sélecteur
                            return (html, codes, banner,
                                    gr.DataFrame(visible=False),
                                    gr.CheckboxGroup(choices=[], value=[], visible=False),
                                    gr.Button(visible=False))

                        # M9.4 — Construire le tableau des résultats RAG
                        ref = LOADER_CVAE or LOADER_GAN
                        rows = []
                        for i, c in enumerate(codes, 1):
                            vtype = ref.types.get(c, "?") if ref else "?"
                            desc  = BEAMM_TO_DESC.get(c, "—")
                            orig  = BEAMM_TO_ORIG.get(c, "—")
                            db    = BEAMM_TO_DB.get(c,   "—")
                            rows.append({
                                "Rang"        : i,
                                "Code BEAMM"  : c,
                                "Description" : desc[:60] if desc != "—" else "—",
                                "Enquête"     : db,
                                "Type"        : vtype,
                                "Code original": orig,
                            })
                        df_rag = pd.DataFrame(rows)

                        # M9.11 — Choix pour le CheckboxGroup
                        choices = []
                        for c in codes:
                            desc = BEAMM_TO_DESC.get(c, "")
                            label_cb = f"{c} — {desc[:45]}" if desc and desc != c else c
                            choices.append((label_cb, c))

                        return (
                            html, codes, banner,
                            gr.DataFrame(value=df_rag, visible=True),
                            gr.CheckboxGroup(choices=choices, value=codes, visible=True),
                            gr.Button(visible=True),
                        )

                    def _confirm_selection(selected_codes):
                        """Met à jour codes_state avec les variables sélectionnées."""
                        return selected_codes

                    sem_btn.click(
                        fn      = _search_and_update,
                        inputs  = [sem_question, sem_top_k, sem_use_llm, sem_enquetes],
                        outputs = [sem_output, codes_state, sem_transfer_md,
                                   sem_results_table, sem_var_selector, sem_confirm_btn],
                    )

                    # Mise à jour immédiate de codes_state quand la sélection change
                    sem_var_selector.change(
                        fn=_confirm_selection,
                        inputs=[sem_var_selector],
                        outputs=[codes_state],
                    )

                    # Bouton de confirmation explicite (UX — rend la transition visible)
                    sem_confirm_btn.click(
                        fn=_confirm_selection,
                        inputs=[sem_var_selector],
                        outputs=[codes_state],
                    )

                else:
                    gr.Markdown(f"""
                    ### ⚠️ Moteur sémantique non disponible
                    `step5_app2.py` ou ses dépendances sont absentes.

                    **Pour activer :** assurez-vous que step3, step4 et step5 sont dans
                    le même dossier et que leurs dépendances sont installées.

                    **Pour tester maintenant :** utilisez le bouton **Charger démonstration**
                    dans l'onglet Associations — il pré-remplit avec des codes valides.

                    *Message système :* `{MSG_RAG}`
                    """)

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — ASSOCIATIONS STATISTIQUES
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("📊 Associations statistiques"):

                gr.Markdown("""
                ### Découverte d'associations statistiques — Score Surprise_B
                Si vous avez utilisé la **Recherche sémantique**, les variables sont
                déjà injectées. Sinon, entrez des codes BEAMM manuellement.
                """)

                with gr.Accordion("🔧 Saisie manuelle de codes BEAMM (optionnel)", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            assoc_requete = gr.Textbox(
                                label       = "Codes BEAMM manuels (si pas de recherche sémantique)",
                                placeholder = "Ex : yin_hyg_sm ypt_hyg_sm xadbest_hm_hsm",
                                lines       = 2,
                                info        = "Laissez vide si vous avez utilisé la recherche sémantique"
                            )
                        with gr.Column(scale=1):
                            assoc_dataset = gr.Radio(
                                ["CVAE", "WGAN", "Les deux"],
                                value = "Les deux",
                                label = "Dataset",
                            )

                    with gr.Row():
                        demo_btn  = gr.Button("📋 Charger démonstration", scale=1)
                        clear_btn = gr.Button("🗑️  Effacer codes", scale=1)

                with gr.Row():
                    assoc_top_n = gr.Slider(5, 30, value=15, step=5,
                                            label="Associations retournées (top_n)")
                    assoc_alpha_fdr = gr.Slider(0.01, 0.20, value=0.05, step=0.01,
                                                label="Seuil de significativité FDR (α)",
                                                info="Contrôle du taux de fausses découvertes — Benjamini-Hochberg (1995)")
                    assoc_alpha_sur = gr.Slider(0.01, 0.20, value=0.10, step=0.01,
                                                label="Seuil d'inattendu — Surprise_B (α)",
                                                info="Définit τ = Q₁₋α(Surprise_B). "
                                                     "α=0.10 → top 10% d'associations inattendues "
                                                     "(valeur calibrée empiriquement, section 3.6.11 du mémoire).")

                assoc_btn = gr.Button(
                    "🚀 Calculer les associations", variant="primary", size="lg"
                )

                # Ancres utilisées (P1)
                anchors_md = gr.Markdown(
                    "*Les variables ancres apparaîtront ici après le calcul.*"
                )
                assoc_statut = gr.Markdown("*Prêt.*")

                with gr.Tabs():
                    with gr.Tab("🔴 CVAE — Découvertes"):
                        gr.Markdown("#### Associations fortes et **inattendues** (CVAE)")
                        df_disc_cvae = gr.DataFrame(interactive=False, wrap=True)
                        gr.Markdown("#### Associations **attendues** (CVAE)")
                        df_att_cvae  = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("🔵 WGAN — Découvertes"):
                        gr.Markdown("#### Associations fortes et **inattendues** (WGAN)")
                        df_disc_gan  = gr.DataFrame(interactive=False, wrap=True)
                        gr.Markdown("#### Associations **attendues** (WGAN)")
                        df_att_gan   = gr.DataFrame(interactive=False, wrap=True)

                    with gr.Tab("⚖️  CVAE vs WGAN"):
                        gr.Markdown("""
#### Stabilité des découvertes communes
Les associations **communes aux deux synthèses** (stables entre CVAE et WGAN)
sont les plus fiables — elles apparaissent indépendamment du mécanisme génératif.
""")
                        comparaison_out = gr.DataFrame(
                            interactive=False, wrap=True,
                            label="Tableau comparatif CVAE / WGAN"
                        )
                        comparaison_detail = gr.Markdown("")
                        # #3 — Stabilité R² et Surprise_B côte à côte
                        gr.Markdown("#### #3 — Stabilité par association commune")
                        stabilite_out = gr.DataFrame(
                            interactive=False, wrap=True,
                            label="R², Surprise_B et écarts CVAE vs WGAN"
                        )

                # ── B — Mini-graphique LOESS + fiche détaillée ──────────
                with gr.Accordion("🔬 Fiche détaillée d'une association", open=False):
                    gr.Markdown("""
Entrez le numéro de ligne (index) d'une association dans les tableaux
pour voir son positionnement dans le modèle LOESS et la décomposition de Surprise_B.
""")
                    with gr.Row():
                        fiche_idx  = gr.Number(label="Index ligne", value=0, precision=0)
                        fiche_ds   = gr.Radio(["CVAE","WGAN"], value="CVAE", label="Dataset")
                        fiche_part = gr.Radio(["S_découverte","S_attendu"],
                                              value="S_découverte", label="Partition")
                    fiche_btn  = gr.Button("📍 Afficher la fiche", variant="secondary")
                    with gr.Row():
                        fiche_plot = gr.Plot(label="LOESS · Relation brute")
                        fiche_text = gr.Markdown("")
                    # #7 — Exploration en chaîne (version minimale)
                    explore_btn = gr.Button(
                        "🔗 → Utiliser la variable associée comme nouvelle ancre",
                        variant="secondary", size="sm"
                    )
                    # E — LLM Ollama (si disponible)
                    with gr.Accordion("🤖 Interprétation LLM (suggestion non validée)",
                                      open=False):
                        llm_btn = gr.Button("Générer une interprétation via Llama",
                                            variant="secondary")
                        llm_out = gr.Markdown(
                            "*Cliquez pour générer une interprétation descriptive "
                            "(non causale, données synthétiques).*"
                        )

                # ── #15 + F — Export ────────────────────────────────────
                with gr.Accordion("⬇️ Export des résultats", open=False):
                    with gr.Row():
                        export_csv_btn   = gr.Button("📄 Exporter CSV")
                        export_latex_btn = gr.Button("📐 LaTeX (format Tableau mémoire)")
                    export_file      = gr.File(label="Fichier CSV", visible=False)
                    export_latex_out = gr.Textbox(
                        label="Code LaTeX — copier dans le mémoire",
                        lines=8, visible=False, interactive=False
                    )

                # Boutons utilitaires
                demo_btn.click(lambda: DEMO_VARS, [], [assoc_requete])
                clear_btn.click(lambda: ("", []), [], [assoc_requete, codes_state])

                # ── B — Fiche LOESS ───────────────────────────────────────
                fiche_btn.click(
                    fn=_fiche_loess,
                    inputs=[fiche_idx, fiche_ds, fiche_part, res_v1_state, res_v2_state],
                    outputs=[fiche_plot, fiche_text],
                )
                # ── E — LLM ───────────────────────────────────────────────
                if llm_btn is not None and llm_out is not None:
                    llm_btn.click(
                        fn=_llm_interpretation,
                        inputs=[fiche_idx, fiche_ds, fiche_part, res_v1_state, res_v2_state],
                        outputs=[llm_out],
                    )
                # ── #7 — Exploration en chaîne ────────────────────────────
                def _extract_var_associee(idx, dataset, partition, rv1, rv2):
                    res  = rv1 if dataset == "CVAE" else rv2
                    if res is None: return ""
                    pool = res.discoveries if partition == "S_découverte" else res.expected
                    if pool.empty or int(idx) >= len(pool): return ""
                    return pool.iloc[int(idx)].get("var_associee", "")
                explore_btn.click(
                    fn=_extract_var_associee,
                    inputs=[fiche_idx, fiche_ds, fiche_part, res_v1_state, res_v2_state],
                    outputs=[assoc_requete],
                )
                # ── #15 — Export CSV ──────────────────────────────────────
                export_csv_btn.click(
                    fn=lambda rv1, rv2: (
                        gr.File(value=_export_csv(rv1, rv2), visible=True),
                        gr.Textbox(visible=False)
                    ),
                    inputs=[res_v1_state, res_v2_state],
                    outputs=[export_file, export_latex_out],
                )
                # ── F — Export LaTeX ──────────────────────────────────────
                export_latex_btn.click(
                    fn=lambda rv1, rv2: (
                        gr.File(visible=False),
                        gr.Textbox(value=_export_latex(rv1, rv2), visible=True)
                    ),
                    inputs=[res_v1_state, res_v2_state],
                    outputs=[export_file, export_latex_out],
                )

                # Calcul principal
                assoc_btn.click(
                    fn      = calculer_associations,
                    inputs  = [assoc_requete, codes_state, assoc_dataset,
                               assoc_top_n, assoc_alpha_fdr, assoc_alpha_sur],
                    outputs = [assoc_statut,
                               df_disc_cvae, df_att_cvae,
                               df_disc_gan,  df_att_gan,
                               comparaison_out, stabilite_out,
                               viz_data_state, anchors_md,
                               res_v1_state, res_v2_state],
                )

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 — VISUALISATION
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("📈 Visualisation"):

                gr.Markdown("""
## Graphique Force × Inattendu
*(R²_unifié : force statistique — Surprise_B : caractère inattendu)*

| Axe | Signification | Zone d'intérêt |
|---|---|---|
| **X** | R²_unifié (force) | Droite = association forte |
| **Y** | Surprise_B = ε̂/σ̂ | Haut = association inattendue |
| **Couleur** | S_découverte / S_attendu | Rouge = inattendu, Bleu = attendu |

                **Coin supérieur droit = S_découverte** (fort ET inattendu)

                Avec les vraies distances sémantiques, le graphique montre
                un **nuage dispersé** — les découvertes se distinguent clairement
                des associations attendues.

                *Lancez d'abord un calcul dans l'onglet Associations.*
                """)

                viz_btn = gr.Button("🔄 Actualiser la visualisation", variant="secondary")

                with gr.Row():
                    viz_ds_filter = gr.Radio(
                        ["Tous","CVAE","WGAN"], value="Tous", label="Dataset"
                    )
                    viz_type_filter = gr.CheckboxGroup(
                        ["quant↔quant","quant↔cat","cat↔cat"],
                        value=["quant↔quant","quant↔cat","cat↔cat"],
                        label="Types de paires"
                    )

                viz_plot = gr.ScatterPlot(
                    label      = "Force × Inattendu (R²_unifié vs Surprise_B)",
                    x          = "r2_unifie",
                    y          = "surprise_b",
                    color      = "Partition",
                    tooltip    = ["var_associee","type_paire","dataset",
                                  "r2_unifie","q_fmt","surprise_b","n_obs","tau"],
                    x_title    = "Force statistique (R²_unifié)",
                    y_title    = "Surprise_B = ε̂/σ̂ (caractère inattendu)",
                    caption    = (
                        "🔴 S_découverte ● (Surprise_B ≥ τ) = associations fortes ET inattendues  "
                        "·  🔵 S_attendu ○ (Surprise_B < τ) = associations prévisibles"
                    ),
                    height     = 500,
                )

                viz_table_data = gr.State(pd.DataFrame())

                viz_show_table = gr.Checkbox(label="Afficher les données brutes", value=False)
                viz_table = gr.DataFrame(interactive=False, visible=False, wrap=True)

                # A — Heatmap Surprise_B
                with gr.Accordion("🌡️ Heatmap Surprise_B", open=False):
                    gr.Markdown("Intensité de Surprise_B par paire (rouge = S_découverte).")
                    heatmap_ds  = gr.Radio(["CVAE","WGAN"], value="CVAE", label="Dataset")
                    heatmap_btn = gr.Button("Générer la heatmap", variant="secondary")
                    heatmap_plot = gr.Plot(label="Heatmap Surprise_B")
                    heatmap_btn.click(
                        fn=_heatmap_surprise,
                        inputs=[res_v1_state, res_v2_state, heatmap_ds],
                        outputs=[heatmap_plot],
                    )

                def _toggle_table(show: bool, df):
                    """Bascule visibilité ET données en un seul output."""
                    data = df if (show and isinstance(df, pd.DataFrame) and not df.empty)                                else pd.DataFrame()
                    return gr.DataFrame(visible=show, value=data)

                viz_btn.click(
                    fn=update_volcano,
                    inputs=[viz_data_state, viz_ds_filter, viz_type_filter],
                    outputs=[viz_plot, viz_table_data],
                )
                viz_show_table.change(
                    fn=_toggle_table,
                    inputs=[viz_show_table, viz_table_data],
                    outputs=[viz_table],   # un seul output suffit
                )
                for fc in [viz_ds_filter, viz_type_filter]:
                    fc.change(
                        fn=update_volcano,
                        inputs=[viz_data_state, viz_ds_filter, viz_type_filter],
                        outputs=[viz_plot, viz_table_data],
                    )

            # ══════════════════════════════════════════════════════════════════
            # TAB 4 — EXPLORATEUR DU DATASET
            with gr.Tab("🗂️ Dataset"):
                gr.Markdown("""
                ### Variables disponibles dans le dataset BEAMM

                Cherchez dans les variables réellement présentes dans les fichiers .rds (CVAE/GAN).
                Utile pour savoir quelles requêtes sémantiques auront des résultats.
                """)
                with gr.Row():
                    explore_search = gr.Textbox(
                        label="Chercher dans les variables disponibles",
                        placeholder="Ex : hyg, xad, dct, yin...",
                    )
                    explore_type = gr.Radio(
                        ["Toutes", "quant", "cat"],
                        value="Toutes", label="Type"
                    )
                explore_btn = gr.Button("🔍 Chercher", variant="secondary")
                explore_out = gr.DataFrame(interactive=False, wrap=True)

                def explorer_dataset(query: str, vtype: str):
                    """Cherche dans les colonnes du .rds."""
                    ref = LOADER_CVAE or LOADER_GAN
                    if ref is None:
                        return pd.DataFrame({"Erreur": ["Aucun dataset chargé"]})
                    query = (query or "").strip().lower()
                    rows = []
                    for col, ctype in ref.types.items():
                        if vtype != "Toutes" and ctype != vtype:
                            continue
                        if query and query not in col.lower():
                            continue
                        desc = BEAMM_TO_DESC.get(col, "")
                        orig = BEAMM_TO_ORIG.get(col, "")
                        db   = BEAMM_TO_DB.get(col,   "")
                        rows.append({
                            "Code BEAMM"  : col,
                            "Type"        : ctype,
                            "Description" : desc[:70] if desc else "—",
                            "Code original": orig or "—",
                            "Enquête"     : db   or "—",
                        })
                    if not rows:
                        return pd.DataFrame({"Info": [f"Aucune variable pour '{query}'"]})
                    df = pd.DataFrame(rows).sort_values("Code BEAMM").reset_index(drop=True)
                    return df

                explore_btn.click(
                    fn      = explorer_dataset,
                    inputs  = [explore_search, explore_type],
                    outputs = [explore_out],
                )
                # Re-filtrer quand le type change
                explore_type.change(
                    fn=explorer_dataset,
                    inputs=[explore_search, explore_type],
                    outputs=[explore_out],
                )

                # Stats du dataset
                if LOADER_CVAE:
                    n_q = sum(1 for t in LOADER_CVAE.types.values() if t == "quant")
                    n_c = sum(1 for t in LOADER_CVAE.types.values() if t == "cat")
                    n_desc = sum(1 for col in LOADER_CVAE.df.columns
                                 if col in BEAMM_TO_DESC and BEAMM_TO_DESC[col] != col)
                    gr.Markdown(f"""
                    **Dataset CVAE** : {LOADER_CVAE.df.shape[1]} variables
                    ({n_q} quantitatives, {n_c} catégorielles) ·
                    {n_desc} avec description lisible via Excel
                    """)

            # TAB 5 — GUIDE
            with gr.Tab("📖 Guide"):
                gr.Markdown(f"""
## Guide d'utilisation — Interface Unifiée

---

### Flux recommandé

**Étape 1 → Onglet 🔎 Recherche sémantique**
Entrez votre question en langage naturel. Le moteur RAG
(E5-Large + ChromaDB) retourne les K variables les plus proches
sémantiquement. Les codes sont **automatiquement transmis** à l'onglet
Associations — un bandeau vert confirme l'injection.

**Étape 2 → Onglet 📊 Associations statistiques**
Cliquez **Calculer les associations**. Le pipeline calcule :
- R²_unifié sur ~5000 paires (K × N variables)
- FDR (Benjamini-Hochberg) → ensemble S significatif
- Distances cosinus **réelles** (via mapping Excel + ChromaDB)
- Surprise_B (résidu LOESS standardisé)

**Étape 3 → Onglet 📈 Visualisation**
Cliquez **Actualiser** pour afficher le volcano plot.
Avec les vraies distances, le graphique montre un nuage dispersé :
les découvertes (haut-droite) se distinguent des associations attendues.

---

### Comment les variables sémantiques sont sélectionnées

**Avec le moteur RAG (Tab 1 actif) :**
E5-Large encode votre question en vecteur 1024D.
ChromaDB retourne les K variables dont la distance cosinus est minimale.

**Sans le RAG (codes manuels) :**
Entrez directement les codes BEAMM séparés par des espaces.
Le calcul est **bloqué** si aucun code valide n'est détecté.

---

### Pourquoi les distances sont réelles ici (pas dans step8)

Le mapping Excel (`Variable_names_BEAMM.xlsx`) relie :
- Code BEAMM (`yin_hyg_sm`) → Code original (`HY090G`)
- Code original → variable_id ChromaDB (`EU-SILC_HY090G`)
- variable_id → embedding E5-Large 1024D

Pour les variables sans code original, la description lisible
est encodée directement avec E5-Large — bien mieux que d'encoder
le code brut `yin_hyg_sm`.

Résultat : {len(VARIABLE_ID_MAP)} variables avec distance cosinus exacte,
{len(BEAMM_TO_DESC)} variables avec description lisible.

---

### Méthodes statistiques
- **R²_adj** : Pearson ajusté (Theil, 1961) — quant↔quant
- **ε²** : Epsilon-squared (Kelley, 1935) — quant↔cat
- **Ṽ²** : V de Cramér-Bergsma (2013) — cat↔cat
- **FDR** : Benjamini-Hochberg (1995) — contrôle des faux positifs
- **Surprise_B** : résidu LOESS standardisé — sérendipité statistique

---

### Statut du système
```
{STATUT}
```
""")

        # ── Connexion cross-tab : sem_calc_btn (Tab 1) → calcul (Tab 2) ────────
        # Doit être défini ici, APRÈS tous les composants Tab 2
        if _RAG_APP is not None and sem_calc_btn is not None:
            sem_calc_btn.click(
                fn      = calculer_associations,
                inputs  = [assoc_requete, codes_state, assoc_dataset,
                           assoc_top_n, assoc_alpha_fdr, assoc_alpha_sur],
                outputs = [assoc_statut,
                           df_disc_cvae, df_att_cvae,
                           df_disc_gan,  df_att_gan,
                           comparaison_out, stabilite_out,
                           viz_data_state, anchors_md,
                           res_v1_state, res_v2_state],
            )

    return app


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("STEP 9 — AssociationExplorer Interface Unifiée")
    print("=" * 65)
    print(STATUT)
    print()

    if LOADER_CVAE is None and LOADER_GAN is None:
        print("❌ Aucun dataset chargé. Vérifiez data/rds/ et data/cache/")
        sys.exit(1)

    app = build_interface()
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7862,
        share       = False,
        show_error  = True,
        inbrowser   = True,
    )
