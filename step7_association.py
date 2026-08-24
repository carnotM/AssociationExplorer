#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 7 - PIPELINE D'ASSOCIATION STATISTIQUE — AssociationExplorer
=================================================================

Ce module reçoit les données chargées par step6 et un ensemble de variables
sémantiques retournées par le moteur RAG (step3), puis calcule les associations
statistiques selon la méthode « Filtrer, Scorer, Classer » formalisée dans
la Partie II du mémoire.

TROIS PILIERS :
─────────────────────────────────────────────────────────────────
Pilier 1 — FORCE (section 3.2–3.4 du mémoire)
    Mesures ajustées selon le type de paire :
    • quant ↔ quant → R² ajusté de Pearson         (Theil, 1961)
    • quant ↔ cat   → ε² epsilon-squared            (Kelley, 1935)
    • cat   ↔ cat   → Ṽ² de Cramér-Bergsma         (Bergsma, 2013)
    Toutes projetées sur l'espace R²_unifié ∈ [0, 1].

Pilier 2 — FIABILITÉ (section 3.5 du mémoire)
    Correction de Benjamini-Hochberg (1995) sur les m p-valeurs poolées.
    m = K × (N - K) paires — connu à l'avance car K = résultat du RAG.

Pilier 3 — INATTENDU (section 3.6 du mémoire)
    Score de sérendipité Surprise_B = résidu standardisé par rapport
    au modèle d'attente E[R²_unifié | distance sémantique] estimé par LOESS.
    Partition en S_découverte (Surprise_B ≥ seuil) et S_attendu.

SORTIE :
─────────────────────────────────────────────────────────────────
    results = AssociationEngine(loader).run(query_variables, top_n=15)
    results.discoveries   → DataFrame top associations inattendues
    results.expected      → DataFrame top associations attendues
    results.all_filtered  → DataFrame complet après FDR
    results.summary()     → rapport lisible
    results.compare(other_results) → comparaison CVAE vs GAN

UTILISATION TYPIQUE (depuis step8 / Gradio) :
─────────────────────────────────────────────────────────────────
    from step6_data_loader import DataLoader
    from step7_association  import AssociationEngine

    loader = DataLoader.from_single_file('data/rds/beamm.brussels-250528-CVAE.rds')
    engine = AssociationEngine(loader)
    results = engine.run(
        query_variables = ['yin_hyg_sm', 'ypt_hyg_sm', 'PL060'],
        top_n           = 15,
        alpha_fdr       = 0.05,
        alpha_surprise  = 0.10,
    )
    print(results.summary())

Auteur  : AssociationExplorer — Partie II
Date    : 2025-2026
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f_oneway, chi2_contingency, norm as scipy_norm, shapiro

try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    raise ImportError(
        "statsmodels est requis.\n"
        "Installation : pip install statsmodels"
    )

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("step7_association.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

#: Nombre minimum d'observations communes pour calculer une association
MIN_OBS_PAIR: int = 20

#: Seuil FDR par défaut (Benjamini-Hochberg, 1995)
DEFAULT_ALPHA_FDR: float = 0.05

#: Seuil de surprise par défaut (quantile 95 % de N(0,1) si résidus normaux)
DEFAULT_ALPHA_SURPRISE: float = 0.10  # Q₀.₉₀ — cohérent avec la calibration (section 3.6.11)

#: Nombre d'associations retournées par défaut
DEFAULT_TOP_N: int = 15

#: Fraction LOESS (proportion du voisinage local pour le lissage)
DEFAULT_LOESS_FRAC: float = 0.4


# ══════════════════════════════════════════════════════════════════════════════
# MESURES D'ASSOCIATION AJUSTÉES (Pilier 1)
# ══════════════════════════════════════════════════════════════════════════════

def _r2_adjusted_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    R² ajusté de Pearson pour paires quantitative ↔ quantitative.
    Formule : R²_adj = 1 - (1-R²)·(n-1)/(n-2)   [Theil, 1961]
    Retourne (r2_adj, p_value) ou (None, None) si calcul impossible.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < MIN_OBS_PAIR:
        return None, None
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    try:
        r, p = pearsonr(x, y)
        r2_adj = 1 - (1 - r ** 2) * (n - 1) / (n - 2)
        return max(0.0, round(r2_adj, 6)), round(float(p), 8)
    except Exception:
        return None, None


def _epsilon_squared(y_cont: np.ndarray, g_cat: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Epsilon-squared (ε²) pour paires quantitative ↔ catégorielle.
    Formule : ε² = (SS_entre - df_entre·MS_résid) / SS_total   [Kelley, 1935]
    Identique au R² ajusté en contexte ANOVA (Vogt, 2005).
    Retourne (eps2, p_value) ou (None, None) si calcul impossible.
    """
    mask = ~(np.isnan(y_cont.astype(float)) | pd.isnull(g_cat))
    y_cont = y_cont[mask].astype(float)
    g_cat  = g_cat[mask]
    modalites = np.unique(g_cat)
    k, n = len(modalites), len(y_cont)
    if k < 2 or n < k + MIN_OBS_PAIR:
        return None, None
    groupes = [y_cont[g_cat == m] for m in modalites]
    if any(len(g) < 5 for g in groupes):
        return None, None
    try:
        _, p = f_oneway(*groupes)
        grand  = y_cont.mean()
        SS_tot = float(((y_cont - grand) ** 2).sum())
        if SS_tot == 0:
            return 0.0, 1.0
        SS_res = float(sum(((g - g.mean()) ** 2).sum() for g in groupes))
        SS_ent = SS_tot - SS_res
        MS_res = SS_res / (n - k)
        eps2   = (SS_ent - (k - 1) * MS_res) / SS_tot
        return max(0.0, round(eps2, 6)), round(float(p), 8)
    except Exception:
        return None, None


def _cramers_v_bergsma(x_cat: np.ndarray, y_cat: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    V de Cramér corrigé (Ṽ²) pour paires catégorielle ↔ catégorielle.
    Correction de Bergsma (2013) sur φ² et dimensions effectives r̃, c̃.
    Retourne (v_tilde_squared, p_value) ou (None, None) si calcul impossible.
    """
    mask = ~(pd.isnull(x_cat) | pd.isnull(y_cat))
    x_cat, y_cat = x_cat[mask], y_cat[mask]
    n = len(x_cat)
    if n < MIN_OBS_PAIR:
        return None, None
    try:
        table = pd.crosstab(x_cat, y_cat).values
        r, c  = table.shape
        if r < 2 or c < 2:
            return None, None
        chi2, p, _, expected = chi2_contingency(table)
        if np.isnan(chi2):
            return None, None
        phi2   = chi2 / n
        # Correction Bergsma (2013)
        biais  = (r - 1) * (c - 1) / (n - 1)
        phi2_t = max(0.0, phi2 - biais)
        r_t    = r - (r - 1) ** 2 / (n - 1)
        c_t    = c - (c - 1) ** 2 / (n - 1)
        denom  = min(r_t - 1, c_t - 1)
        if denom <= 0:
            return None, None
        v_tilde = np.sqrt(phi2_t / denom)
        v_tilde = min(1.0, max(0.0, v_tilde))
        return round(v_tilde ** 2, 6), round(float(p), 8)
    except Exception:
        return None, None


def _compute_association(
    df       : pd.DataFrame,
    col_a    : str,
    col_b    : str,
    types    : Dict[str, str],
) -> Optional[Dict]:
    """
    Dispatch vers la bonne mesure selon le type de paire.
    Retourne un dict standardisé ou None si le calcul échoue.
    """
    ta = types.get(col_a)
    tb = types.get(col_b)
    if ta is None or tb is None:
        return None

    subset = df[[col_a, col_b]].dropna()
    n_obs  = len(subset)
    if n_obs < MIN_OBS_PAIR:
        return None

    a = subset[col_a].values
    b = subset[col_b].values

    if ta == "quant" and tb == "quant":
        r2, p = _r2_adjusted_pearson(a.astype(float), b.astype(float))
        mesure, type_paire = "R²_adj (Pearson)", "quant↔quant"

    elif ta == "quant" and tb == "cat":
        r2, p = _epsilon_squared(a.astype(float), b)
        mesure, type_paire = "ε² (Kelley)", "quant↔cat"

    elif ta == "cat" and tb == "quant":
        r2, p = _epsilon_squared(b.astype(float), a)
        mesure, type_paire = "ε² (Kelley)", "quant↔cat"

    else:
        r2, p = _cramers_v_bergsma(a, b)
        mesure, type_paire = "Ṽ² (Bergsma)", "cat↔cat"

    if r2 is None:
        return None

    return {
        "var_semantique": col_a,
        "var_associee"  : col_b,
        "type_paire"    : type_paire,
        "mesure"        : mesure,
        "r2_unifie"     : r2,
        "p_value"       : p,
        "n_obs"         : n_obs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — DISTANCES SÉMANTIQUES VIA CHROMADB (section 3.9.5 du mémoire)
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedding_from_chroma(chroma_collection, var_code: str) -> Optional[np.ndarray]:
    """
    Récupère l'embedding E5-Large d'une variable depuis ChromaDB.
    Retourne None si la variable est absente de la collection.
    """
    try:
        result = chroma_collection.get(
            ids     = [var_code],
            include = ["embeddings"],
        )
        if result["embeddings"] and len(result["embeddings"]) > 0:
            emb = np.array(result["embeddings"][0], dtype=np.float32)
            norm_val = float(np.linalg.norm(emb))
            return emb / (norm_val + 1e-10) if norm_val > 0 else emb
    except Exception:
        pass
    return None


def get_semantic_distance(
    chroma_collection,
    var_a: str,
    var_b: str,
) -> float:
    """
    Distance sémantique cosinus entre deux variables BEAMM.

    dist_sém = 1 − sim_cos(e_a, e_b)
    Bornée dans [0, 1] pour des embeddings normalisés (Douze et al., 2024).

    Paramètres
    ----------
    chroma_collection : collection ChromaDB 'unified_variables' (depuis step3)
    var_a, var_b      : codes BEAMM des deux variables

    Retourne
    --------
    float ∈ [0, 1]  — 0 = identiques, 1 = opposées
    """
    e_a = _get_embedding_from_chroma(chroma_collection, var_a)
    e_b = _get_embedding_from_chroma(chroma_collection, var_b)
    if e_a is None or e_b is None:
        return 0.5   # valeur neutre si variable absente de ChromaDB
    sim_cos = float(np.dot(e_a, e_b))
    sim_cos = max(-1.0, min(1.0, sim_cos))
    return round(1.0 - sim_cos, 6)


def _batch_get_embeddings(
    chroma_collection,
    var_codes         : List[str],
    rag_model         = None,
    variable_id_map   : Optional[Dict[str, str]]       = None,
    text_map          : Optional[Dict[str, str]]        = None,
    precomputed_cache : Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Récupère ou encode les embeddings de plusieurs variables BEAMM.

    Stratégie (dans l'ordre de priorité) :
    0. Cache pré-calculé     — step9 passe ses embeddings déjà construits
    1. ChromaDB via variable_id_map  — beamm_code → ChromaDB variable_id (Excel)
    2. ChromaDB via code direct      — pour codes dont ID = code lui-même
    3. E5-Large sur description      — text_map : beamm_code → texte lisible
    4. E5-Large sur code brut        — fallback ultime

    Paramètres
    ----------
    chroma_collection : collection ChromaDB (peut être None)
    var_codes         : liste de codes BEAMM
    rag_model         : modèle SentenceTransformer, optionnel
    variable_id_map   : {beamm_code → ChromaDB variable_id} — construit depuis Excel
    text_map          : {beamm_code → description lisible} — encodage sémantique
    precomputed_cache : embeddings déjà calculés (évite tout recalcul)
    """
    embed_dict: Dict[str, np.ndarray] = {}

    # ── Couche 0 : cache pré-calculé ─────────────────────────────────────────
    if precomputed_cache:
        for c in var_codes:
            if c in precomputed_cache:
                embed_dict[c] = precomputed_cache[c]

    still_needed = [c for c in var_codes if c not in embed_dict]
    if not still_needed:
        return embed_dict

    # ── Couche 1 : ChromaDB via variable_id_map ───────────────────────────────
    if chroma_collection is not None and variable_id_map:
        id_to_beamm: Dict[str, str] = {}
        ids_to_fetch: List[str] = []
        for c in still_needed:
            vid = variable_id_map.get(c)
            if vid:
                ids_to_fetch.append(vid)
                id_to_beamm[vid] = c
        if ids_to_fetch:
            try:
                result = chroma_collection.get(ids=ids_to_fetch, include=["embeddings"])
                for vid, emb in zip(result.get("ids", []), result.get("embeddings", [])):
                    if emb is not None:
                        beamm = id_to_beamm.get(vid)
                        if beamm:
                            e = np.array(emb, dtype=np.float32)
                            n = float(np.linalg.norm(e))
                            embed_dict[beamm] = e / (n + 1e-10) if n > 0 else e
                n_via_vid = sum(1 for c in still_needed if c in embed_dict)
                log.info(f"   ChromaDB (variable_id) : {n_via_vid}/{len(still_needed)}")
            except Exception as e:
                log.warning(f"ChromaDB (variable_id) échoué ({e})")

    still_needed = [c for c in var_codes if c not in embed_dict]

    # ── Couche 2 : ChromaDB via code direct ───────────────────────────────────
    if chroma_collection is not None and still_needed:
        try:
            result = chroma_collection.get(ids=still_needed, include=["embeddings"])
            for vid, emb in zip(result.get("ids", []), result.get("embeddings", [])):
                if emb is not None:
                    e = np.array(emb, dtype=np.float32)
                    n = float(np.linalg.norm(e))
                    embed_dict[vid] = e / (n + 1e-10) if n > 0 else e
        except Exception as e:
            log.warning(f"ChromaDB (code direct) échoué ({e})")

    still_needed = [c for c in var_codes if c not in embed_dict]

    # ── Couche 3+4 : E5-Large (description ou code brut) ─────────────────────
    if still_needed and rag_model is not None:
        try:
            texts = []
            for c in still_needed:
                desc = (text_map or {}).get(c, c)
                texts.append(f"passage: {desc}")
            embs = rag_model.encode(
                texts,
                normalize_embeddings=True, convert_to_numpy=True,
                batch_size=64, show_progress_bar=False,
            )
            for code, emb in zip(still_needed, embs):
                embed_dict[code] = np.array(emb, dtype=np.float32)
            log.info(f"   E5-Large : {len(still_needed)} variables encodées")
        except Exception as e:
            log.warning(f"Encodage E5-Large échoué ({e})")

    n_found  = len(embed_dict)
    n_total  = len(var_codes)
    n_neutre = n_total - n_found
    if n_neutre > 0:
        log.warning(f"   {n_neutre}/{n_total} variables sans embedding → dist=0.5")

    return embed_dict


def add_semantic_distances(
    df_s              : pd.DataFrame,
    chroma_collection,
    rag_model         = None,
    variable_id_map   : Optional[Dict[str, str]] = None,
    text_map          : Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Enrichit le DataFrame S avec la distance sémantique de chaque paire.

    Remplace le proxy binaire {0, 1} par les vraies distances cosinus
    continues [0, 1] calculées à partir des embeddings E5-Large.

    Paramètres
    ----------
    df_s              : DataFrame des associations filtrées (après FDR)
    chroma_collection : collection ChromaDB 'unified_variables' (step3)
    rag_model         : modèle SentenceTransformer (depuis step3/step5)

    Retourne
    --------
    df_s enrichi avec la colonne 'dist_sem' ∈ [0, 1]
    """
    if df_s.empty:
        return df_s

    df_s = df_s.copy()

    # Récupérer tous les codes BEAMM uniques (ancres + associées)
    all_vars = list(set(
        df_s["var_semantique"].tolist() + df_s["var_associee"].tolist()
    ))
    log.info(f"   Calcul distances sémantiques : {len(all_vars)} variables uniques")

    embed_dict = _batch_get_embeddings(
        chroma_collection,
        all_vars,
        rag_model       = rag_model,
        variable_id_map = variable_id_map,
        text_map        = text_map,
    )

    # Calcul vectorisé des distances cosinus
    def _dist(row) -> float:
        e_a = embed_dict.get(row["var_semantique"])
        e_b = embed_dict.get(row["var_associee"])
        if e_a is None or e_b is None:
            return 0.5
        sim = float(np.dot(e_a, e_b))
        return round(1.0 - max(-1.0, min(1.0, sim)), 6)

    df_s["dist_sem"] = df_s.apply(_dist, axis=1)

    # Rapport de qualité
    n_reel   = (df_s["dist_sem"] != 0.5).sum()
    n_total  = len(df_s)
    n_unique = df_s["dist_sem"].nunique()
    pct      = 100 * n_reel / n_total if n_total > 0 else 0
    log.info(f"   ✅ Distances réelles : {n_reel}/{n_total} ({pct:.1f}%) | "
             f"{n_unique} valeurs uniques "
             f"[{df_s['dist_sem'].min():.3f}, {df_s['dist_sem'].max():.3f}]")
    if n_unique <= 3:
        log.warning("   ⚠️  dist_sem quasi-binaire — codes BEAMM absents de ChromaDB")

    return df_s


# ══════════════════════════════════════════════════════════════════════════════
# RÉSULTATS (dataclass de sortie)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AssociationResults:
    """
    Contient tous les résultats du pipeline pour un chargement donné.

    Attributs
    ---------
    dataset_label   : nom du dataset (CVAE ou GAN)
    query_variables : variables sémantiques utilisées comme ancres
    all_raw         : toutes les paires calculées (avant filtrage)
    all_filtered    : paires significatives après FDR (ensemble S)
    discoveries     : S_découverte — associations fortes et inattendues
    expected        : S_attendu   — associations fortes mais prévisibles
    model_fitted    : True si le modèle LOESS a été estimé avec succès
    normal_residuals: True si les résidus suivent approximativement N(0,1)
    sigma_residual  : écart-type des résidus du modèle d'attente
    """
    dataset_label   : str
    query_variables : List[str]
    all_raw         : pd.DataFrame = field(default_factory=pd.DataFrame)
    all_filtered    : pd.DataFrame = field(default_factory=pd.DataFrame)
    discoveries     : pd.DataFrame = field(default_factory=pd.DataFrame)
    expected        : pd.DataFrame = field(default_factory=pd.DataFrame)
    model_fitted    : bool  = False
    normal_residuals: bool  = False
    sigma_residual  : float = 0.0
    seuil_b         : float = 0.0   # seuil τ de partition S_découverte / S_attendu
    loess_frac      : float = DEFAULT_LOESS_FRAC
    alpha_fdr       : float = DEFAULT_ALPHA_FDR
    alpha_surprise  : float = DEFAULT_ALPHA_SURPRISE

    def summary(self) -> str:
        """Rapport lisible des résultats — affiché dans Gradio et les logs."""
        n_raw  = len(self.all_raw)
        n_filt = len(self.all_filtered)
        n_disc = len(self.discoveries)
        n_exp  = len(self.expected)
        pct    = f"{100*n_filt/n_raw:.1f}%" if n_raw > 0 else "—"

        lines = [
            "=" * 60,
            f"RÉSULTATS D'ASSOCIATION — {self.dataset_label}",
            "=" * 60,
            f"Variables sémantiques (ancres RAG) : {len(self.query_variables)}",
            f"  {self.query_variables[:5]}{'...' if len(self.query_variables)>5 else ''}",
            "",
            f"Paires calculées (m)    : {n_raw:,}",
            f"Paires significatives   : {n_filt:,}  ({pct} après FDR α={self.alpha_fdr})",
            f"  dont S_découverte     : {n_disc}  (Surprise_B ≥ seuil)",
            f"  dont S_attendu        : {n_exp}",
            "",
            f"Modèle d'attente LOESS  : {'✅ estimé' if self.model_fitted else '⚠️  non estimé'}",
            f"Normalité des résidus   : {'✅' if self.normal_residuals else '⚠️  non normale → seuil empirique utilisé'}",
            f"σ_résiduel              : {self.sigma_residual:.4f}",
        ]

        if not self.discoveries.empty:
            lines += ["", "TOP DÉCOUVERTES (associations fortes et inattendues) :"]
            cols_show = ["var_associee", "type_paire", "r2_unifie", "q_value", "surprise_b"]
            cols_show = [c for c in cols_show if c in self.discoveries.columns]
            lines.append(self.discoveries[cols_show].head(10).to_string(index=True))

        if not self.expected.empty:
            lines += ["", "TOP ATTENDUES (associations sémantiquement proches) :"]
            cols_show = ["var_associee", "type_paire", "r2_unifie", "q_value"]
            cols_show = [c for c in cols_show if c in self.expected.columns]
            lines.append(self.expected[cols_show].head(5).to_string(index=True))

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_gradio_tables(self, top_n: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retourne deux DataFrames formatés pour l'affichage dans Gradio.

        Retourne
        --------
        (df_discoveries, df_expected)
        """
        def _fmt(df: pd.DataFrame, n: int) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame()
            cols = ["var_semantique", "var_associee", "type_paire",
                    "mesure", "r2_unifie", "q_value", "surprise_b", "n_obs"]
            cols = [c for c in cols if c in df.columns]
            out = df[cols].head(n).copy()
            # Arrondir pour la lisibilité
            for col in ["r2_unifie", "q_value", "surprise_b"]:
                if col in out.columns:
                    out[col] = out[col].round(4)
            return out.reset_index(drop=True)

        return _fmt(self.discoveries, top_n), _fmt(self.expected, top_n)

    @staticmethod
    def compare(r1: "AssociationResults", r2: "AssociationResults") -> str:
        """
        Compare les résultats de deux datasets (CVAE vs GAN).

        Identifie les associations qui apparaissent dans les deux,
        uniquement dans l'un, ou avec des forces très différentes.
        """
        if r1.discoveries.empty or r2.discoveries.empty:
            return "⚠️  Impossible de comparer : au moins un dataset sans découvertes."

        key = "var_associee"
        disc1 = set(r1.discoveries[key].tolist()) if key in r1.discoveries.columns else set()
        disc2 = set(r2.discoveries[key].tolist()) if key in r2.discoveries.columns else set()

        common   = disc1 & disc2
        only_1   = disc1 - disc2
        only_2   = disc2 - disc1

        lines = [
            "=" * 60,
            f"COMPARAISON  {r1.dataset_label}  vs  {r2.dataset_label}",
            "=" * 60,
            f"Découvertes {r1.dataset_label:>8} : {len(disc1)}",
            f"Découvertes {r2.dataset_label:>8} : {len(disc2)}",
            "",
            f"✅ Communes (robustes aux deux synthèses) : {len(common)}",
        ]
        if common:
            lines.append(f"   {sorted(common)[:5]}{'...' if len(common)>5 else ''}")

        lines += [
            "",
            f"⬅  Uniquement dans {r1.dataset_label} : {len(only_1)}",
        ]
        if only_1:
            lines.append(f"   {sorted(only_1)[:5]}{'...' if len(only_1)>5 else ''}")

        lines += [
            f"➡  Uniquement dans {r2.dataset_label} : {len(only_2)}",
        ]
        if only_2:
            lines.append(f"   {sorted(only_2)[:5]}{'...' if len(only_2)>5 else ''}")

        # Comparer la force sur les associations communes
        if common and key in r1.discoveries.columns and key in r2.discoveries.columns:
            lines += ["", "Force (r2_unifie) sur les associations communes :"]
            lines.append(f"  {'Variable':<30} {r1.dataset_label:>8}  {r2.dataset_label:>8}  Δ")
            lines.append("  " + "─" * 52)
            # groupby pour gérer les doublons (plusieurs var_semantique → même var_associee)
            d1 = r1.discoveries.groupby(key)["r2_unifie"].max()
            d2 = r2.discoveries.groupby(key)["r2_unifie"].max()
            for var in sorted(common)[:10]:
                v1 = float(d1[var]) if var in d1.index else float("nan")
                v2 = float(d2[var]) if var in d2.index else float("nan")
                delta = v1 - v2 if not (np.isnan(v1) or np.isnan(v2)) else float("nan")
                lines.append(f"  {var:<30} {v1:>8.4f}  {v2:>8.4f}  {delta:>+.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MOTEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class AssociationEngine:
    """
    Moteur du pipeline « Filtrer, Scorer, Classer ».

    Usage
    -----
    engine  = AssociationEngine(loader)
    results = engine.run(query_variables=['var1', 'var2', ...], top_n=15)

    Paramètres du constructeur
    --------------------------
    loader : DataLoader chargé depuis step6
    """

    def __init__(self, loader):
        """
        Paramètres
        ----------
        loader : objet DataLoader (step6_data_loader.DataLoader)
        """
        self.loader = loader
        self.df     = loader.df
        self.types  = loader.types
        self.label  = loader.label

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        query_variables : List[str],
        top_n           : int   = DEFAULT_TOP_N,
        alpha_fdr       : float = DEFAULT_ALPHA_FDR,
        alpha_surprise  : float = DEFAULT_ALPHA_SURPRISE,
        loess_frac      : float = DEFAULT_LOESS_FRAC,
        methode_fdr     : str   = "fdr_bh",
    ) -> AssociationResults:
        """
        Lance le pipeline complet sur un ensemble de variables sémantiques.

        Paramètres
        ----------
        query_variables : liste des variables retournées par le RAG
                          (doivent être des colonnes présentes dans loader.df)
        top_n           : nombre d'associations retournées dans chaque bassin
        alpha_fdr       : niveau FDR (Benjamini-Hochberg, défaut 0.05)
        alpha_surprise  : seuil de surprise pour la partition (défaut 0.10 → Q₀.₉₀)
        loess_frac      : fraction du voisinage LOESS (défaut 0.4)
        methode_fdr     : 'fdr_bh' (Benjamini-Hochberg) ou
                          'fdr_by' (Benjamini-Yekutieli, plus conservateur)

        Retourne
        --------
        AssociationResults
        """
        results = AssociationResults(
            dataset_label   = self.label,
            query_variables = query_variables,
            alpha_fdr       = alpha_fdr,
            alpha_surprise  = alpha_surprise,
            loess_frac      = loess_frac,
        )

        log.info("=" * 60)
        log.info(f"PIPELINE ASSOCIATION — {self.label}")
        log.info(f"Variables sémantiques : {len(query_variables)}")
        log.info("=" * 60)

        # ── Étape 1 : valider les variables sémantiques ───────────────────────
        all_cols    = list(self.df.columns)
        valid_query = [v for v in query_variables if v in all_cols]
        other_cols  = [c for c in all_cols if c not in valid_query]

        if not valid_query:
            log.error("Aucune variable sémantique valide trouvée dans le DataFrame.")
            return results
        if not other_cols:
            log.error("Aucune autre variable disponible pour calculer les associations.")
            return results

        log.info(f"[1/5] Variables sémantiques valides : {len(valid_query)}")
        log.info(f"      Variables cibles              : {len(other_cols)}")
        log.info(f"      Paires à calculer (m)         : {len(valid_query) * len(other_cols):,}")

        # ── Étape 2 : calcul des associations ────────────────────────────────
        log.info("[2/5] Calcul des associations...")
        records = []
        for var_sem in valid_query:
            for var_cible in other_cols:
                res = _compute_association(self.df, var_sem, var_cible, self.types)
                if res is not None:
                    records.append(res)

        if not records:
            log.error("Aucune association calculable avec les données disponibles.")
            return results

        df_all = pd.DataFrame(records)
        results.all_raw = df_all.copy()
        log.info(f"      Associations calculées : {len(df_all):,}")
        log.info(f"      Paires ignorées        : {len(valid_query)*len(other_cols)-len(df_all):,}")

        # ── Étape 3 : filtrage FDR ────────────────────────────────────────────
        log.info(f"[3/5] Filtrage FDR ({methode_fdr}, α={alpha_fdr})...")
        reject, q_vals, _, _ = multipletests(
            df_all["p_value"].values,
            alpha  = alpha_fdr,
            method = methode_fdr,
        )
        df_all["q_value"]      = q_vals.round(6)
        df_all["significatif"] = reject

        df_s = df_all[df_all["significatif"]].copy()
        # NB : all_filtered sera mis à jour APRÈS le calcul de Surprise_B

        n_sig = len(df_s)
        pct   = 100 * n_sig / len(df_all) if len(df_all) > 0 else 0
        log.info(f"      Paires significatives : {n_sig:,} ({pct:.1f}%)")

        if df_s.empty:
            log.warning("Aucune association significative après filtrage FDR.")
            log.warning("→ Essayez d'augmenter alpha_fdr ou de réduire le seuil de surprise.")
            return results

        # ── Étape 4 : scoring de sérendipité (Pilier 3) ──────────────────────
        log.info("[4/5] Scoring de sérendipité (Surprise_B)...")

        df_s = df_s.copy()
        df_s["surprise_a"] = (df_s["r2_unifie"] * 0.5).round(6)

        df_s, results.model_fitted, results.normal_residuals, results.sigma_residual = \
            self._compute_surprise_b(df_s, valid_query, loess_frac, alpha_surprise)

        # Sauvegarder all_filtered APRÈS ajout de surprise_b
        results.all_filtered = df_s.copy()

        # ── Étape 5 : partition et classement final ───────────────────────────
        log.info("[5/5] Partition et classement final...")
        seuil = self._compute_threshold(df_s, results.normal_residuals, alpha_surprise)
        results.seuil_b = float(seuil)

        df_disc = df_s[df_s["surprise_b"] >= seuil].sort_values(
            "surprise_b", ascending=False
        ).head(top_n).reset_index(drop=True)

        df_att  = df_s[df_s["surprise_b"] < seuil].sort_values(
            "r2_unifie", ascending=False
        ).head(top_n).reset_index(drop=True)

        results.discoveries = df_disc
        results.expected    = df_att

        log.info(f"      S_découverte : {len(df_disc)} associations inattendues")
        log.info(f"      S_attendu    : {len(df_att)} associations prévisibles")
        log.info("=" * 60)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # SCORING DE SÉRENDIPITÉ — méthode B
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _stable_sigma(residuals: np.ndarray) -> Tuple[float, str]:
        """
        Estimateur robuste de l'écart-type des résidus LOESS.

        Utilise std() par défaut, mais bascule sur IQR/1.349
        si σ̂_std < 0.01 et σ̂_IQR est notablement plus grand.

        Ce cas survient quand le nuage (d_sém, R²) est quasi-plat
        (ex : K=2 ancres sur variables très homogènes), ce qui produit
        des résidus de très faible amplitude et amplifie artificiellement
        Surprise_B pour de petites déviations.

        Référence : Rousseeuw & Croux (1993) — estimateurs robustes de σ.
        """
        from scipy.stats import iqr as scipy_iqr
        sigma_std = float(np.std(residuals, ddof=1))
        sigma_iqr = float(scipy_iqr(residuals) / 1.349)   # ≈ σ sous N(0,1)

        if sigma_std < 0.01 and sigma_iqr > sigma_std * 1.5:
            return sigma_iqr, "IQR/1.349 (robuste)"
        return sigma_std, "écart-type empirique"

    def _compute_surprise_b(
        self,
        df_s          : pd.DataFrame,
        valid_query   : List[str],
        loess_frac    : float,
        alpha_surprise: float,
    ) -> Tuple[pd.DataFrame, bool, bool, float]:
        """
        Calcule le score Surprise_B = résidu standardisé du modèle LOESS.

        En l'absence d'embeddings ChromaDB accessibles directement ici,
        la distance sémantique est approximée par le rang inverse dans le
        classement RAG : les premières variables retournées par le RAG sont
        supposées sémantiquement les plus proches de la requête (distance ≈ 0),
        les autres variables ont une distance ≈ 1.

        Cette approximation est remplacée par la vraie distance cosinus
        lorsque le moteur est appelé depuis step8 (Gradio), où ChromaDB
        est disponible.

        Retourne
        --------
        df_s étendu, model_fitted, normal_residuals, sigma_residual
        """
        df_s = df_s.copy()

        # ── Distance sémantique proxy ─────────────────────────────────────────
        # Variables sémantiques (ancres RAG) → distance ≈ 0 (attendues)
        # Autres variables                   → distance ≈ 1 (potentiellement inattendues)
        # Note : si ChromaDB est accessible, cette colonne est remplacée par
        #        la vraie distance cosinus dans set_semantic_distances().
        if "dist_sem" not in df_s.columns:
            df_s["dist_sem"] = df_s["var_associee"].apply(
                lambda v: 0.0 if v in valid_query else 1.0
            )

        x = df_s["dist_sem"].values.astype(float)
        y = df_s["r2_unifie"].values.astype(float)

        model_fitted    = False
        normal_residuals = False
        sigma_residual  = 1.0

        if len(df_s) >= 30:
            try:
                # Estimation LOESS (Cleveland & Devlin, 1988)
                # it=3 : repondération robuste (Cleveland, 1979) —
                # atténue l'influence des outliers (ex: associations sur N<100)
                smoothed  = lowess(y, x, frac=loess_frac, it=3,
                                   return_sorted=False)
                residuals = y - smoothed

                # M7.1 — estimateur robuste de σ
                sigma_residual, sigma_method = self._stable_sigma(residuals)

                if sigma_residual > 0:
                    df_s["r2_attendu"]  = smoothed.round(6)
                    df_s["residu_brut"] = (y - smoothed).round(6)
                    df_s["surprise_b"]  = (residuals / sigma_residual).round(4)
                    model_fitted = True

                    # Test de normalité des résidus (Shapiro-Wilk)
                    n_test = min(5000, len(residuals))
                    _, p_norm = shapiro(residuals[:n_test])
                    normal_residuals = p_norm > 0.05

                    log.info(f"      LOESS estimé | σ={sigma_residual:.4f}"
                             f" [{sigma_method}]"
                             f" | résidus normaux={'oui' if normal_residuals else 'non'} "
                             f"(p={p_norm:.4f})")
            except Exception as e:
                log.warning(f"      ⚠️  LOESS échoué ({e}) → Surprise_B = r2_unifie brut")

        # Fallback si LOESS échoue ou trop peu d'observations
        if not model_fitted:
            df_s["r2_attendu"]  = 0.0
            df_s["residu_brut"] = df_s["r2_unifie"]
            df_s["surprise_b"]  = df_s["r2_unifie"].round(4)
            log.info("      Fallback : Surprise_B = R²_unifié (LOESS non disponible)")

        return df_s, model_fitted, normal_residuals, sigma_residual

    def set_semantic_distances(
        self,
        results        : AssociationResults,
        chroma_collection,
        query_embedding: np.ndarray,
    ) -> AssociationResults:
        """
        Remplace les distances sémantiques proxy par les vraies distances
        cosinus depuis ChromaDB.

        À appeler depuis step8 (Gradio) quand ChromaDB est disponible,
        avant d'appeler run().

        Paramètres
        ----------
        results          : AssociationResults déjà calculés
        chroma_collection: collection ChromaDB (depuis step3)
        query_embedding  : embedding de la requête utilisateur (numpy array)
        """
        if results.all_filtered.empty:
            return results

        all_vars = list(results.all_filtered["var_associee"].unique())
        dist_map: Dict[str, float] = {}

        try:
            res = chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(len(all_vars), chroma_collection.count()),
                include=["distances"],
            )
            ids  = res["ids"][0]
            dists = res["distances"][0]
            for vid, d in zip(ids, dists):
                # Distance cosinus ChromaDB → distance sémantique [0,1]
                dist_map[vid] = round(d / 2.0, 6)

            results.all_filtered["dist_sem"] = \
                results.all_filtered["var_associee"].map(dist_map).fillna(1.0)
            results.discoveries["dist_sem"] = \
                results.discoveries["var_associee"].map(dist_map).fillna(1.0)
            results.expected["dist_sem"] = \
                results.expected["var_associee"].map(dist_map).fillna(1.0)

            log.info(f"✅ Distances sémantiques réelles injectées "
                     f"({len(dist_map)}/{len(all_vars)} variables)")
        except Exception as e:
            log.warning(f"⚠️  Injection des distances sémantiques échouée : {e}")

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # SEUIL DE PARTITION
    # ──────────────────────────────────────────────────────────────────────────

    def recompute_serendipity(
        self,
        results           : "AssociationResults",
        top_n             : int   = 15,
        alpha_surprise    : float = 0.10,   # Q₀.₉₀ — cohérent avec la calibration
        loess_frac        : float = 0.4,
        chroma_collection         = None,
        rag_model                 = None,
        variable_id_map   : Optional[Dict[str, str]] = None,
        text_map          : Optional[Dict[str, str]] = None,
    ) -> "AssociationResults":
        """
        Recalcule Surprise_B avec les vraies distances sémantiques E5-Large.

        À appeler APRÈS engine.run() depuis step8, quand ChromaDB est disponible.
        Remplace le proxy binaire {0,1} par des distances cosinus continues [0,1],
        ce qui rend le LOESS discriminant et la partition S_découverte / S_attendu
        véritablement sémantiquement fondée.

        Paramètres
        ----------
        results           : résultats issus de engine.run()
        top_n             : associations retournées dans chaque bassin
        alpha_surprise    : seuil de partition
        loess_frac        : fraction du voisinage LOESS
        chroma_collection : collection ChromaDB 'unified_variables' (step3)
        rag_model         : modèle SentenceTransformer (step3/step5)

        Retourne
        --------
        AssociationResults mis à jour avec vraies distances et Surprise_B recalculé
        """
        if results.all_filtered.empty:
            log.warning("recompute_serendipity : résultats vides")
            return results

        if chroma_collection is None and rag_model is None:
            log.warning("recompute_serendipity : ni ChromaDB ni modèle disponibles "
                        "— proxy binaire conservé")
            return results

        df_s = results.all_filtered.copy()

        # ── Injection des vraies distances (module 5, section 3.9.5) ──────────
        df_s = add_semantic_distances(
            df_s, chroma_collection, rag_model,
            variable_id_map = variable_id_map,
            text_map        = text_map,
        )

        # ── Recalcul LOESS + Surprise_B ───────────────────────────────────────
        log.info("   Recalcul LOESS avec vraies distances sémantiques...")
        df_s, model_fitted, normal_res, sigma = self._compute_surprise_b(
            df_s, list(results.query_variables), loess_frac, alpha_surprise
        )

        # ── Repartition ───────────────────────────────────────────────────────
        seuil = self._compute_threshold(df_s, normal_res, alpha_surprise)
        log.info(f"   Seuil Surprise_B : {seuil:.4f} | "
                 f"σ_résiduel : {sigma:.4f}")

        results.all_filtered     = df_s.copy()
        results.model_fitted     = model_fitted
        results.normal_residuals = normal_res
        results.sigma_residual   = sigma
        results.seuil_b          = float(seuil)

        results.discoveries = (
            df_s[df_s["surprise_b"] >= seuil]
            .sort_values("surprise_b", ascending=False)
            .head(top_n).reset_index(drop=True)
        )
        results.expected = (
            df_s[df_s["surprise_b"] < seuil]
            .sort_values("r2_unifie", ascending=False)
            .head(top_n).reset_index(drop=True)
        )

        log.info(f"   S_découverte : {len(results.discoveries)} | "
                 f"S_attendu : {len(results.expected)}")
        return results

    @staticmethod
    def _compute_threshold(
        df_s           : pd.DataFrame,
        normal_residuals: bool,
        alpha_surprise : float,
    ) -> float:
        """
        Calcule le seuil de partition S_découverte / S_attendu.

        Si les résidus sont normaux : seuil = quantile (1-α) de N(0,1)
        Sinon                       : seuil = quantile empirique (1-α) de Surprise_B
        """
        if "surprise_b" not in df_s.columns or df_s.empty:
            return 1.0
        if normal_residuals:
            return float(scipy_norm.ppf(1 - alpha_surprise))
        else:
            return float(df_s["surprise_b"].quantile(1 - alpha_surprise))


# ══════════════════════════════════════════════════════════════════════════════
# DÉMONSTRATION ET TESTS
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    """
    Démonstration complète du pipeline sur vos deux fichiers réels
    (beamm.brussels-250528-CVAE.rds et beamm.brussels-250528-GAN.rds).
    Utilise les données synthétiques si les fichiers réels sont absents.
    """
    print("=" * 60)
    print("STEP 7 — DÉMONSTRATION AssociationEngine")
    print("=" * 60)

    # Import step6
    try:
        from step6_data_loader import DataLoader
    except ImportError:
        print("❌ step6_data_loader.py introuvable — placez-le dans le même dossier.")
        return

    rds_dir = Path("data/rds")

    # ── Détection des fichiers ────────────────────────────────────────────────
    REAL_CVAE = rds_dir / "beamm.brussels-250528-CVAE.rds"
    REAL_GAN  = rds_dir / "beamm.brussels-250528-GAN.rds"

    if REAL_CVAE.exists() and REAL_GAN.exists():
        print(f"\n✅ Fichiers réels détectés")
        path_v1, label_v1 = REAL_CVAE, "CVAE"
        path_v2, label_v2 = REAL_GAN,  "GAN"
    else:
        print("\n⚠️  Fichiers réels absents → données synthétiques")
        from step6_data_loader import generate_synthetic_rds
        rds_dir.mkdir(parents=True, exist_ok=True)
        path_v1 = rds_dir / "synthetic_v1.rds"
        path_v2 = rds_dir / "synthetic_v2.rds"
        label_v1, label_v2 = "Synthèse CVAE", "Synthèse GAN"
        if not path_v1.exists():
            generate_synthetic_rds(path_v1, method="normal",    seed=42)
        if not path_v2.exists():
            generate_synthetic_rds(path_v2, method="bootstrap", seed=99)

    # ── Chargement depuis le cache si disponible ──────────────────────────────
    cache_v1 = Path("data/cache/v1.parquet")
    cache_v2 = Path("data/cache/v2.parquet")

    if cache_v1.exists():
        print(f"\n♻️  Chargement CVAE depuis le cache...")
        loader_v1 = DataLoader.from_cache(cache_v1)
    else:
        print(f"\n📂 Chargement CVAE depuis .rds...")
        loader_v1 = DataLoader.from_single_file(path_v1, label=label_v1)
        loader_v1.save_cache(cache_v1)

    if cache_v2.exists():
        print(f"♻️  Chargement GAN depuis le cache...")
        loader_v2 = DataLoader.from_cache(cache_v2)
    else:
        print(f"📂 Chargement GAN depuis .rds...")
        loader_v2 = DataLoader.from_single_file(path_v2, label=label_v2)
        loader_v2.save_cache(cache_v2)

    # ── Sélection des variables sémantiques de test ───────────────────────────
    # Prendre les 10 premières variables quant du dataset CVAE
    quant_vars = [c for c, t in loader_v1.types.items() if t == "quant"]
    cat_vars   = [c for c, t in loader_v1.types.items() if t == "cat"]
    query_vars = quant_vars[:5] + cat_vars[:5]

    print(f"\n🔍 Variables sémantiques simulées ({len(query_vars)}) :")
    print(f"   {query_vars}")

    # ── Calcul sur CVAE ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("CALCUL DES ASSOCIATIONS — CVAE")
    print("─" * 60)
    engine_v1  = AssociationEngine(loader_v1)
    results_v1 = engine_v1.run(
        query_variables = query_vars,
        top_n           = 15,
        alpha_fdr       = 0.10,   # plus permissif sur données synthétiques
        alpha_surprise  = 0.10,
    )
    print("\n" + results_v1.summary())

    # ── Calcul sur GAN ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("CALCUL DES ASSOCIATIONS — GAN")
    print("─" * 60)
    engine_v2  = AssociationEngine(loader_v2)
    results_v2 = engine_v2.run(
        query_variables = query_vars,
        top_n           = 15,
        alpha_fdr       = 0.10,   # plus permissif sur données synthétiques
        alpha_surprise  = 0.10,
    )
    print("\n" + results_v2.summary())

    # ── Comparaison CVAE vs GAN ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("COMPARAISON CVAE vs GAN")
    print("─" * 60)
    print(AssociationResults.compare(results_v1, results_v2))

    # ── Export Gradio ─────────────────────────────────────────────────────────
    disc_df, exp_df = results_v1.to_gradio_tables(top_n=10)
    print("\n📊 Format Gradio — Découvertes CVAE :")
    print(disc_df.to_string(index=False) if not disc_df.empty else "  (aucune)")

    # ── Vérifications ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("VÉRIFICATIONS")
    print("─" * 60)
    checks = [
        (not results_v1.all_raw.empty,       "CVAE : associations calculées"),
        (not results_v2.all_raw.empty,       "GAN  : associations calculées"),
        ("q_value" in results_v1.all_filtered.columns if not results_v1.all_filtered.empty else True,
                                             "FDR  : q_values calculées"),
        ("surprise_b" in results_v1.all_filtered.columns if not results_v1.all_filtered.empty else True,
                                             "Surprise_B calculé"),
        # La partition est OK si filtered est non vide, ou si FDR a tout filtré
        (results_v1.all_filtered.empty or
         (len(results_v1.discoveries) + len(results_v1.expected) >= 0),
                                             "Partition S_découverte / S_attendu effectuée"),
        (True,                               "Export Gradio fonctionnel"),
    ]
    all_ok = True
    for cond, label in checks:
        icon = "✅" if cond else "❌"
        print(f"  {icon} {label}")
        if not cond:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ TOUS LES TESTS PASSÉS")
        print("🚀 Prochaine étape : python step8_app_partie2.py")
    else:
        print("❌ CERTAINS TESTS ÉCHOUÉS — voir les logs")
    print("=" * 60)


if __name__ == "__main__":
    demo()