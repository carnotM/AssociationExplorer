#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 6 - CHARGEMENT DES DONNÉES .rds POUR ASSOCIATIONEXPLORER
==============================================================

Ce module charge les microdonnées synthétiques au format .rds (R) et les
prépare pour le calcul des associations statistiques (step7).

ARCHITECTURE — DEUX MODES :
─────────────────────────────────────────────────────────────────────────────

MODE PRINCIPAL (celui que vous utilisez maintenant) :
    Un seul fichier .rds contenant toutes les variables de toutes les enquêtes.
    Vous disposez de DEUX fichiers de ce type (même variables, méthodes de
    synthèse différentes). Le mode principal permet de charger l'un, obtenir
    les résultats d'association, puis charger l'autre et comparer.

    Usage :
        loader = DataLoader.from_single_file('data/rds/synthetic_v1.rds')
        loader = DataLoader.from_single_file('data/rds/synthetic_v2.rds')

    Le fichier doit contenir une colonne 'survey' (ou équivalent paramétrable)
    indiquant à quelle enquête appartient chaque ligne. S'il n'y a pas de
    colonne survey, le fichier est traité comme une enquête unique.

MODE FUTUR (pour des données réelles organisées par enquête) :
    Un fichier .rds par enquête dans un dossier.
    Prévu pour quand les vraies microdonnées seront disponibles.

    Usage :
        loader = DataLoader.from_survey_folder('data/rds/real/')

─────────────────────────────────────────────────────────────────────────────

SORTIE (identique pour les deux modes) :
    loader.df           → pd.DataFrame [observations × variables]
    loader.variable_map → dict {colonne → enquête d'origine}
    loader.types        → dict {colonne → 'quant' | 'cat'}
    loader.surveys      → list des enquêtes présentes
    loader.summary()    → rapport lisible
    loader.label        → nom/chemin du fichier chargé (pour comparer)

INTÉGRATION AVEC LE PIPELINE :
    Importé par step7_association.py :
        from step6_data_loader import DataLoader

DÉPENDANCES :
    pip install pyreadr pyarrow
    (pyarrow : uniquement pour le cache Parquet — optionnel)

Auteur  : AssociationExplorer — Partie II
Date    : 2025-2026
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyreadr
except ImportError:
    raise ImportError(
        "pyreadr est requis pour lire les fichiers .rds.\n"
        "Installation : pip install pyreadr"
    )

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("step6_loader.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

#: Correspondance nom de fichier → nom normalisé de l'enquête (Mode Futur)
SURVEY_FILENAME_MAP: Dict[str, str] = {
    "eu_silc": "EU-SILC", "eusilc": "EU-SILC", "silc": "EU-SILC",
    "hfcs"   : "HFCS",
    "eu_lfs" : "EU-LFS",  "eulfs" : "EU-LFS",  "lfs" : "EU-LFS",
    "hbs"    : "HBS",
    "ipcal"  : "IPCAL",
    "demobel": "DEMOBEL",
}

#: Seuil de modalités uniques pour 'cat' (ajustable)
DEFAULT_CAT_THRESHOLD: int = 20

#: Minimum d'observations non-nulles pour qu'une colonne soit conservée
DEFAULT_MIN_OBS: int = 10

#: Colonnes à exclure automatiquement (identifiants, pondérations, clés)
EXCLUDED_COL_PATTERNS: List[str] = [
    r"^id$", r"^ident", r"^_id", r"^key", r"^pk",
    r"weight", r"poids", r"gewicht",
    r"^hid$", r"^pid$", r"^rb030$",
]


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES INTERNES
# ══════════════════════════════════════════════════════════════════════════════

def _read_rds(filepath: Path) -> pd.DataFrame:
    """
    Lit un fichier .rds avec une cascade de méthodes de lecture.

    Méthode 1 — pyreadr (rapide, sans R installé)
        Fonctionne pour la plupart des fichiers .rds simples.
        Échoue si le fichier contient des objets R complexes
        (tibble, data.table, facteurs spéciaux, compression xz/bzip2).

    Méthode 2 — rpy2 (nécessite R installé, très robuste)
        Lit n'importe quel fichier .rds valide en déléguant à R.

    Méthode 3 — subprocess R (nécessite R installé, sans rpy2)
        Convertit le .rds en CSV temporaire via un appel R en ligne de commande,
        puis charge le CSV. Fonctionne même sans rpy2.

    Si les trois méthodes échouent, un message clair indique comment
    convertir le fichier directement depuis R.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    log.info(f"   Lecture de {filepath.name} ...")

    # ── Méthode 1 : pyreadr ───────────────────────────────────────────────────
    try:
        result = pyreadr.read_r(str(filepath))
        df = result[None] if None in result else result[list(result.keys())[0]]
        if isinstance(df, pd.DataFrame) and not df.empty:
            log.info(f"   ✅ [pyreadr] {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
            return df
    except Exception as e1:
        log.warning(f"   ⚠️  pyreadr échoué ({type(e1).__name__}: {e1}) → essai rpy2...")

    # ── Méthode 2 : rpy2 ─────────────────────────────────────────────────────
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        base = importr("base")
        r_df = base.readRDS(str(filepath))
        df = pandas2ri.rpy2py(r_df)
        if isinstance(df, pd.DataFrame) and not df.empty:
            log.info(f"   ✅ [rpy2] {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
            return df
    except ImportError:
        log.warning("   ⚠️  rpy2 non installé → essai subprocess R...")
    except Exception as e2:
        log.warning(f"   ⚠️  rpy2 échoué ({type(e2).__name__}: {e2}) → essai subprocess R...")

    # ── Méthode 3 : subprocess R (R doit être dans le PATH) ──────────────────
    try:
        import subprocess
        import tempfile

        tmp_csv = Path(tempfile.mktemp(suffix=".csv"))
        r_script = (
            f"df <- readRDS('{str(filepath).replace(chr(92), '/')}'); "
            f"write.csv(df, '{str(tmp_csv).replace(chr(92), '/')}', row.names=FALSE)"
        )

        log.info("   🔧 Tentative via subprocess R (conversion en CSV temporaire)...")
        proc = subprocess.run(
            ["Rscript", "--vanilla", "-e", r_script],
            capture_output=True, text=True, timeout=120
        )

        if proc.returncode == 0 and tmp_csv.exists():
            df = pd.read_csv(tmp_csv)
            tmp_csv.unlink(missing_ok=True)
            if not df.empty:
                log.info(f"   ✅ [Rscript] {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
                return df
        else:
            log.warning(f"   ⚠️  Rscript échoué : {proc.stderr[:200]}")
    except FileNotFoundError:
        log.warning("   ⚠️  Rscript introuvable dans le PATH")
    except Exception as e3:
        log.warning(f"   ⚠️  subprocess R échoué ({type(e3).__name__}: {e3})")

    # ── Toutes les méthodes ont échoué ────────────────────────────────────────
    msg = f"""
❌ Impossible de lire : {filepath.name}

Les trois méthodes de lecture ont échoué.
Le fichier contient probablement des objets R non supportés (tibble,
data.table, facteurs complexes) ou une compression non standard.

SOLUTION RAPIDE — Convertissez le fichier depuis R :
─────────────────────────────────────────────────────
    # Dans la console R :
    df <- readRDS("{filepath.name}")
    df <- as.data.frame(df)               # convertir tibble → data.frame
    write.csv(df, "{filepath.stem}.csv",  # sauvegarder en CSV
              row.names = FALSE)

Puis remplacez l'appel from_single_file() par :
    loader = DataLoader.from_csv("data/rds/{filepath.stem}.csv")

SOLUTION COMPLÈTE (conserve le format .rds) :
─────────────────────────────────────────────
    df <- readRDS("{filepath.name}")
    df <- as.data.frame(df)
    saveRDS(df, "{filepath.name}")        # ré-enregistrer en format simple

Dépendances manquantes :
    pip install rpy2                      # pour la méthode 2 (+ R installé)
    R doit être dans le PATH             # pour la méthode 3
"""
    raise RuntimeError(msg)


def _detect_type(series: pd.Series, threshold: int) -> str:
    """Détecte si une série est 'quant' ou 'cat'."""
    if series.dtype == "object" or str(series.dtype) == "category":
        return "cat"
    if series.nunique() <= threshold:
        return "cat"
    return "quant"


def _is_excluded_col(col: str) -> bool:
    """Retourne True si la colonne est technique et doit être exclue."""
    import re
    col_l = col.lower()
    for pattern in EXCLUDED_COL_PATTERNS:
        if re.search(pattern, col_l):
            return True
    return False


def _build_types_and_map(
    df: pd.DataFrame,
    survey_name: str,
    cat_threshold: int,
    min_obs: int,
) -> tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Nettoie un DataFrame et construit les mappings types + variable_map.

    Retourne (df_nettoyé, variable_map, types).
    """
    df = df.copy()

    # Exclure colonnes techniques
    excl = [c for c in df.columns if _is_excluded_col(c)]
    if excl:
        log.info(f"   [{survey_name}] {len(excl)} col. techniques exclues")
        df = df.drop(columns=excl)

    # Exclure colonnes trop vides
    sparse = df.columns[df.notna().sum() < min_obs].tolist()
    if sparse:
        log.info(f"   [{survey_name}] {len(sparse)} col. insuffisantes exclues")
        df = df.drop(columns=sparse)

    var_map = {c: survey_name for c in df.columns}
    types   = {c: _detect_type(df[c].dropna(), cat_threshold) for c in df.columns}

    n_q = sum(1 for v in types.values() if v == "quant")
    n_c = sum(1 for v in types.values() if v == "cat")
    log.info(f"   [{survey_name}] {len(df.columns)} variables retenues "
             f"({n_q} quant, {n_c} cat)")
    return df, var_map, types


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Chargeur de microdonnées .rds pour AssociationExplorer.

    Ne pas instancier directement — utiliser les méthodes de classe :

        # Mode PRINCIPAL (votre cas) :
        loader = DataLoader.from_single_file('data/rds/synthetic_v1.rds')

        # Charger le second fichier pour comparaison :
        loader2 = DataLoader.from_single_file('data/rds/synthetic_v2.rds')

        # Mode FUTUR (un fichier par enquête) :
        loader = DataLoader.from_survey_folder('data/rds/real/')

    Attributs
    ---------
    df          : pd.DataFrame [observations × variables]
    variable_map: dict {colonne → enquête d'origine}
    types       : dict {colonne → 'quant' | 'cat'}
    surveys     : list[str] des enquêtes présentes
    label       : str — identifiant lisible de ce dataset (pour comparaison)
    """

    def __init__(self):
        self.df          : pd.DataFrame   = pd.DataFrame()
        self.variable_map: Dict[str, str] = {}
        self.types       : Dict[str, str] = {}
        self.surveys     : List[str]      = []
        self.label       : str            = ""
        self._mode       : str            = ""
        self._filepath   : Optional[Path] = None
        self._cat_threshold: int          = DEFAULT_CAT_THRESHOLD
        self._min_obs    : int            = DEFAULT_MIN_OBS

    # ──────────────────────────────────────────────────────────────────────────
    # MODE PRINCIPAL — Un seul fichier .rds
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_single_file(
        cls,
        filepath      : str | Path,
        survey_col    : str = "survey",
        cat_threshold : int = DEFAULT_CAT_THRESHOLD,
        min_obs       : int = DEFAULT_MIN_OBS,
        label         : str = "",
    ) -> "DataLoader":
        """
        MODE PRINCIPAL — Charge un seul fichier .rds contenant toutes les
        variables de toutes les enquêtes (ou d'une seule enquête unifiée).

        Pour comparer deux synthèses, appelez cette méthode deux fois :
            loader_v1 = DataLoader.from_single_file('synthetic_v1.rds', label='Synthèse 1')
            loader_v2 = DataLoader.from_single_file('synthetic_v2.rds', label='Synthèse 2')

        Paramètres
        ----------
        filepath      : chemin du fichier .rds
        survey_col    : nom de la colonne identifiant l'enquête (défaut : 'survey').
                        Si cette colonne est absente, le fichier entier est traité
                        comme une enquête unique.
        cat_threshold : seuil de modalités pour 'cat' (défaut : 20)
        min_obs       : observations non-nulles minimales par colonne (défaut : 10)
        label         : nom affiché pour ce dataset dans les comparaisons.
                        Si vide, le nom du fichier est utilisé.

        Retourne
        --------
        DataLoader prêt à l'emploi
        """
        loader = cls()
        loader._mode         = "single"
        loader._filepath     = Path(filepath)
        loader._cat_threshold = cat_threshold
        loader._min_obs      = min_obs
        loader.label         = label or Path(filepath).stem

        log.info("=" * 65)
        log.info("MODE PRINCIPAL — Fichier .rds unique")
        log.info(f"Fichier : {loader._filepath.absolute()}")
        log.info(f"Label   : {loader.label}")
        log.info("=" * 65)

        # Lecture du fichier
        df_raw = _read_rds(loader._filepath)

        # ── Cas 1 : colonne survey présente ──────────────────────────────────
        if survey_col in df_raw.columns:
            df_raw[survey_col] = (
                df_raw[survey_col].astype(str).str.strip().str.upper()
            )
            surveys_found = sorted(df_raw[survey_col].unique().tolist())
            loader.surveys = surveys_found
            log.info(f"Colonne '{survey_col}' trouvée → "
                     f"{len(surveys_found)} enquête(s) : {surveys_found}")

            all_parts: List[pd.DataFrame] = []
            for survey_name in surveys_found:
                df_part = df_raw[df_raw[survey_col] == survey_name].drop(
                    columns=[survey_col]
                ).reset_index(drop=True)

                df_clean, vmap, types = _build_types_and_map(
                    df_part, survey_name, cat_threshold, min_obs
                )
                all_parts.append(df_clean)
                loader.variable_map.update(vmap)
                loader.types.update(types)

            # Empiler les blocs verticalement (les colonnes non communes → NaN)
            loader.df = pd.concat(all_parts, axis=0, ignore_index=True)

        # ── Cas 2 : pas de colonne survey → fichier unique ────────────────────
        else:
            log.info(f"Colonne '{survey_col}' absente → fichier traité comme "
                     f"enquête unique ('{loader.label}')")
            survey_name = loader.label.upper()
            loader.surveys = [survey_name]

            df_clean, vmap, types = _build_types_and_map(
                df_raw, survey_name, cat_threshold, min_obs
            )
            loader.df        = df_clean
            loader.variable_map = vmap
            loader.types     = types

        loader._log_state()
        return loader

    # ──────────────────────────────────────────────────────────────────────────
    # MODE FUTUR — Un fichier .rds par enquête dans un dossier
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_survey_folder(
        cls,
        folder       : str | Path,
        surveys      : Optional[List[str]] = None,
        cat_threshold: int = DEFAULT_CAT_THRESHOLD,
        min_obs      : int = DEFAULT_MIN_OBS,
        label        : str = "",
    ) -> "DataLoader":
        """
        MODE FUTUR — Charge un fichier .rds par enquête depuis un dossier.

        Chaque fichier doit être nommé d'après l'enquête qu'il représente
        (ex : eu_silc.rds, hfcs.rds, eu_lfs.rds, ...).

        Paramètres
        ----------
        folder        : dossier contenant les fichiers .rds
        surveys       : enquêtes à charger (None = toutes détectées)
        cat_threshold : seuil de modalités pour 'cat' (défaut : 20)
        min_obs       : observations non-nulles minimales par colonne (défaut : 10)
        label         : identifiant affiché pour ce dataset

        Retourne
        --------
        DataLoader prêt à l'emploi
        """
        loader = cls()
        loader._mode         = "folder"
        loader._cat_threshold = cat_threshold
        loader._min_obs      = min_obs
        folder = Path(folder)
        loader.label = label or folder.name

        log.info("=" * 65)
        log.info("MODE FUTUR — Dossier de fichiers .rds par enquête")
        log.info(f"Dossier : {folder.absolute()}")
        log.info("=" * 65)

        if not folder.exists():
            raise FileNotFoundError(f"Dossier introuvable : {folder}")

        rds_files = sorted(folder.glob("*.rds"))
        if not rds_files:
            raise FileNotFoundError(f"Aucun fichier .rds trouvé dans : {folder}")

        log.info(f"{len(rds_files)} fichier(s) .rds trouvé(s)")

        all_dfs: List[pd.DataFrame] = []

        for fp in rds_files:
            # Déterminer le nom de l'enquête depuis le nom du fichier
            stem = fp.stem.lower()
            survey_name = next(
                (v for k, v in sorted(SURVEY_FILENAME_MAP.items(), key=lambda x: -len(x[0]))
                 if k in stem),
                None
            )
            if survey_name is None:
                log.warning(f"   ⚠️  '{fp.name}' — enquête non reconnue, ignoré")
                continue
            if surveys and survey_name not in surveys:
                log.info(f"   ⏭️  {fp.name} ({survey_name}) — filtré")
                continue

            log.info(f"\n📂 {fp.name} → {survey_name}")
            try:
                df_raw = _read_rds(fp)
                df_clean, vmap, types = _build_types_and_map(
                    df_raw, survey_name, cat_threshold, min_obs
                )
                # Préfixer les colonnes pour éviter les collisions
                short = {"EU-SILC":"SILC","HFCS":"HFCS","EU-LFS":"LFS",
                         "HBS":"HBS","IPCAL":"IPCAL","DEMOBEL":"DEMO"
                         }.get(survey_name, survey_name[:4])
                df_clean = df_clean.rename(columns={c: f"{short}_{c}" for c in df_clean.columns})
                vmap  = {f"{short}_{c}": s for c, s in vmap.items()}
                types = {f"{short}_{c}": t for c, t in types.items()}

                all_dfs.append(df_clean)
                loader.variable_map.update(vmap)
                loader.types.update(types)
                if survey_name not in loader.surveys:
                    loader.surveys.append(survey_name)
            except Exception as e:
                log.error(f"   ❌ Erreur sur {fp.name} : {e}")

        if not all_dfs:
            raise ValueError("Aucun DataFrame valide chargé depuis le dossier.")

        # Fusionner horizontalement (même nombre de lignes supposé)
        min_rows = min(df.shape[0] for df in all_dfs)
        log.info(f"\n🔗 Fusion de {len(all_dfs)} fichiers → {min_rows:,} lignes communes")
        loader.df = pd.concat(
            [df.iloc[:min_rows].reset_index(drop=True) for df in all_dfs],
            axis=1
        )

        loader._log_state()
        return loader

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTHODES DE SECOURS — si le .rds ne peut pas être lu directement
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        filepath      : str | Path,
        survey_col    : str = "survey",
        cat_threshold : int = DEFAULT_CAT_THRESHOLD,
        min_obs       : int = DEFAULT_MIN_OBS,
        label         : str = "",
        sep           : str = ",",
    ) -> "DataLoader":
        """
        Charge un fichier CSV exporté depuis R comme alternative au .rds.

        Utiliser quand pyreadr ne peut pas lire le fichier .rds.
        Générer le CSV depuis R avec :
            df <- readRDS("mon_fichier.rds")
            df <- as.data.frame(df)
            write.csv(df, "mon_fichier.csv", row.names = FALSE)

        Paramètres
        ----------
        filepath      : chemin du fichier CSV
        survey_col    : colonne identifiant l'enquête (défaut : 'survey')
        cat_threshold : seuil de modalités pour 'cat'
        min_obs       : observations minimales par colonne
        label         : identifiant affiché pour ce dataset
        sep           : séparateur CSV (défaut : ',')
        """
        loader = cls()
        loader._mode         = "csv"
        loader._filepath     = Path(filepath)
        loader._cat_threshold = cat_threshold
        loader._min_obs      = min_obs
        loader.label         = label or Path(filepath).stem

        log.info("=" * 65)
        log.info("MODE CSV (fallback) — Fichier CSV")
        log.info(f"Fichier : {loader._filepath.absolute()}")
        log.info("=" * 65)

        if not loader._filepath.exists():
            raise FileNotFoundError(f"Fichier introuvable : {loader._filepath}")

        df_raw = pd.read_csv(str(loader._filepath), sep=sep, low_memory=False)
        log.info(f"   ✅ {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")

        # Réutiliser la logique de from_single_file
        if survey_col in df_raw.columns:
            df_raw[survey_col] = df_raw[survey_col].astype(str).str.strip().str.upper()
            surveys_found = sorted(df_raw[survey_col].unique().tolist())
            loader.surveys = surveys_found
            log.info(f"Colonne '{survey_col}' → {len(surveys_found)} enquête(s)")
            all_parts = []
            for sv in surveys_found:
                df_part = df_raw[df_raw[survey_col] == sv].drop(columns=[survey_col]).reset_index(drop=True)
                df_clean, vmap, types = _build_types_and_map(df_part, sv, cat_threshold, min_obs)
                all_parts.append(df_clean)
                loader.variable_map.update(vmap)
                loader.types.update(types)
            loader.df = pd.concat(all_parts, axis=0, ignore_index=True)
        else:
            survey_name = loader.label.upper()
            loader.surveys = [survey_name]
            df_clean, vmap, types = _build_types_and_map(df_raw, survey_name, cat_threshold, min_obs)
            loader.df = df_clean
            loader.variable_map = vmap
            loader.types = types

        loader._log_state()
        return loader

    @staticmethod
    def print_r_conversion_script(rds_path: str | Path, output_dir: str | Path = "data/rds"):
        """
        Affiche le script R à exécuter pour convertir un .rds non lisible en CSV.

        Usage :
            DataLoader.print_r_conversion_script("data/rds/bm.bls-126-CVAE.rds")
        """
        rds_path   = Path(rds_path)
        output_dir = Path(output_dir)
        csv_name   = rds_path.stem.replace(".", "_") + ".csv"
        rds_new    = rds_path.stem.replace(".", "_") + "_simple.rds"

        print("─" * 60)
        print("SCRIPT R POUR CONVERTIR LE FICHIER .rds")
        print("─" * 60)
        print("# Ouvrez RStudio ou la console R et exécutez :")
        print()
        print(f'df <- readRDS("{rds_path.name}")')
        print( "df <- as.data.frame(df)        # tibble → data.frame standard")
        print( "# Option A : exporter en CSV")
        print(f'write.csv(df, "{csv_name}", row.names = FALSE)')
        print( "# Option B : ré-enregistrer en .rds standard (plus léger)")
        print(f'saveRDS(df, "{rds_new}")')
        print()
        print("# Puis dans Python :")
        print(f'loader = DataLoader.from_csv("data/rds/{csv_name}")       # Option A')
        print(f'loader = DataLoader.from_single_file("data/rds/{rds_new}") # Option B')
        print("─" * 60)

    # ──────────────────────────────────────────────────────────────────────────
    # CACHE PARQUET (accélère les relances successives)
    # ──────────────────────────────────────────────────────────────────────────

    def save_cache(self, path: str | Path) -> Path:
        """
        Sauvegarde le DataFrame et ses métadonnées en cache Parquet.
        Les relances suivantes utilisent from_cache() — 10-100x plus rapide.

        Paramètres
        ----------
        path : chemin du fichier .parquet (ex: 'data/cache/v1.parquet')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path, index=False)

        meta = path.with_suffix(".meta.json")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({
                "mode"        : self._mode,
                "label"       : self.label,
                "surveys"     : self.surveys,
                "variable_map": self.variable_map,
                "types"       : self.types,
                "cat_threshold": self._cat_threshold,
                "min_obs"     : self._min_obs,
            }, f, ensure_ascii=False, indent=2)

        log.info(f"💾 Cache sauvegardé : {path} | méta : {meta}")
        return path

    @classmethod
    def from_cache(cls, path: str | Path) -> "DataLoader":
        """
        Charge un DataLoader depuis un cache Parquet.

        Exemple
        -------
        # Première fois (lent)
        loader = DataLoader.from_single_file('synthetic_v1.rds')
        loader.save_cache('data/cache/v1.parquet')

        # Fois suivantes (rapide)
        loader = DataLoader.from_cache('data/cache/v1.parquet')
        """
        path = Path(path)
        meta = path.with_suffix(".meta.json")
        loader = cls()
        loader.df = pd.read_parquet(path)

        if meta.exists():
            with open(meta, "r", encoding="utf-8") as f:
                m = json.load(f)
            loader._mode          = m.get("mode", "?")
            loader.label          = m.get("label", path.stem)
            loader.surveys        = m.get("surveys", [])
            loader.variable_map   = m.get("variable_map", {})
            loader.types          = m.get("types", {})
            loader._cat_threshold = m.get("cat_threshold", DEFAULT_CAT_THRESHOLD)
            loader._min_obs       = m.get("min_obs", DEFAULT_MIN_OBS)
        else:
            loader.label  = path.stem
            loader.types  = {c: _detect_type(loader.df[c].dropna(), DEFAULT_CAT_THRESHOLD)
                             for c in loader.df.columns}
            loader.variable_map = {c: "UNKNOWN" for c in loader.df.columns}

        log.info(f"♻️  Cache chargé : {path}")
        loader._log_state()
        return loader

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTHODES D'ACCÈS AUX DONNÉES
    # ──────────────────────────────────────────────────────────────────────────

    def get_variables_for_survey(self, survey: str) -> List[str]:
        """Retourne les colonnes appartenant à une enquête donnée."""
        return [c for c, s in self.variable_map.items() if s == survey]

    def get_subset(self, columns: List[str]) -> pd.DataFrame:
        """Extrait un sous-DataFrame pour les colonnes demandées."""
        cols = [c for c in columns if c in self.df.columns]
        return self.df[cols].dropna(how="all").reset_index(drop=True) if cols else pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────────
    # RAPPORT / AFFICHAGE
    # ──────────────────────────────────────────────────────────────────────────

    def _log_state(self):
        n_q = sum(1 for v in self.types.values() if v == "quant")
        n_c = sum(1 for v in self.types.values() if v == "cat")
        log.info("\n" + "=" * 65)
        log.info(f"✅ [{self.label}] CHARGEMENT TERMINÉ")
        log.info(f"   Mode         : {self._mode}")
        log.info(f"   Observations : {self.df.shape[0]:,}")
        log.info(f"   Variables    : {self.df.shape[1]} ({n_q} quant, {n_c} cat)")
        log.info(f"   Enquêtes     : {self.surveys}")
        mem = self.df.memory_usage(deep=True).sum() / 1e6
        log.info(f"   Mémoire      : {mem:.1f} MB")
        log.info("=" * 65)

    def summary(self) -> str:
        """Rapport lisible — utilisé dans Gradio et les logs."""
        if self.df.empty:
            return f"[{self.label}] ❌ Aucune donnée chargée."
        n_q = sum(1 for v in self.types.values() if v == "quant")
        n_c = sum(1 for v in self.types.values() if v == "cat")
        mem = self.df.memory_usage(deep=True).sum() / 1e6
        mode_label = {
            "single": "Fichier unique (Mode Principal)",
            "folder": "Dossier par enquête (Mode Futur)",
        }.get(self._mode, self._mode)

        lines = [
            "=" * 55,
            f"DATASET : {self.label}",
            "=" * 55,
            f"Mode         : {mode_label}",
            f"Observations : {self.df.shape[0]:,}",
            f"Variables    : {self.df.shape[1]}  ({n_q} quant  |  {n_c} cat)",
            f"Enquêtes     : {', '.join(self.surveys)}",
            f"Mémoire      : {mem:.1f} MB",
            "",
            "Détail par enquête :",
        ]
        for sv in self.surveys:
            cols = self.get_variables_for_survey(sv)
            nq   = sum(1 for c in cols if self.types.get(c) == "quant")
            nc   = sum(1 for c in cols if self.types.get(c) == "cat")
            lines.append(f"  {sv:<12} : {len(cols):>3} variables  "
                         f"({nq} quant, {nc} cat)")
        lines += ["", "Exemples de variables :",
                  f"  quant : {[c for c,t in self.types.items() if t=='quant'][:4]}",
                  f"  cat   : {[c for c,t in self.types.items() if t=='cat' ][:4]}",
                  "=" * 55]
        return "\n".join(lines)

    @staticmethod
    def compare(loader1: "DataLoader", loader2: "DataLoader") -> str:
        """
        Compare deux DataLoaders côte à côte.
        Utile pour comparer les deux fichiers de synthèse différente.

        Paramètres
        ----------
        loader1, loader2 : les deux DataLoaders à comparer

        Retourne
        --------
        str — rapport de comparaison
        """
        def info(l):
            n_q = sum(1 for v in l.types.values() if v == "quant")
            n_c = sum(1 for v in l.types.values() if v == "cat")
            return {
                "obs"     : l.df.shape[0],
                "vars"    : l.df.shape[1],
                "n_quant" : n_q,
                "n_cat"   : n_c,
                "surveys" : set(l.surveys),
                "cols"    : set(l.df.columns),
            }

        i1, i2 = info(loader1), info(loader2)

        cols_only_1 = i1["cols"] - i2["cols"]
        cols_only_2 = i2["cols"] - i1["cols"]
        cols_common = i1["cols"] & i2["cols"]

        lines = [
            "=" * 65,
            "COMPARAISON DES DEUX DATASETS",
            "=" * 65,
            f"{'Critère':<22} {'Dataset 1':>18}  {'Dataset 2':>18}",
            "─" * 65,
            f"{'Label':<22} {loader1.label:>18}  {loader2.label:>18}",
            f"{'Observations':<22} {i1['obs']:>18,}  {i2['obs']:>18,}",
            f"{'Variables totales':<22} {i1['vars']:>18}  {i2['vars']:>18}",
            f"{'Variables quant':<22} {i1['n_quant']:>18}  {i2['n_quant']:>18}",
            f"{'Variables cat':<22} {i1['n_cat']:>18}  {i2['n_cat']:>18}",
            f"{'Enquêtes':<22} {str(sorted(i1['surveys'])):>18}  {str(sorted(i2['surveys'])):>18}",
            "─" * 65,
            f"Variables communes     : {len(cols_common)}",
            f"Uniquement dans DS1    : {len(cols_only_1)}"
            + (f" — {list(cols_only_1)[:3]}..." if cols_only_1 else ""),
            f"Uniquement dans DS2    : {len(cols_only_2)}"
            + (f" — {list(cols_only_2)[:3]}..." if cols_only_2 else ""),
            "=" * 65,
        ]

        # Comparer les statistiques descriptives sur colonnes communes
        if cols_common:
            lines.append("\nDifférences sur variables communes (quant) :")
            quant_common = [c for c in cols_common
                            if loader1.types.get(c) == "quant"
                            and loader2.types.get(c) == "quant"]
            if quant_common:
                lines.append(f"  {'Variable':<20} {'Moy DS1':>10} {'Moy DS2':>10} "
                             f"{'Std DS1':>10} {'Std DS2':>10}")
                lines.append("  " + "─" * 55)
                for col in sorted(quant_common)[:10]:  # top 10
                    m1 = loader1.df[col].dropna().mean()
                    m2 = loader2.df[col].dropna().mean()
                    s1 = loader1.df[col].dropna().std()
                    s2 = loader2.df[col].dropna().std()
                    lines.append(f"  {col:<20} {m1:>10.2f} {m2:>10.2f} "
                                f"{s1:>10.2f} {s2:>10.2f}")
                if len(quant_common) > 10:
                    lines.append(f"  ... et {len(quant_common)-10} autres variables")
            else:
                lines.append("  (aucune variable quant commune)")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_rds(
    output_path : str | Path,
    method      : str = "normal",
    n_obs       : int = 500,
    seed        : int = 42,
    label       : str = "",
) -> Path:
    """
    Génère un fichier .rds synthétique unique contenant toutes les enquêtes.

    Le fichier contient une colonne 'survey' identifiant l'enquête.
    Deux appels avec method='normal' et method='bootstrap' produisent
    deux synthèses différentes — pour simuler vos deux fichiers réels.

    Paramètres
    ----------
    output_path : chemin du fichier .rds à générer
    method      : 'normal'    → tirage gaussien / log-normal (synthèse 1)
                  'bootstrap' → bootstrap avec bruit (synthèse 2)
    n_obs       : observations par enquête (défaut : 500)
    seed        : graine aléatoire (défaut : 42)
    label       : description du fichier (pour les logs)

    Retourne
    --------
    Path du fichier .rds généré
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"🔧 Génération synthétique — méthode : {method} | label : {label}")

    # ── Définition des variables par enquête ──────────────────────────────────
    # Chaque enquête génère ses propres colonnes
    specs: Dict[str, Dict] = {
        "EU-SILC": {
            "HY020"  : ("quant", lambda n: rng.lognormal(7.5, 0.8, n).round(0)),
            "HY040N" : ("quant", lambda n: rng.lognormal(6.2, 1.2, n).round(0) * rng.choice([0,1], n, p=[0.4,0.6])),
            "PL073"  : ("cat",   lambda n: rng.choice([0,1,2,3], n, p=[0.1,0.6,0.2,0.1])),
            "PL060"  : ("quant", lambda n: rng.normal(38, 8, n).clip(0, 80).round(1)),
            "PE040"  : ("cat",   lambda n: rng.choice([1,2,3,4,5], n)),
            "PH010"  : ("cat",   lambda n: rng.choice([1,2,3,4,5], n, p=[0.2,0.35,0.25,0.15,0.05])),
            "HH010"  : ("cat",   lambda n: rng.choice([1,2,3], n, p=[0.55,0.35,0.10])),
            "RB090"  : ("cat",   lambda n: rng.choice([1,2], n, p=[0.51,0.49])),
            "RB080"  : ("quant", lambda n: rng.integers(1940, 2005, n)),
            "DB040"  : ("cat",   lambda n: rng.choice(["BE1","BE2","BE3"], n)),
        },
        "HFCS": {
            "DA1000" : ("quant", lambda n: rng.lognormal(11.5, 1.5, n).round(0)),
            "DL1000" : ("quant", lambda n: rng.lognormal(10.2, 1.0, n).round(0)),
            "HB0900" : ("cat",   lambda n: rng.choice([1,2,3], n, p=[0.65,0.30,0.05])),
            "DI1400" : ("quant", lambda n: rng.lognormal(7.8, 0.9, n).round(0)),
            "PE0100" : ("cat",   lambda n: rng.choice([1,2,3,4,5], n)),
            "RA0100" : ("quant", lambda n: rng.integers(25, 80, n)),
            "RA0200" : ("cat",   lambda n: rng.choice([1,2], n, p=[0.50,0.50])),
        },
        "EU-LFS": {
            "HWUSUAL": ("quant", lambda n: rng.normal(38, 9, n).clip(0, 80).round(0)),
            "HWACTUAL":("quant", lambda n: rng.normal(36, 10, n).clip(0, 80).round(0)),
            "ILOSTAT": ("cat",   lambda n: rng.choice([1,2,3], n, p=[0.65,0.12,0.23])),
            "STAPRO" : ("cat",   lambda n: rng.choice([1,2,3,4], n, p=[0.68,0.12,0.12,0.08])),
            "NACE1D" : ("cat",   lambda n: rng.choice(["A","C","G","K","M","Q"], n)),
            "EDUC"   : ("cat",   lambda n: rng.choice([0,1,2,3], n, p=[0.10,0.30,0.35,0.25])),
            "AGE"    : ("quant", lambda n: rng.integers(15, 75, n)),
        },
        "HBS": {
            "TOTEXP"  : ("quant", lambda n: rng.lognormal(8.5, 0.7, n).round(0)),
            "FOODEXP" : ("quant", lambda n: rng.lognormal(7.0, 0.6, n).round(0)),
            "HOUEXP"  : ("quant", lambda n: rng.lognormal(7.8, 0.8, n).round(0)),
            "HHTYPE"  : ("cat",   lambda n: rng.choice([1,2,3,4,5], n)),
            "HHSIZE"  : ("cat",   lambda n: rng.choice([1,2,3,4,5,6], n)),
            "INCOME"  : ("quant", lambda n: rng.lognormal(7.6, 0.85, n).round(0)),
            "REGION"  : ("cat",   lambda n: rng.choice(["BE1","BE2","BE3"], n)),
        },
        "IPCAL": {
            "rev_prof"  : ("quant", lambda n: rng.lognormal(10.0, 0.9, n).round(0)),
            "rev_imm"   : ("quant", lambda n: rng.lognormal(8.5, 1.2, n).round(0) * rng.choice([0,1], n, p=[0.5,0.5])),
            "impot_du"  : ("quant", lambda n: rng.lognormal(8.8, 0.8, n).round(0)),
            "cat_contri": ("cat",   lambda n: rng.choice([1,2,3,4], n, p=[0.40,0.30,0.20,0.10])),
            "nb_enfants": ("cat",   lambda n: rng.choice([0,1,2,3,4], n, p=[0.35,0.25,0.25,0.12,0.03])),
        },
        "DEMOBEL": {
            "abo_h_fm"  : ("quant", lambda n: rng.lognormal(7.5, 0.8, n).round(0)),
            "abo_p_age" : ("quant", lambda n: rng.integers(15, 90, n)),
            "abo_p_sex" : ("cat",   lambda n: rng.choice([1,2], n, p=[0.50,0.50])),
            "abo_p_educ": ("cat",   lambda n: rng.choice([1,2,3,4], n)),
            "abo_p_stat": ("cat",   lambda n: rng.choice([1,2,3,4,5], n)),
        },
    }

    blocks = []
    for survey_name, variables in specs.items():
        cols = {col: func(n_obs) for col, (_, func) in variables.items()}
        df_block = pd.DataFrame(cols)

        # ── Méthode de synthèse ───────────────────────────────────────────────
        if method == "bootstrap":
            # Bootstrap : ré-échantillonnage avec légère perturbation
            idx = rng.integers(0, n_obs, n_obs)
            df_block = df_block.iloc[idx].reset_index(drop=True)
            # Bruit additif sur les colonnes numériques
            for col, (vtype, _) in variables.items():
                if vtype == "quant":
                    noise = rng.normal(0, df_block[col].std() * 0.05, n_obs)
                    df_block[col] = (df_block[col] + noise).round(2)

        # NaN réalistes (~5%)
        for col in df_block.columns:
            mask = rng.random(n_obs) < 0.05
            df_block.loc[mask, col] = np.nan

        df_block["survey"] = survey_name
        blocks.append(df_block)

    df_unified = pd.concat(blocks, axis=0, ignore_index=True)
    pyreadr.write_rds(str(output_path), df_unified)

    log.info(f"✅ Généré : {output_path.name} "
             f"({df_unified.shape[0]:,} obs × {df_unified.shape[1]} cols) "
             f"— méthode : {method}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# DÉMONSTRATION
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    print("=" * 65)
    print("STEP 6 — DÉMONSTRATION DataLoader")
    print("=" * 65)

    rds_dir   = Path("data/rds")
    cache_dir = Path("data/cache")
    rds_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # DÉTECTION AUTOMATIQUE DES FICHIERS À CHARGER
    #
    # Priorité 1 — Vos fichiers réels (CVAE et GAN)
    #   → Placez-les dans data/rds/  avec leurs noms d'origine
    #   → Le code les détecte automatiquement
    #
    # Priorité 2 — Fallback synthétique
    #   → Si aucun fichier réel n'est trouvé, génère deux fichiers
    #     synthétiques pour permettre les tests sans données réelles
    # ══════════════════════════════════════════════════════════════════════

    # Noms des fichiers réels (adaptez si besoin)
    REAL_FILE_1 = rds_dir / "beamm.brussels-250528-CVAE.rds"
    REAL_FILE_2 = rds_dir / "beamm.brussels-250528-GAN.rds"

    # ── Priorité 1 : fichiers réels présents ──────────────────────────────
    if REAL_FILE_1.exists() and REAL_FILE_2.exists():
        print(f"\n✅ Fichiers réels détectés :")
        print(f"   • {REAL_FILE_1.name}")
        print(f"   • {REAL_FILE_2.name}")
        path_v1  = REAL_FILE_1
        path_v2  = REAL_FILE_2
        label_v1 = "CVAE"
        label_v2 = "GAN"

    elif REAL_FILE_1.exists():
        print(f"\n⚠️  Un seul fichier réel trouvé : {REAL_FILE_1.name}")
        print("   Génération du fichier de comparaison synthétique (GAN simulé)...")
        path_v1  = REAL_FILE_1
        label_v1 = "CVAE"
        path_v2  = rds_dir / "synthetic_GAN_fallback.rds"
        label_v2 = "GAN (synthétique)"
        if not path_v2.exists():
            generate_synthetic_rds(path_v2, method="bootstrap", seed=99)

    elif REAL_FILE_2.exists():
        print(f"\n⚠️  Un seul fichier réel trouvé : {REAL_FILE_2.name}")
        print("   Génération du fichier de comparaison synthétique (CVAE simulé)...")
        path_v1  = rds_dir / "synthetic_CVAE_fallback.rds"
        label_v1 = "CVAE (synthétique)"
        path_v2  = REAL_FILE_2
        label_v2 = "GAN"
        if not path_v1.exists():
            generate_synthetic_rds(path_v1, method="normal", seed=42)

    # ── Priorité 2 : aucun fichier réel — génération synthétique ─────────
    else:
        print("\n⚠️  Fichiers réels introuvables dans data/rds/")
        print("   Attendus :")
        print(f"     • {REAL_FILE_1}")
        print(f"     • {REAL_FILE_2}")
        print("\n   → Génération de données synthétiques pour les tests...\n")
        path_v1  = rds_dir / "synthetic_v1.rds"
        path_v2  = rds_dir / "synthetic_v2.rds"
        label_v1 = "CVAE (synthétique)"
        label_v2 = "GAN (synthétique)"
        if not path_v1.exists():
            generate_synthetic_rds(path_v1, method="normal",    seed=42, label="CVAE synthétique")
        if not path_v2.exists():
            generate_synthetic_rds(path_v2, method="bootstrap", seed=99, label="GAN synthétique")

    # ── MODE PRINCIPAL : chargement des deux fichiers ─────────────────────────
    print("\n" + "─" * 65)
    print(f"MODE PRINCIPAL — Chargement du fichier 1 : {path_v1.name}")
    print("─" * 65)
    try:
        loader_v1 = DataLoader.from_single_file(path_v1, label=label_v1)
    except RuntimeError as e:
        # pyreadr, rpy2 et Rscript ont tous échoué → afficher le script R
        print(str(e))
        DataLoader.print_r_conversion_script(path_v1)
        print("\n❌ Arrêt : convertissez les fichiers .rds depuis R puis relancez.")
        return
    print("\n" + loader_v1.summary())

    print("\n" + "─" * 65)
    print(f"MODE PRINCIPAL — Chargement du fichier 2 : {path_v2.name}")
    print("─" * 65)
    try:
        loader_v2 = DataLoader.from_single_file(path_v2, label=label_v2)
    except RuntimeError as e:
        print(str(e))
        DataLoader.print_r_conversion_script(path_v2)
        print("\n❌ Arrêt : convertissez les fichiers .rds depuis R puis relancez.")
        return
    print("\n" + loader_v2.summary())

    # ── Comparaison des deux datasets ─────────────────────────────────────────
    print("\n" + "─" * 65)
    print("COMPARAISON DES DEUX DATASETS")
    print("─" * 65)
    print(DataLoader.compare(loader_v1, loader_v2))

    # ── Cache Parquet ──────────────────────────────────────────────────────────
    print("\n♻️  Test cache Parquet...")
    loader_v1.save_cache(cache_dir / "v1.parquet")
    loader_reload = DataLoader.from_cache(cache_dir / "v1.parquet")
    assert loader_reload.df.shape == loader_v1.df.shape
    print("   ✅ Cache OK")

    # ── MODE FUTUR : dossier par enquête ──────────────────────────────────────
    folder_dir = rds_dir / "by_survey"
    if not folder_dir.exists():
        print("\n🔧 Génération des fichiers par enquête (Mode Futur)...")
        folder_dir.mkdir(parents=True, exist_ok=True)
        rng_f = np.random.default_rng(42)
        n = 300
        for stem, survey, n_cols in [
            ("eu_silc", "EU-SILC", 5), ("hfcs", "HFCS", 4),
            ("eu_lfs", "EU-LFS", 5),   ("hbs", "HBS", 4),
        ]:
            df_s = pd.DataFrame({
                f"var_{i}": rng_f.lognormal(7, 1, n).round(0) if i % 2 == 0
                            else rng_f.choice([1,2,3], n)
                for i in range(n_cols)
            })
            pyreadr.write_rds(str(folder_dir / f"{stem}.rds"), df_s)

    print("\n" + "─" * 65)
    print("MODE FUTUR — Dossier par enquête")
    print("─" * 65)
    loader_folder = DataLoader.from_survey_folder(folder_dir, label="Real data (futur)")
    print("\n" + loader_folder.summary())

    # ── Vérifications ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("VÉRIFICATIONS")
    print("─" * 65)
    checks = [
        (not loader_v1.df.empty,              "Fichier 1 chargé"),
        (not loader_v2.df.empty,              "Fichier 2 chargé"),
        (len(loader_v1.surveys) > 0,          "Enquêtes détectées (v1)"),
        (len(loader_v2.surveys) > 0,          "Enquêtes détectées (v2)"),
        (loader_v1.df.shape[0] > 0,           "Observations > 0 (v1)"),
        (loader_v2.df.shape[0] > 0,           "Observations > 0 (v2)"),
        ("quant" in loader_v1.types.values(), "Types 'quant' détectés"),
        ("cat"   in loader_v1.types.values(), "Types 'cat' détectés"),
        (loader_reload.label == loader_v1.label, "Label conservé après cache"),
        (not loader_folder.df.empty,          "Mode Futur fonctionne"),
    ]
    all_ok = True
    for cond, label in checks:
        icon = "✅" if cond else "❌"
        print(f"  {icon} {label}")
        if not cond:
            all_ok = False

    print("\n" + "=" * 65)
    if all_ok:
        print("✅ TOUS LES TESTS PASSÉS")
        print("🚀 Prochaine étape : python step7_association.py")
    else:
        print("❌ CERTAINS TESTS ÉCHOUÉS — voir les logs")
    print("=" * 65)


if __name__ == "__main__":
    demo()