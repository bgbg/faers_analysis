import io
import json
import logging
import os
from collections import namedtuple
from glob import glob

import pandas as pd
import numpy as np
import scipy.stats as stats
import re

logger = logging.getLogger("FAERS")
import tqdm
import base64


def html_from_fig(fig, caption=None, width=None):
    figfile = io.BytesIO()
    fig.savefig(figfile, format="png")
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = figfile.getvalue()  # extract string (stream of bytes)
    figdata_png = base64.b64encode(figdata_png).decode("utf8")
    if width is None:
        width = ""
    else:
        width = f'width="{width}"'
    ret = f'<figure><img src="data:image/png;base64,{figdata_png}"{width}>'
    if caption is not None:
        ret += f"<figcaption>{caption}</figcaption>"
    ret += "</figure>"
    return ret


class Quarter:
    def __init__(self, *args):
        if len(args) == 2:
            self.year = int(args[0])
            self.quarter = int(args[1])
        else:
            self.year, self.quarter = self.parse_string(args[0])

        self.__verify()

    @staticmethod
    def parse_string(s):
        m = re.search(r"^(\d\d\d\d)-?q(\d)$", s)
        if m is None:
            raise RuntimeError(
                f'Quarter definition "{s}" does not follow the format YYYYqQ'
            )
        year, quarter = m.groups()
        year = int(year)
        quarter = int(quarter)
        return (year, quarter)

    def __verify(self):
        assert 1900 < self.year < 2100
        assert self.quarter in {1, 2, 3, 4}

    def increment(self):
        year = self.year
        quarter = self.quarter
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
        return Quarter(year, quarter)

    def __eq__(self, other):
        if other is self:
            return True
        return (self.year == other.year) and (self.year == other.year)

    def __lt__(self, other):
        if self.year == other.year:
            return self.quarter < other.quarter
        else:
            return self.year < other.year

    def __hash__(self):
        return hash((self.year, self.quarter))

    def __str__(self):
        return f"{self.year}q{self.quarter}"


def generate_quarters(start, end):
    while start < end:  # NOTE: not including *end*
        yield start
        start = start.increment()


class ContingencyMatrix:
    def __init__(self, tbl=None):
        if tbl is None or tbl.empty:
            tbl = pd.DataFrame(columns=["exposure", "outcome", "n"])
        else:
            if tbl.shape == (2, 2):
                try:
                    tbl = (
                        pd.melt(
                            tbl.reset_index().fillna(0),
                            id_vars=["exposure"],
                            value_vars=[False, True],
                            value_name="n",
                        )
                        .set_index(["exposure", "outcome"])
                        .sort_index()
                    )
                except:
                    tbl = pd.DataFrame(columns=["exposure", "outcome", "n"])
            else:

                for c in ["exposure", "outcome", "n"]:
                    assert c in tbl.columns
                for c in ["exposure", "outcome"]:
                    tbl[c] = tbl[c].astype(bool)
                tbl = tbl.set_index(["exposure", "outcome"]).sort_index()
                for expo in (True, False):
                    for outcome in (True, False):
                        pair = (expo, outcome)
                        if pair not in tbl.index:
                            row = pd.Series({"n": 0}, name=pair)
                            tbl = tbl.append(row)
        self.tbl = tbl.sort_index()

    def __add__(self, other):
        self.tbl.n += other.tbl.n
        return self

    EXCLUSION_SEMANTICS_ELIGIBILITY = "eligibility"
    EXCLUSION_SEMANTICS_OUTCOME = "outcome"
    _EXCLUSION_SEMANTICS = (
        EXCLUSION_SEMANTICS_ELIGIBILITY,
        EXCLUSION_SEMANTICS_OUTCOME,
    )

    @classmethod
    def from_results_table(
        cls, data, config, exclusion_semantics=EXCLUSION_SEMANTICS_ELIGIBILITY
    ):
        """Build the 2x2 exposure/outcome table for one question.

        A case is counted iff it reports >= 1 PT in the inclusion set `I`
        (``config.reactions``) and 0 PTs in the exclusion set `E`
        (``config.exclusions``) -- lab issue #4, R23. That specification is
        ambiguous between two designs (methodology entry 31, D4), both
        selectable via `exclusion_semantics`:

        - "eligibility" (default / entry 31's "beta"): a case reporting any
          `E` term is removed from the analysis entirely, from the exposed
          and the unexposed arm alike.
        - "outcome" (entry 31's "alpha"): such a case stays in the table
          and is forced outcome-negative.

        When `config.exclusions` is empty/None -- i.e. `E` is empty -- both
        semantics are no-ops and this reproduces the pre-#4 pipeline
        exactly (R24).
        """
        if exclusion_semantics not in cls._EXCLUSION_SEMANTICS:
            raise ValueError(
                f"exclusion_semantics must be one of {cls._EXCLUSION_SEMANTICS}, "
                f"got {exclusion_semantics!r}"
            )
        exposure = data[f"exposed {config.name}"].copy()
        exposure.name = "exposure"
        outcome = data[f"reacted {config.name}"].copy()
        outcome.name = "outcome"

        excluded_col = f"excluded {config.name}"
        if excluded_col in data.columns:
            excluded = data[excluded_col]
            if exclusion_semantics == cls.EXCLUSION_SEMANTICS_ELIGIBILITY:
                eligible = ~excluded
                exposure = exposure.loc[eligible]
                outcome = outcome.loc[eligible]
            else:  # EXCLUSION_SEMANTICS_OUTCOME
                outcome = outcome & ~excluded
                outcome.name = "outcome"  # `&` between differently-named
                # Series drops the name; crosstab/melt below rely on it.

        crosstab = (
            pd.crosstab(exposure, outcome)
            .reindex([False, True], axis=0, fill_value=0)
            .reindex([False, True], axis=1, fill_value=0)
        )
        return ContingencyMatrix(crosstab)

    def get_count_value(self, exposure, outcome):
        ret = self.tbl.loc[(exposure, outcome)]["n"]
        return ret

    def ror_components(self, smoothing=0):
        # Smoothing is mentioned here https://pdfs.semanticscholar.org/9639/66a1e9ee60bfcdb13a1a98527022c7cc59ba.pdf
        a = self.get_count_value(True, True)
        b = self.get_count_value(True, False)
        c = self.get_count_value(False, True)
        d = self.get_count_value(False, False)
        assert a + b + c + d == self.tbl.n.sum()
        if smoothing < 0:
            smoothing = 1 / self.tbl.n.sum()
        return (a + smoothing, b + smoothing, c + smoothing, d + smoothing)

    def ror(self, alpha=0.05, smoothing=0):
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2938757/
        a, b, c, d = self.ror_components(smoothing=smoothing)
        nominator = a * c
        denominator = b * c
        if denominator:
            ror = (a * d) / (b * c)
        else:
            ror = np.nan
        if alpha is not None:
            if np.all(np.array([a, b, c, d], dtype=bool)):
                # eq 2 from https://arxiv.org/pdf/1307.1078.pdf
                ln_ror = np.log(ror)
                standard_error_ln_ror = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
                interval = np.multiply(
                    stats.distributions.norm.interval(1 - alpha), standard_error_ln_ror
                )
                ci_ln_ror = ln_ror + interval
                ci = tuple(np.exp(ci_ln_ror))
            else:
                ci = (np.nan, np.nan)
            return ror, ci
        else:
            return ror

    def crosstab(self):
        ret = self.tbl.pivot_table(
            values="n", index="exposure", columns="outcome", aggfunc="sum"
        )
        return ret

    def __str__(self):
        ret = "Contingency matrix\n" + str(self.tbl)
        return ret

    def __repr__(self):
        return self.__str__()


class QuestionConfig:
    def __init__(self, name, drugs, reactions, control, exclusions=None):
        self.name = name
        self.drugs = drugs
        self.reactions = reactions
        self.control = control
        # `exclusions` is the exclusion term set E of lab issue #4 (R23): a
        # set of PTs on the *reaction* axis, distinct from the inclusion set
        # `reactions` (I). None/empty means the question has no exclusion
        # set, which must reproduce the pre-#4 pipeline exactly (R24).
        self.exclusions = exclusions

    @classmethod
    def load_config_items(cls, dir_config):
        ret = []
        for f in sorted(glob(os.path.join(dir_config, "*.json"))):
            ret.append(cls.config_from_json_file(f))

        for f in glob(os.path.join(dir_config, "[a-zA-Z0-9]*.xls?")):
            ret.extend(cls.configs_from_excel_file(f))
        return ret

    @staticmethod
    def normalize_reaction_name(r):
        return r.strip().lower()

    @staticmethod
    def normalize_drug_name(d):
        ret = d.strip().lower()
        if ret.endswith("."):
            ret = ret[:-1]
        return ret

    @classmethod
    def config_from_json_file(cls, fn):
        name = os.path.split(fn)[-1].replace(".json", "")
        config = json.load(open(fn))
        drugs = [cls.normalize_drug_name(d) for d in config["drug"]]
        reactions = [cls.normalize_reaction_name(r) for r in config["reaction"]]
        if "control" in config:
            control = [cls.normalize_drug_name(d) for d in config["control"]]
        else:
            control = None
        if "exclusion" in config:
            exclusions = [cls.normalize_reaction_name(r) for r in config["exclusion"]]
        else:
            exclusions = None
        return QuestionConfig(
            name,
            drugs=drugs,
            reactions=reactions,
            control=control,
            exclusions=exclusions,
        )

    @classmethod
    def configs_from_excel_file(cls, fn):
        name = os.path.splitext(os.path.split(fn)[-1])[0]
        xl = pd.ExcelFile(fn)
        ret = []
        for sheetname in xl.sheet_names:
            tbl = xl.parse(sheetname)
            tbl.columns = [c.lower().strip() for c in tbl.columns]
            if len(tbl.columns) > 4:
                logger.warning(
                    f"Skipping {fn} sheet {sheetname} that has {len(tbl.columns)} columns"
                )
                continue
            drugs = tbl["drug"].dropna().tolist()
            reactions = tbl["reaction"].dropna().tolist()
            drugs = [cls.normalize_drug_name(d) for d in drugs]
            reactions = [cls.normalize_reaction_name(r) for r in reactions]
            if "control" in tbl.columns:
                control = [
                    cls.normalize_drug_name(d) for d in tbl["control"].dropna().tolist()
                ]
            else:
                control = None
            if "exclusion" in tbl.columns:
                exclusions = [
                    cls.normalize_reaction_name(r)
                    for r in tbl["exclusion"].dropna().tolist()
                ]
            else:
                exclusions = None
            ret.append(
                QuestionConfig(
                    f"{name} - {sheetname}",
                    drugs=drugs,
                    reactions=reactions,
                    control=control,
                    exclusions=exclusions,
                )
            )
        return ret

    def filename_from_config(self, directory, extension=".csv"):
        if extension:
            assert extension.startswith(".")
        config_name = self.name
        return os.path.join(directory, f"{config_name}{extension}")

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


def read_demo_data(fn_demo, **kwargs):
    dtypes = {
        "caseid": str,
        "event_dt_num": str,
        "age": float,
        "age_cod": str,
        "sex": str,
        "wt": float,
        "wt_cod": str,
    }
    df_demo = pd.read_csv(fn_demo, dtype=dtypes, usecols=dtypes.keys(), **kwargs)

    to_year_conversion_factor = {
        "YR": 1.0,
        "DY": 365.25,
        "MON": 12,
        "DEC": 0.1,
        "WK": 52.2,
        "HR": 24 * 365.25,
    }
    to_year_conversion_factor = pd.Series(to_year_conversion_factor)

    to_kg_conversion_factor = {"KG": 1.0, "LBS": 2.20462}
    to_kg_conversion_factor = pd.Series(to_kg_conversion_factor)
    df_demo.wt = (
        df_demo.wt / to_kg_conversion_factor.reindex(df_demo.wt_cod.values).values
    )
    df_demo.age = (
        df_demo.age / to_year_conversion_factor.reindex(df_demo.age_cod.values).values
    )
    df_demo["event_date"] = pd.to_datetime(
        df_demo.event_dt_num, dayfirst=False, errors="ignore"
    )
    df_demo.drop(["age_cod", "wt_cod", "event_dt_num"], axis=1, inplace=True)
    return df_demo


def read_therapy_data(fn_therapy, **kwargs):
    dtypes = {"caseid": str, "dur": float, "dur_cod": str}
    df_therapy = pd.read_csv(fn_therapy, dtype=dtypes, usecols=dtypes.keys(), **kwargs)
    to_day_conversion_factor = {
        "MON": 30.5,
        "YR": 365.25,
        "WK": 7,
        "DAY": 1,
        "HR": 1 / 24.0,
        "MIN": 1 / 24.0 / 60,
    }
    to_day_conversion_factor = pd.Series(to_day_conversion_factor)
    df_therapy["to_day_factor"] = to_day_conversion_factor.reindex(
        df_therapy.dur_cod
    ).values
    df_therapy["duration_days"] = df_therapy.dur * df_therapy.to_day_factor
    return df_therapy[["caseid", "duration_days"]]


def compute_df_uniqueness(df, cols=None, do_print=False, print_prefix=None):
    if do_print and print_prefix is None:
        print_prefix = ""
    n_total = len(df)
    n_unique = len(df.drop_duplicates(subset=cols))
    n_unique, n_total
    frac_unique = n_unique / n_total
    if do_print:
        print(
            f"{print_prefix}Of {n_total:6,d} entries {n_unique:6,d} are unique ({frac_unique*100:.1f}%)"
        )
    return frac_unique
