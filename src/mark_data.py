import os
import logging
import shutil
import warnings
import pickle
from multiprocessing import Pool
from functools import partial

import pandas as pd
import defopt
import numpy as np
import tqdm

from src import utils
from src.utils import Quarter, generate_quarters, QuestionConfig

logger = logging.getLogger("FAERS")


def mark_drug_data(df, drug_names):
    df.drugname = df.drugname.apply(QuestionConfig.normalize_drug_name)
    for drug in sorted(drug_names):
        df[f"drug {drug}"] = df.drugname == drug
    drug_columns = [f"drug {drug}" for drug in drug_names]
    ret = df.groupby("caseid")[drug_columns].any()
    return ret


def mark_reaction_data(df, reaction_types):
    df.pt = df.pt.apply(QuestionConfig.normalize_reaction_name)

    # Create a dict of new columns
    new_cols = {
        f"reaction {reaction}": df.pt == reaction for reaction in sorted(reaction_types)
    }

    # Concatenate in a single operation to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    reaction_columns = list(new_cols.keys())
    ret = df.groupby("caseid")[reaction_columns].any()

    return ret


def handle_duplicates_within_case(df, cols_boolean, cols_rest):
    if df.shape[0] == 1:
        return df.iloc[0]
    ret = df[cols_rest].iloc[-1]  # take the last available data
    ret["q"] = df["q"].iloc[0]  # take the earliest timepoint
    ret = pd.concat([ret, df[cols_boolean].any(axis=0)])
    return ret


def handle_duplicates(df):
    cols_boolean = [c for c in df.columns if df.dtypes[c] == np.dtype(bool)]
    cols_rest = [c for c in df.columns if (c not in cols_boolean) and (c != "q")]
    rows_per_caseid = df["caseid"].value_counts()
    df["rows_per_caseid"] = rows_per_caseid.reindex(df.caseid).values
    sel = df.rows_per_caseid == 1
    already_good = df.loc[sel]
    logger.info(f"{sel.sum():,d} rows are already good")
    need_to_fix = df.loc[~sel]
    logger.info(f"{len(need_to_fix):,d} rows need some fixing")
    fixed = need_to_fix.groupby("caseid").apply(
        lambda d: handle_duplicates_within_case(d, cols_boolean, cols_rest)
    )
    logger.info(f"Done fixing, combining the results")
    ret = pd.concat([already_good, fixed], sort=False)
    uniqueness = utils.compute_df_uniqueness(ret, ["caseid"], do_print=False)
    assert uniqueness == 1.0
    return ret.set_index("caseid")


def mark_data(df_drug, df_reac, df_demo, config_items):
    logger.info("Marking the data")
    cols_to_collect = list(df_demo.columns)
    df_merged = df_demo.join(df_reac).join(df_drug)
    for config in config_items:
        logger.info(f"Config {config.name[0:40]}....")
        drugs_curr = set(config.drugs)
        drug_columns = [f"drug {drug}" for drug in drugs_curr]
        exposed = f"exposed {config.name}"
        df_merged[exposed] = df_merged[drug_columns].any(axis=1)
        cols_to_collect.append(exposed)
        if config.control is not None:
            control_columns = [f"drug {drug}" for drug in config.control]
            control = f"control {config.name}"
            df_merged[control] = df_merged[control_columns].any(axis=1)
            cols_to_collect.append(control)
        reaction_columns = [f"reaction {reaction}" for reaction in config.reactions]
        reacted = f"reacted {config.name}"
        df_merged[reacted] = df_merged[reaction_columns].any(axis=1)
        cols_to_collect.append(reacted)
        if config.exclusions:
            exclusion_columns = [
                f"reaction {reaction}" for reaction in config.exclusions
            ]
            excluded = f"excluded {config.name}"
            df_merged[excluded] = df_merged[exclusion_columns].any(axis=1)
            cols_to_collect.append(excluded)
    df_merged = df_merged[cols_to_collect].reset_index().sort_values(["caseid", "q"])
    logger.info(f"Handling duplicates of {len(df_merged):,d} rows")
    ret = handle_duplicates(df_merged)
    return ret


def load_quarder_files(template, quarters, **kwargs) -> pd.DataFrame:
    dtype = kwargs.pop("dtype", str)
    ret = []
    # Check if quarters is iterable but not a string
    if hasattr(quarters, "__iter__") and not isinstance(quarters, (str, bytes)):
        quarters_to_process = quarters
    else:
        # Handle the case where a single quarter is passed
        quarters_to_process = [quarters]

    for q in quarters_to_process:
        fn = template.replace("Q", str(q))
        tmp = pd.read_csv(fn, dtype=dtype, **kwargs)
        ret.append(tmp)
    return pd.concat(ret)


def process_quarters(
    quarters, dir_in, dir_out, config_items, drug_names, reaction_types
):
    DEBUG = None
    template_drug = os.path.join(dir_in, "drugQ.csv.zip")
    usecols = ["primaryid", "caseid", "drugname"]
    df_drug = load_quarder_files(
        template_drug, quarters, usecols=usecols, nrows=DEBUG
    ).dropna()
    df_drug = mark_drug_data(df_drug, drug_names)

    template_reac = os.path.join(dir_in, "reacQ.csv.zip")
    usecols = ["primaryid", "caseid", "pt"]
    df_reac = load_quarder_files(template_reac, quarters, usecols=usecols, nrows=DEBUG)
    df_reac = mark_reaction_data(df_reac, reaction_types)

    df_demo = []
    for q in quarters:
        fn_demo = os.path.join(dir_in, f"demo{q}.csv.zip")
        tmp = utils.read_demo_data(fn_demo, nrows=DEBUG).set_index("caseid")
        tmp["q"] = str(q)
        df_demo.append(tmp)
    df_demo = pd.concat(df_demo)
    df_marked = mark_data(
        df_drug=df_drug, df_reac=df_reac, df_demo=df_demo, config_items=config_items
    )
    logger.info("Marked the data, dumping the file")

    # Save the combined file
    pickle.dump(df_marked, open(os.path.join(dir_out, "marked_data.pkl"), "wb"))

    # Save individual quarterly files
    for q in quarters:
        df_q = df_marked[df_marked.q == str(q)]
        if not df_q.empty:
            pickle.dump(df_q, open(os.path.join(dir_out, f"{q}.pkl"), "wb"))
            logger.info(f"Saved quarterly file for {q}")

    return df_marked


def process_quarter_wrapper(
    q, dir_in, dir_out, config_items, drug_names, reaction_types
):
    """Wrapper function for process_quarter to use with multiprocessing"""
    output_file = os.path.join(dir_out, f"{q}.pkl")
    if os.path.exists(output_file):
        logger.debug(f"Skipping {q} because {output_file} already exists")
        return
    # Process the single quarter
    template_drug = os.path.join(dir_in, "drugQ.csv.zip")
    usecols = ["primaryid", "caseid", "drugname"]
    df_drug = load_quarder_files(template_drug, [q], usecols=usecols).dropna()
    df_drug = mark_drug_data(df_drug, drug_names)

    template_reac = os.path.join(dir_in, "reacQ.csv.zip")
    usecols = ["primaryid", "caseid", "pt"]
    df_reac = load_quarder_files(template_reac, [q], usecols=usecols)
    df_reac = mark_reaction_data(df_reac, reaction_types)

    fn_demo = os.path.join(dir_in, f"demo{q}.csv.zip")
    df_demo = utils.read_demo_data(fn_demo).set_index("caseid")
    df_demo["q"] = str(q)

    df_marked = mark_data(
        df_drug=df_drug, df_reac=df_reac, df_demo=df_demo, config_items=config_items
    )

    # Only save the quarterly file for this quarter

    logger.info(f"Saving quarterly file {output_file}")
    pickle.dump(df_marked, open(output_file, "wb"))

    return df_marked


def main(
    *,
    year_q_from,
    year_q_to,
    dir_in,
    config_dir,
    dir_out,
    threads=1,
    clean_on_failure=True,
):

    # --skip-if-exists --year-q-from=$(QUARTER_FROM) --year-q-to=$(QUARTER_TO) --dir-in=$(DIR_FAERS_DEDUPLICATED) --config-dir=$(CONFIG_DIR) --dir-out=$(DIR_MARKED_FILES) -t $(N_THREADS) --no-clean-on-failure
    """

    :param str year_q_from:
        XXXXqQ, where XXXX is the year, q is the literal "q" and Q is 1, 2, 3 or 4
    :param str year_q_to:
        XXXXqQ, where XXXX is the year, q is the literal "q" and Q is 1, 2, 3 or 4
    :param str dir_in:
        Input directory
    :param str config_dir:
        Directory with config files
    :param str dir_out:
        Output directory
    :param int threads:
        Threads in parallel processing
    :param bool clean_on_failure:
        ???

    :return: None

    """

    dir_out = os.path.abspath(dir_out)
    os.makedirs(dir_out, exist_ok=True)
    try:
        q_from = Quarter(year_q_from)
        q_to = Quarter(year_q_to)
        config_items = QuestionConfig.load_config_items(config_dir)
        drug_names = set()
        reaction_types = set()
        for config in config_items:
            drug_names.update(set(config.drugs))
            if config.control is not None:
                drug_names.update(set(config.control))
            reaction_types.update(set(config.reactions))
            if config.exclusions:
                reaction_types.update(set(config.exclusions))
        print(
            f"Will analyze {len(drug_names)} drugs and {len(reaction_types)} reactions"
        )
        quarters = list(generate_quarters(q_from, q_to))
        process_quarters(
            quarters,
            dir_in=dir_in,
            dir_out=dir_out,
            config_items=config_items,
            drug_names=drug_names,
            reaction_types=reaction_types,
        )
        with Pool(threads) as pool:
            wrapper_func = partial(
                process_quarter_wrapper,
                dir_in=dir_in,
                dir_out=dir_out,
                config_items=config_items,
                drug_names=drug_names,
                reaction_types=reaction_types,
            )
            _ = list(tqdm.tqdm(pool.imap(wrapper_func, quarters), total=len(quarters)))
    except Exception as err:
        if clean_on_failure:
            shutil.rmtree(dir_out)
        raise err


if __name__ == "__main__":
    _ = defopt.run(main)
