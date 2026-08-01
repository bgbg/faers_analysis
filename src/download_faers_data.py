import logging
import os
import shutil
import urllib.request
import defopt
from multiprocessing.dummy import Pool as ThreadPool
import tqdm

from src.utils import Quarter, generate_quarters

logger = logging.getLogger("FAERS")

WHAT = [
        "demo",
        "drug",
        "reac",
        "outc",
        # "indi",
        "ther"]

def quarter_urls(quarter):
    ret = []
    year = quarter.year
    yearquarter = str(quarter)
    for w in WHAT:
        # tmplt = f"https://data.nber.org/fda/faers/{year}/csv/{w}{yearquarter}.csv.zip"
        # tmplt = f"https://data.nber.org/fda/faers/{year}/{w}{yearquarter}.csv.zip"
        if year <= 2018:
            tmplt = (
                f"https://data.nber.org/fda/faers/{year}/{w}{yearquarter}.csv.zip"
            )
        else:
            tmplt = f"https://data.nber.org/fda/faers/{year}/csv/{w}{yearquarter}.csv.zip"
        ret.append(tmplt)
    return ret

def get_expected_filenames(year_q_from, year_q_to, dir_out):
    quarters_to_process = generate_quarters(Quarter(year_q_from), Quarter(year_q_to))
    expected_files = []
    for q in quarters_to_process:
        for w in WHAT:
            yearquarter=str(q)
            filename = f"{w}{yearquarter}.csv.zip"
            expected_files.append(os.path.join(dir_out, filename))
    return expected_files

def download_url(url, dir_out):
    fn_out = os.path.split(url)[-1]
    fn_out = os.path.join(dir_out, fn_out)
    if os.path.exists(fn_out):
        logger.debug(f"Skipping {url} because {fn_out} already exists")
        return
    try:
        urllib.request.urlretrieve(url, fn_out)
    except Exception as err:
        logger.error(f"Failed to download {url} to {fn_out} {err}")
    else:
        logger.info(f"Saved {fn_out}")
        assert os.path.exists(fn_out)


def main(*, year_q_from, year_q_to, dir_out, threads=4, clean_on_failure=True):
    """

    :param str year_q_from:
        XXXXqQ, where XXXX is the year, q is the literal "q" and Q is 1, 2, 3 or 4
    :param str year_q_to:
        XXXXqQ, where XXXX is the year, q is the literal "q" and Q is 1, 2, 3 or 4
    :param str dir_out:
        Output directory
    :param int threads:
        N of parallel threads
    :param bool clean_on_failure:
        ???

    :return: None

    """
    dir_out = os.path.abspath(dir_out)
    os.makedirs(dir_out, exist_ok=True)
    try:
        q_first = Quarter(year_q_from)
        q_last = Quarter(year_q_to)
        urls = []
        for q in generate_quarters(q_first, q_last):
            urls.extend(quarter_urls(q))

        print(f"will download {len(urls)} urls")
        with ThreadPool(threads) as pool:
            _ = list(
                tqdm.tqdm(
                    pool.imap(lambda url: download_url(url, dir_out), urls),
                    total=len(urls),
                )
            )
    except Exception as err:
        if clean_on_failure:
            shutil.rmtree(dir_out)
        raise err


if __name__ == "__main__":
    defopt.run(main)
