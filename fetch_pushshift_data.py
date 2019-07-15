"""Get data from pushshift and put into a pickles."""


import argparse
import collections
import copy
import itertools
import pickle
import logging
import os
import random
from pathlib import Path

import pandas as pd
from psaw import PushshiftAPI
from tqdm import tqdm

api = PushshiftAPI()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--out_path",
    type=str,
    default="./data/reddit_threads/",
    help="Path or url of the dataset. If empty download from S3.",
)
parser.add_argument(
    "-s",
    "--subreddit",
    type=str,
    action="append",
    default=[
        # "aww",
        # "funny",
        # "jokes",
        # "art",
        # "programmingcirclejerk",
        # "futurology",
        # "theonion",
        # "upliftingnews",
        # "news",
        # "bruhmoment",
        # "moviescirclejerk",
        # "copypasta",
        # "emojipasta",
        # "nosleep",
        # "rareinsults",
        # "psychonauts",
        # "squaredcircle",
        # "whowouldwin",
        # "Scotland",
        # "singularity",
        # "roast_me",
        # "RoastMe",
        # "OldieRoast",
        # "ScenesFromAHat",
        # "Showerthoughts"
    ],
    help="Subreddit names to scrape e.g. ' - s aww - s news '",
)
parser.add_argument(
    "-n",
    "--number_of_threads",
    type=int,
    default=10000,
    help="Number of threads to scrape",
)

args = parser.parse_args()
print(args)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


data_dir = Path(args.out_path)


def get_id_for_comments(thing):
    if thing["type"] == "submission":
        return "t3_" + thing["id"]
    else:
        return "t1_" + thing["id"]


def format_comments_dict(comment_dict, submission):
    # Now we want to reconstruct the comment heirachy.
    # 0. Init with the submission in the queue. start with this as target
    # 1. Look at target item in the queue, find it's top rated child comment
    #  1a. If it has one, pop it out, put it at end of queue, go to 1
    #  1b. If it doesn't have comment left, go to previous item in queue
    queue = [submission]
    submission_id = get_id_for_comments(submission)
    while len(list(itertools.chain(*comment_dict.values()))) > 0:
        for queue_position in range(len(queue) - 1, -1, -1):
            current_id = get_id_for_comments(queue[queue_position])
            found = comment_dict[current_id]
            if len(found):
                break
        next_comment = comment_dict[current_id].pop()
        queue.append(next_comment)

    # now format
    text = format_thread(queue, submission_id=submission_id)
    return text


def format_thing(thing, submission_id):
    if thing["type"] == "submission":
        return (
            "****S\n"
            + "\n".join([thing["url"], thing["title"], thing["selftext"]])
            + "\n****ES "
            + thing["id"]
            + "\n"
        )
    elif thing["parent_id"] == submission_id:
        return (
            "****T "
            + thing["parent_id"][3:]
            + "\n"
            + thing["body"]
            + "\n****ET "
            + thing["id"]
            + "\n"
        )
    else:
        return (
            "****R "
            + thing["parent_id"][3:]
            + "\n"
            + thing["body"]
            + "\n****ER "
            + thing["id"]
            + "\n"
        )


def format_thread(queue, submission_id):
    return "\n".join([format_thing(t, submission_id=submission_id) for t in queue])


def psaw_to_dict(thing):
    type_name = type(thing).__name__
    thing = thing.d_
    thing["type"] = type_name
    return thing


def comment_praw2psaw(comment_praw):
    """Convert praw comment to psaw type dict(ish)."""
    cp_dict = copy.deepcopy(comment_praw.__dict__)
    del cp_dict["_reddit"]
    cp_dict["author"] = cp_dict["author"].name
    cp_dict["subreddit"] = cp_dict["subreddit"].name
    cp_dict["parent_id"] = cp_dict["parent_id"][3:]
    return cp_dict


# random.shuffle(args.subreddit)
for subreddit in args.subreddit:

    # Since the api often only returns 1000, lets query in monthly intervals
    date_first = "2018-01-01"
    date_last = "2019-01-01"

    dates = pd.date_range(date_first, date_last, freq="3M")
    date_bins = list(zip(dates[:-1], dates[1:]))
    random.shuffle(date_bins)

    # First result is the stats, which we can use to get number of threads
    submissions = api.search_submissions(
        subreddit=subreddit,
        num_comments=">10",
        after=date_first,
        before=date_last,
        sort_type="num_comments",
        aggs="subreddit",
    )
    agg = next(submissions)
    if not agg["subreddit"]:
        logger.warning(f"No submissions within filter for subreddit found:{subreddit}")
        continue
    total_submissions = agg["subreddit"][0]["doc_count"]

    with tqdm(desc=subreddit, unit="submission", total=total_submissions) as prog:

        for after, before in date_bins:
            logger.debug(
                "%s",
                dict(
                    subreddit=subreddit,
                    num_comments=">10",
                    after=after,
                    before=before,
                    sort_type="num_comments",
                ),
            )

            submissions = api.search_submissions(
                subreddit=subreddit,
                num_comments=">10",
                after=after,
                before=before,
                sort_type="num_comments",
                aggs="subreddit",
            )
            # First result is the stats, which we can use to get number of threads
            agg = next(submissions)
            if not agg["subreddit"]:
                continue
            doc_count = agg["subreddit"][0]["doc_count"]

            out_dir = data_dir.joinpath(subreddit)
            os.makedirs(out_dir, exist_ok=True)
            if len(list(out_dir.glob("*.text"))) > args.number_of_threads:
                print(f"stopping at {args.number_of_threads} threads")
                break
            for submission in submissions:
                submission = psaw_to_dict(submission)
                submission_id = get_id_for_comments(submission)
                out_file = out_dir.joinpath(submission_id + ".pickle")

                if not out_file.is_file():
                    # Get comments
                    submission_comment_ids = api._get_submission_comment_ids(
                        submission["id"]
                    )
                    if len(submission_comment_ids) > 3000:
                        continue  # because it's too slow to parse these large trees with the current code
                    comment_dict = collections.defaultdict(list)

                    # Batch to avoid 414: Url too long
                    batch_size = 200
                    for i in range(0, len(submission_comment_ids), batch_size):
                        batch_ids = submission_comment_ids[i : i + batch_size]

                        # Use psaw
                        try:
                            comments = api.search_comments(ids=batch_ids)
                            # It will just repeat unless we set a limit
                            comments = [
                                next(comments)
                                for _ in tqdm(
                                    range(submission["num_comments"]),
                                    leave=False,
                                    unit="comment",
                                )
                            ]
                            # Or praw... nah slow
                            #             comments = [comment_praw2psaw(reddit.comment(id).refresh()) for id in submission_comment_ids]
                            for comment in comments:
                                comment = psaw_to_dict(comment)
                                comment_dict[comment["parent_id"]].append(comment)

                            # sort by karma, if available
                            for key in comment_dict.keys():
                                comment_dict[key].sort(
                                    key=lambda x: x.get("score", 0), reverse=True
                                )

                            # json it so we will have original data if wanted, that way we can make changes to input data formatting
                            out_jsn = out_dir.joinpath(submission_id + ".pickle")
                            pickle.dump(
                                dict(submission=submission, comment_dict=comment_dict),
                                out_jsn.open("wb"),
                            )
                            logger.debug("writing pickle %s", out_jsn)

                            # format
                            # text = format_comments_dict(comment_dict, submission)

                            # # write out thread
                            # out_file.write_text(text)
                        except Exception as e:
                            logger.warning(
                                f"Exception {e}, for subreddit={subreddit}, submission_id={submission['id']} submission_comment_ids={len(submission_comment_ids)} after={after} before={before}"
                            )
                            raise e
                        prog.update(1)
                else:
                    logger.debug("skipping existing file %s", out_file)
                    prog.update(1)
