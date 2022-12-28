from santa_2022.common import *

import pandas as pd

import os


def find_duplicate_points(path):
    duplicate_points = {}
    for c in path:
        p = get_position(c)
        if p != (0, 0):
            duplicate_points[p] = duplicate_points.get(p, 0) + 1
    return duplicate_points


def vector_diff_one(path):
    for i in range(len(path) - 1):
        for c0, c1 in zip(path[i], path[i + 1]):
            if abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]) > 1:
                #                 print(path[i])
                #                 print(path[i+1])
                return False
    return True


def run_remove(path):
    print("-- run remove --")
    print(f"Current length: {len(path)}")
    duplicate_points = find_duplicate_points(path)
    i = len(path) - 2
    while i >= 0:
        local_p = path[i:i + 3]
        p = get_position(local_p[1])
        new_local_p = compress_path(local_p)
        if vector_diff_one(new_local_p) and duplicate_points.get(p, 0) > 1 and len(
                new_local_p) < 3:
            path = path[:i + 1] + path[i + 2:]
            duplicate_points[p] -= 1
        i -= 1
    print(f"New length: {len(path)}")
    return path


def generate_submission(path, name):
    if name is None:
        raise ValueError("Please enter a filename")
    submission = pd.Series(
        [config_to_string(config) for config in path],
        name="configuration",
    )
    submission.head()
    submission.to_csv(f"../../output/submissions/{name}.csv", index=False)


def main():
    df_image = pd.read_csv("../../data/image.csv")
    image = df_to_image(df_image)

    submission_directory = '../../output/submissions'
    files = os.listdir(submission_directory)
    latest = max(files,
                 key=lambda x: os.path.getmtime(os.path.join(submission_directory, x)))
    csv = os.path.join(submission_directory, latest)

    df = pd.read_csv(csv).astype("string")
    path = [string_to_config(row) for row in df['configuration']]
    print(f'Total cost: {total_cost(path, image)}')

    deduped_path = run_remove(path)
    print(f'Total cost: {total_cost(deduped_path, image)}')
    generate_submission(deduped_path, latest[:-3] + ' deduped')


if __name__ == '__main__':
    main()
