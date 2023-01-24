from santa_2022.common import *
import pandas as pd
import matplotlib.pyplot as plt

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


def save_submission(path, name):
    if name is None:
        raise ValueError("Please enter a filename")
    submission = pd.Series(
        [config_to_string(config) for config in path],
        name="configuration",
    )
    file_name = f"../../output/submissions/{name}.csv"
    if os.path.isfile(file_name):
        print(f'File {file_name} already exists')
        return
    submission.to_csv(file_name, index=False)


def save_descriptive_stats(df, name):
    if name is None:
        raise ValueError("Please enter a filename")
    file_name = f"../../output/info/{name}.csv"
    if os.path.isfile(file_name):
        print(f'File {file_name} already exists')
        return
    df.to_csv(file_name, index=False)


def deduplicate_latest_path():
    df_image = pd.read_csv("../../data/image.csv")
    image = df_to_image(df_image)

    latest, submission_directory = get_latest_submission()
    csv = os.path.join(submission_directory, latest)

    df = pd.read_csv(csv).astype("string")
    path = [string_to_config(row) for row in df['configuration']]
    print(f'Total cost: {total_cost(path, image)}')

    deduped_path = run_remove(path)
    print(f'Total cost: {total_cost(deduped_path, image)}')
    save_submission(deduped_path, latest[:-3] + ' deduped')


def get_latest_submission(dedup=False):
    submission_directory = '../../output/submissions'
    files = os.listdir(submission_directory)
    if not dedup:
        files = filter(lambda x: 'deduped' not in x, files)
    latest = max(files,
                 key=lambda x: os.path.getmtime(os.path.join(submission_directory, x)))
    return latest, submission_directory


def get_submission_arrow_file(latest):
    return f"../../output/arrows/{latest}"


def check_positions(path):
    n = 128
    points = list(product(range(-n, n + 1), repeat=2))
    points = set(points)

    path = set([get_position(c) for c in path])
    try:
        assert points == path
    except AssertionError:
        return points - path


def path_to_df(path, image):
    position = [get_position(config) for config in path]
    reconf_costs = [None] + [reconfiguration_cost(config, next_config)
                             for config, next_config
                             in zip(path, path[1:])]
    step_costs = [None] + [step_cost(config, next_config, image)
                           for config, next_config
                           in zip(path, path[1:])]
    color_costs = [None] + [color_cost_from_config(config, next_config, image)
                            for config, next_config
                            in zip(path, path[1:])]
    return pd.DataFrame({"config": path,
                         "position": position,
                         "reconf_cost": reconf_costs,
                         "color_cost": color_costs,
                         "step_cost": step_costs,
                         })


def add_costs(df, image):
    path = submission_to_path(df)
    df['reconf_cost'] = [reconfiguration_cost(config, next_config)
                         for config, next_config
                         in zip(path, path[1:])] + [pd.NA]
    df['color_cost'] = [color_cost_from_config(config, next_config, image)
                        for config, next_config
                        in zip(path, path[1:])] + [pd.NA]
    df['step_cost'] = [step_cost(config, next_config, image)
                       for config, next_config
                       in zip(path, path[1:])] + [pd.NA]
    df['reconf_frac'] = df['reconf_cost'] / df['step_cost']
    df['color_frac'] = df['color_cost'] / df['step_cost']
    return df


def string_to_config(row_string):
    a = (pair.split() for pair in row_string.split(";"))
    b = ([int(x), int(y)] for (x, y) in a)
    return list(b)


def submission_to_path(submission):
    return [string_to_config(row) for row in submission['configuration']]


def path_to_arrows(path):
    data = []
    for conf, next_conf in zip(path, path[1:]):
        x, y = get_position(conf)
        u, v = get_position(next_conf)
        dx = u - x
        dy = v - y
        data.append([x, y, dx, dy, 'not_given', pd.NA])
    x, y = get_position(path[-1])
    data.append([x, y, 0, 0, 'not_given', pd.NA])
    assert len(data) == len(path)
    return pd.DataFrame(data=data,
                        columns=['x', 'y', 'dx', 'dy', 'move_type', 'slow_counter'])


def plot_path_over_image(config, info_df, save_path=None, image=None, ax=None,
                         **figure_args):
    if os.path.isfile(save_path):
        print(f'Image {save_path} already exists')
        return

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), **figure_args)

    k = 2 ** (len(config) - 1) + 1
    x = info_df.loc[:, 'x'].to_numpy()
    y = info_df.loc[:, 'y'].to_numpy()
    dx = info_df.loc[:, 'dx'].to_numpy()
    dy = info_df.loc[:, 'dy'].to_numpy()

    if (info_df.loc[:, 'move_type'] == 'not_given').all():
        color = plt.cm.plasma(np.linspace(0, 1, len(x)))
    else:
        replace_dict = {
            'down': 'b',
            'up': 'b',
            'down127': 'b',
            'cheapest': 'g',
            'slow': 'r',
            'return_to_origin': 'k',
            'origin': 'k',
            'connect_ends': 'm',
            'corner': 'm',
        }

        color = info_df.loc[:, 'move_type']
        color = color.replace(replace_dict)
        color = color.to_numpy()

    ax.quiver(
        x, y, dx, dy,
        color=color,
        angles='xy', scale_units='xy', scale=1,
        alpha=0.5,
        width=0.0005,
    )
    l = k - 1 + 0.5
    if image is not None:
        ax.matshow(image, extent=[-l, l, -l, l])
    ax.set_xlim(-l-1, l+1)
    ax.set_ylim(-l-1, l+1)
    ax.set_aspect('equal')
    if save_path:
        plt.savefig(save_path,
                    dpi=1000,
                    # bbox_inches='tight',
                    # pad_inches=0.,
                    )
    else:
        plt.show()
    return ax
