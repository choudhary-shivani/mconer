import glob


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e, filepath) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent, fp):
        # print(tfevent.summary.value, fp)
        return dict(
            wall_time=tfevent.wall_time,
            # name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            name = fp.split('\\')[-2],
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ['wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        # print(root, filenames)
        for filename in filenames:
            # print(filename)
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


if __name__ == "__main__":
    dir_path = r"C:\Users\Rah12937\PycharmProjects\mconer\old run"
    import os
    os.chdir(dir_path)
    for file in glob.glob(r'runid_*'):

        if not file.endswith('.csv'):
            print(file)
            exp_name = f"{file}"
            df = convert_tb_data(f"{dir_path}/{exp_name}")
            df.to_csv(f'{exp_name}_metric.csv', index=False)
    # print(df)