"""Json output plugin."""
import json


def to_json(self, filename, mode="w", date_format="%Y-%m-%dT%H:%M:%SZ"):
    """Write spectra in json format.

    Xarray's `to_dict` it used to dump dataset into dictionary to save as a json file.

    Args:
        - filename (str): name of output json file.
        - mode (str): file mode, by default `w` (create or overwrite).
        - date_format(str): strftime format for serializing datetimes.

    """
    dset_dict = self.to_dict()
    for item in ["coords", "data_vars"]:
        if "time" in dset_dict[item]:
            times = list(getattr(self, item)["time"].dt.strftime(date_format).values)
            dset_dict[item]["time"]["data"] = times
    with open(filename, mode=mode) as fp:
        json.dump(dset_dict, fp)
