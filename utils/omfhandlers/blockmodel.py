from pprint import pformat
import pandas as pd
import numpy as np


class BlockModelHandler:
    def __init__(self, bm) -> None:
        self.bm = bm
        self.name = bm.name
        self._bm_dataframe = self._bm_to_pandas_dataframe()

    def __str__(self):
        return "\nBlock model info:\n\n" + \
            f'Instance of {__class__.__name__}\n' + \
            pformat(self.get_bm_info, depth=3, indent=1, compact=True, width=250)

    @property
    def get_bm_attributes_name(self) -> list:
        return [attr.name for attr in self.bm.attributes]

    @property
    def get_bm_info(self):
        return self.bm.serialize()

    @property
    def get_bm_origin(self):
        return self.get_bm_info.get("center", self.bm.origin)

    def _bm_to_pandas_dataframe(self):
        origin = np.array(self.get_bm_origin)
        rows = []
        x_cumsum = np.cumsum(np.insert(self.bm.tensor_u, 0, 0))[:-1] + origin[0]
        y_cumsum = np.cumsum(np.insert(self.bm.tensor_v, 0, 0))[:-1] + origin[1]
        z_cumsum = np.cumsum(np.insert(self.bm.tensor_w, 0, 0))[:-1] + origin[2]
        for i, x in enumerate(self.bm.tensor_u):
            for j, y in enumerate(self.bm.tensor_v):
                for k, z in enumerate(self.bm.tensor_w):
                    rows.append({
                        'x_size': x, 'y_size': y, 'z_size': z,
                        'x_coord': x_cumsum[i], 'y_coord': y_cumsum[j], 'z_coord': z_cumsum[k],
                        'ijk_index': f'{i}-{j}-{k}',
                        'bench': str(int(z_cumsum[k]))
                    })

        df = pd.DataFrame(rows)

        for attribute in self._get_attribute_list:
            for key, value in attribute.items():
                df[key] = value.flatten(order="F")
        return df

    @property
    def get_outer_blocks_as_dataframe(self):
        return self._bm_dataframe[
            self._bm_dataframe.index.map(lambda idx: self._is_outer_block(idx, self._bm_dataframe))]

    @property
    def _get_attribute_list(self):
        bm_attribute_list = []
        for attribute in self.bm.attributes:
            bm_attribute_list.append({attribute.name: attribute.array.array})
        return bm_attribute_list

    @property
    def get_bm_dataframe(self):
        return self._bm_dataframe

    @property
    def get_bm_extends(self) -> dict:
        bm_extends = dict()
        bm_extends["min_x"] = self._bm_dataframe["x_coord"].min()
        bm_extends["max_x"] = self._bm_dataframe["x_coord"].max()
        bm_extends["min_y"] = self._bm_dataframe["y_coord"].min()
        bm_extends["mid_x"] = self._bm_dataframe["x_coord"].median()
        bm_extends["max_y"] = self._bm_dataframe["y_coord"].max()
        bm_extends["mid_y"] = self._bm_dataframe["y_coord"].median()
        bm_extends["min_z"] = self._bm_dataframe["z_coord"].min()
        bm_extends["max_z"] = self._bm_dataframe["z_coord"].max()
        bm_extends["mid_z"] = self._bm_dataframe["z_coord"].median()
        return bm_extends
