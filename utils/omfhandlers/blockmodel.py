from pprint import pformat
import pandas as pd
import numpy as np
import utils.pygltf.tools as gltf


class BlockModelHandler:
    def __init__(self, bm, filter_condition=None, compact=False) -> None:
        self.bm = bm
        self.filter = filter_condition
        self.is_compact = compact
        self.name = bm.name
        self._bm_dataframe = self._bm_to_pandas_dataframe()

    def __str__(self):
        return "\nBlock model info:\n\n" + \
            f'Instance of {__class__.__name__}\n' + \
            pformat(self.get_bm_info, depth=3, indent=1, compact=True, width=250)

    @property
    def get_bm_attributes_list(self) -> list:
        return [attr.name for attr in self.bm.attributes]

    @property
    def get_bm_info(self):
        return self.bm.serialize()

    @property
    def get_bm_origin(self):
        return self.get_bm_info.get("center", self.bm.origin)

    def _prepare_gltf_data(self, x_size, y_size, z_size, x, y, z, indices_offset, grade):
        if not self.is_compact:
            buffer_array_size = 24
            # v6----------v5
            # /|          /|
            # v1---------v0|
            # | |        | |
            # | |v7------|-|v4
            # |/         |/
            # v2---------v3
            vertexes = [
                # v0-v1-v2-v3 front
                (x + x_size, y, z + z_size), (x, y, z + z_size), (x, y, z), (x + x_size, y, z),
                # v0-v3-v4-v5 right
                (x + x_size, y, z + z_size), (x + x_size, y, z), (x + x_size, y + y_size, z),
                (x + x_size, y + y_size, z + z_size),
                # v0-v5-v6-v1 up
                (x + x_size, y, z + z_size), (x + x_size, y + y_size, z + z_size), (x, y + y_size, z + z_size),
                (x, y, z + z_size),
                # v1-v6-v7-v2 left
                (x, y, z + z_size), (x, y + y_size, z + z_size), (x, y + y_size, z), (x, y, z),
                # v7-v4-v3-v2 down
                (x, y + y_size, z), (x + x_size, y + y_size, z), (x + x_size, y, z), (x, y, z),
                # v4-v7-v6-v5 back
                (x + x_size, y + y_size, z), (x, y + y_size, z), (x, y + y_size, z + z_size),
                (x + x_size, y + y_size, z + z_size),
            ]
            normals = [
                (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),  # v0-v1-v2-v3 front
                (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),  # v0-v3-v4-v5 right
                (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),  # v0-v5-v6-v1 up
                (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),  # v1-v6-v7-v2 left
                (0.0, -1.0, 0.0), (0.0, -1.0, 0.0), (0.0, -1.0, 0.0), (0.0, -1.0, 0.0),  # v7-v4-v3-v2 down
                (0.0, 0.0, -1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -1.0)  # v4-v7-v6-v5 back
            ]
            indexes = [
                0, 1, 2, 0, 2, 3,  # front
                4, 5, 6, 4, 6, 7,  # right
                8, 9, 10, 8, 10, 11,  # up
                12, 13, 14, 12, 14, 15,  # left
                16, 17, 18, 16, 18, 19,  # down
                20, 21, 22, 20, 22, 23  # back
            ]
        else:
            buffer_array_size = 8
            # v6----------v5
            # /|          /|
            # v1---------v0|
            # | |        | |
            # | |v7------|-|v4
            # |/         |/
            # v2---------v3
            vertexes = [
                (x, y, z),  # 0 Bottom-front-left
                (x + x_size, y, z),  # 1 Bottom-front-right
                (x, y + y_size, z),  # 2 Bottom-back-left
                (x + x_size, y + y_size, z),  # 3 Bottom-back-right
                (x, y, z + z_size),  # 4 Top-front-left
                (x + x_size, y, z + z_size),  # 5 Top-front-right
                (x, y + y_size, z + z_size),  # 6 Top-back-left
                (x + x_size, y + y_size, z + z_size),  # 7 Top-back-right
            ]
            normals = [
                (-1, -1, 1),  # 0 Bottom-front-left (average of bottom, front, left face normals)
                (1, -1, 1),   # 1 Bottom-front-right (average of bottom, front, right face normals)
                (-1, 1, 1),   # 2 Bottom-back-left (average of bottom, back, left face normals)
                (1, 1, 1),    # 3 Bottom-back-right (average of bottom, back, right face normals)
                (-1, -1, -1), # 4 Top-front-left (average of top, front, left face normals)
                (1, -1, -1),  # 5 Top-front-right (average of top, front, right face normals)
                (-1, 1, -1),  # 6 Top-back-left (average of top, back, left face normals)
                (1, 1, -1),   # 7 Top-back-right (average of top, back, right face normals)
            ]
            indexes = [
                # Front face
                0, 1, 2,  # First triangle (bottom-left, bottom-right, top-left)
                2, 1, 3,  # Second triangle (top-left, bottom-right, top-right)

                # Back face
                5, 4, 7,  # First triangle (bottom-right, bottom-left, top-right)
                7, 4, 6,  # Second triangle (top-right, bottom-left, top-left)

                # Top face
                2, 3, 6,  # First triangle (top-back-left, top-back-right, top-front-left)
                6, 3, 7,  # Second triangle (top-front-left, top-back-right, top-front-right)

                # Bottom face
                1, 0, 5,  # First triangle (bottom-back-right, bottom-back-left, bottom-front-right)
                5, 0, 4,  # Second triangle (bottom-front-right, bottom-back-left, bottom-front-left)

                # Left face
                4, 0, 6,  # First triangle (bottom-front-left, bottom-back-left, top-front-left)
                6, 0, 2,  # Second triangle (top-front-left, bottom-back-left, top-back-left)

                # Right face
                1, 5, 3,  # First triangle (bottom-back-right, bottom-front-right, top-back-right)
                3, 5, 7,  # Second triangle (top-back-right, bottom-front-right, top-front-right)
            ]

        vertex_data = np.zeros(buffer_array_size, dtype=[
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("color", np.float32, 4),
        ])

        vertex_data["position"] = vertexes
        vertex_data["normal"] = normals
        index_data = np.array(indexes, dtype=np.uint16) + indices_offset * buffer_array_size
        color = self.set_color(grade)
        vertex_data["color"] = [color] * buffer_array_size
        vertex_data["normal"] = [gltf.normalize_vector(np.array(normal)) for normal in vertex_data["normal"]]

        return vertex_data, index_data

    def create_gltf_from_dataframe(self, location):
        vertex_data_list = []
        index_data_list = []
        indices_offset = 0

        for index, block in self.get_bm_dataframe.iterrows():
            vertex_data, index_data = self._prepare_gltf_data(block['x_size'], block['y_size'], block['z_size'],
                                                              block['x_coord'],block['y_coord'], block['z_coord'],
                                                              indices_offset, block['CU_pct'])
            indices_offset += 1

            vertex_data_list.append(vertex_data)
            index_data_list.append(index_data)

        # Concatenate data
        final_vertex_data = np.concatenate(vertex_data_list)
        final_index_data = np.concatenate(index_data_list)
        gltf_path = f"{location}.gltf"
        bin_path = f"{location}.bin"

        document, buffers = gltf.numpy_to_gltf(final_vertex_data,
                                               final_index_data,
                                               gltf_path,
                                               bin_path,
                                               "TRIANGLES")

        gltf.save(gltf_path, bin_path, document, buffers)

    def set_color(self, grade):
        if grade < 2.5:
            color = (0, 0, 1, 1)
        elif grade < 3:
            color = (0, 0.5, 1, 1)
        elif grade < 3.5:
            color = (0, 1, 0.5, 1)
        elif grade < 4:
            color = (1, 1, 0, 1)
        elif grade < 4.5:
            color = (1, 0.5, 0, 1)
        else:
            color = (1, 0, 0, 1)
        return color

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

        if self.filter:
            return df.query(self._filter_query_string)
        return df

    @property
    def _filter_query_string(self):
        # example: {'CU_pct': ['>=', 2.4], 'rocktype': ['!=', 'air']}
        query_string = ""
        for i, (column, condition) in enumerate(self.filter.items()):
            operator = condition[0]
            value = condition[1]
            if i > 0:
                query_string += " and "
            query_string += f"{column} {operator} {value}"
        return query_string

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
