from pprint import pformat
import numpy as np
import utils.pygltf.tools as gltf
import pandas as pd


class LinesHandler:
    def __init__(self, lines, id_field) -> None:
        self.lines = lines
        self.name = lines.name
        self.id_field = id_field

    def lineset_to_pandas_dataframe(self):
        # print(len(end_points))
        #     'start_x': [point[0] for point in start_points],
        #     'end_x': [point[0] for point in end_points],
        #     'start_y': [point[1] for point in start_points],
        #     'end_y': [point[1] for point in end_points],
        #     'start_z': [point[2] for point in start_points],
        #     'end_z': [point[2] for point in end_points],

        lineset_df = pd.DataFrame()

        for attribute in self.lines.attributes:
            lineset_df[attribute.name] = attribute.array.array

        for index, row in lineset_df.iterrows():
            if index < len(lineset_df) - 1:
                id = row[self.id_field]
                next_id = lineset_df.iloc[index + 1][self.id_field]
                if id == next_id:
                    lineset_df.at[index, "x_start"] = self.lines.vertices[index][0]
                    lineset_df.at[index, "y_start"] = self.lines.vertices[index][1]
                    lineset_df.at[index, "z_start"] = self.lines.vertices[index][2]
                    lineset_df.at[index, "x_end"] = self.lines.vertices[index + 1][0]
                    lineset_df.at[index, "y_end"] = self.lines.vertices[index + 1][1]
                    lineset_df.at[index, "z_end"] = self.lines.vertices[index + 1][2]

        return lineset_df