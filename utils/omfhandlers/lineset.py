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

        lineset_df = pd.DataFrame()

        for attribute in self.lines.attributes:
            lineset_df[attribute.name] = attribute.array.array
        offset = 0
        for index in range(len(self.lines.vertices) - 1):

            if index < len(lineset_df) - 1:

                id = lineset_df.iloc[index][self.id_field]
                next_id = lineset_df.iloc[index + 1][self.id_field]
                if id != next_id:
                    offset += 1

                lineset_df.at[index, "x_start"] = self.lines.vertices[index + offset][0]
                lineset_df.at[index, "y_start"] = self.lines.vertices[index + offset][1]
                lineset_df.at[index, "z_start"] = self.lines.vertices[index + offset][2]
                lineset_df.at[index, "x_end"] = self.lines.vertices[index + 1 + offset][0]
                lineset_df.at[index, "y_end"] = self.lines.vertices[index + 1 + offset][1]
                lineset_df.at[index, "z_end"] = self.lines.vertices[index + 1 + offset][2]
        lineset_df.at[len(lineset_df) - 1, "x_start"] = self.lines.vertices[-2][0]
        lineset_df.at[len(lineset_df) - 1, "y_start"] = self.lines.vertices[-2][1]
        lineset_df.at[len(lineset_df) - 1, "z_start"] = self.lines.vertices[-2][2]
        lineset_df.at[len(lineset_df) - 1, "x_end"] = self.lines.vertices[-1][0]
        lineset_df.at[len(lineset_df) - 1, "y_end"] = self.lines.vertices[-1][1]
        lineset_df.at[len(lineset_df) - 1, "z_end"] = self.lines.vertices[-1][2]

        return lineset_df