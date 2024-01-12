from pprint import pformat
import numpy as np
import utils.pygltf.tools as gltf


class SurfaceHandler:
    def __init__(self, surface) -> None:
        self.surface = surface
        self.name = surface.name
        self.geometry = {"vertices": self.surface.vertices.array,
                         "triangles": self.surface.triangles.array}

    def __str__(self):
        return "\nSurface info:\n\n" + \
            f'Instance of {__class__.__name__}\n' + \
            pformat(self.get_surface_info, depth=3, indent=1, compact=True, width=250)

    @property
    def get_surface_attributes_list(self) -> list:
        return [attr.name for attr in self.surface.attributes]

    @property
    def get_surface_info(self):
        return self.surface.serialize()

    @property
    def get_surface_origin(self):
        return self.get_surface_info.get("center", self.surface.origin)

    def _prepare_gltf_data(self):
        vertexes = self.geometry["vertices"]
        indexes = self.geometry["triangles"].ravel()

        buffer_array_size = len(vertexes)
        normals = gltf.calculate_normals(vertexes, indexes)

        vertex_data = np.zeros(buffer_array_size, dtype=[
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("color", np.float32, 4),
        ])

        vertex_data["position"] = vertexes
        vertex_data["normal"] = normals
        index_data = np.array(indexes, dtype=np.uint16)
        # color = self.set_color(color_attribute)
        # TODO: set up coloring
        vertex_data["color"] = [(1, 1, 0, 1)] * buffer_array_size
        vertex_data["normal"] = [gltf.normalize_vector(np.array(normal)) for normal in vertex_data["normal"]]

        return vertex_data, index_data

    # def _prepare_gltf_data(self):
    #     # Extracting vertex positions, normals, and indices
    #     vertexes = self.geometry["vertices"]
    #     indexes = self.geometry["triangles"].ravel()
    #
    #     # Calculating normals
    #     normals = gltf.calculate_normals(vertexes, indexes)
    #
    #     # Creating separate arrays for positions, normals, and colors
    #     position_data = np.array(vertexes, dtype=np.float32)
    #     normal_data = np.array([gltf.normalize_vector(np.array(normal)) for normal in normals], dtype=np.float32)
    #     color_data = np.array([(1, 1, 0, 1)] * len(vertexes), dtype=np.float32)
    #     # Flattening the arrays and concatenating them
    #     # This creates a single buffer with non-interleaved data
    #     vertex_data = [position_data, normal_data, color_data]
    #     # Creating index data
    #     index_data = np.array(indexes, dtype=np.uint16)
    #     return vertex_data, index_data

    def create_gltf_from_dataset(self, location):

        vertex_data, index_data = self._prepare_gltf_data()

        gltf_path = f"{location}.gltf"
        bin_path = f"{location}.bin"

        document, buffers = gltf.numpy_to_gltf(vertex_data, index_data, gltf_path, bin_path)

        gltf.save(gltf_path, bin_path, document, buffers)

    @property
    def get_surface_extends(self) -> dict:

        surface_extends = dict()
        surface_extends["min_x"] = np.min(self.geometry["vertices"][:, 0])
        surface_extends["max_x"] = np.max(self.geometry["vertices"][:, 0])
        surface_extends["middle_x"] = (surface_extends["min_x"] + surface_extends["max_x"]) / 2

        surface_extends["min_y"] = np.min(self.geometry["vertices"][:, 1])
        surface_extends["max_y"] = np.max(self.geometry["vertices"][:, 1])
        surface_extends["middle_y"] = (surface_extends["min_y"] + surface_extends["max_y"]) / 2

        surface_extends["min_z"] = np.min(self.geometry["vertices"][:, 2])
        surface_extends["max_z"] = np.max(self.geometry["vertices"][:, 2])
        surface_extends["middle_z"] = (surface_extends["min_z"] + surface_extends["max_z"]) / 2

        return surface_extends
