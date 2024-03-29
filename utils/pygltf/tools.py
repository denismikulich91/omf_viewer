import os
import json
import numpy as np
from . import gltf2 as gltf

ATTRIBUTE_BY_NAME = {
    "position": gltf.Attribute.POSITION,
    "normal": gltf.Attribute.NORMAL,
    "texCoord": gltf.Attribute.TEXCOORD,
    "texCoord0": gltf.Attribute.TEXCOORD_0,
    "texCoord1": gltf.Attribute.TEXCOORD_1,
    "color": gltf.Attribute.COLOR_0,
}

COMPONENT_TYPE_BY_DTYPE = {
    np.int8: gltf.ComponentType.BYTE,
    np.uint8: gltf.ComponentType.UNSIGNED_BYTE,
    np.int16: gltf.ComponentType.SHORT,
    np.uint16: gltf.ComponentType.UNSIGNED_SHORT,
    np.uint32: gltf.ComponentType.UNSIGNED_INT,
    np.float32: gltf.ComponentType.FLOAT,
}

ACCESSOR_TYPE_BY_SHAPE = {
    (): gltf.AccessorType.SCALAR,
    (1,): gltf.AccessorType.SCALAR,
    (2,): gltf.AccessorType.VEC2,
    (3,): gltf.AccessorType.VEC3,
    (4,): gltf.AccessorType.VEC4,
    (1, 1): gltf.AccessorType.SCALAR,
    (2, 2): gltf.AccessorType.MAT2,
    (3, 3): gltf.AccessorType.MAT3,
    (4, 4): gltf.AccessorType.MAT4,
}


def from_np_type(dtype, shape):
    accessorType = ACCESSOR_TYPE_BY_SHAPE.get(shape)
    componentType = COMPONENT_TYPE_BY_DTYPE.get(dtype.type)
    return accessorType, componentType


def subtype(dtype):
    try:
        dtype, shape = dtype.subdtype
        return dtype, shape
    except TypeError:
        dtype, shape = dtype, ()
        return dtype, shape


def generate_structured_array_accessors(data, buffer_number, offset=None, count=None, name=None):
    name = "{key}" if name is None else name
    count = len(data) if count is None else count
    
    result = {}
    for key, value in data.dtype.fields.items():
        dtype, delta = value
        dtype, shape = subtype(dtype)
        print(delta)
        accessorType, componentType = from_np_type(dtype, shape)
        accessor = gltf.Accessor(buffer_number, delta, count, accessorType, componentType, name=name.format(key=key))
        attribute = ATTRIBUTE_BY_NAME.get(key)
        if attribute == gltf.Attribute.POSITION:
            accessor.max = np.amax(data[key], axis=0).tolist()
            accessor.min = np.amin(data[key], axis=0).tolist()
        result[attribute] = accessor
    return result

def generate_array_accessor(data, buffer_number, offset=None, count=None, name=None):
    count = len(data) if count is None else count
    dtype, shape = data.dtype, data.shape
    accessorType, componentType = from_np_type(dtype, shape[1:])
    result = gltf.Accessor(buffer_number, offset, count, accessorType, componentType, name=name)
    return result



def generate_structured_array_buffer_views(data, buffer, target, offset=None, name=None):
    # name = "{key}" if name is None else name
    offset = 0 if offset is None else offset
    length = data.nbytes
    stride = data.itemsize
    result = {}
    key = "verticies"
    buffer_view = gltf.BufferView(buffer, offset, length, stride, target, name=name.format(key=key))
    result[key] = buffer_view
    print(result)
    return result


def generate_array_buffer_view(data, buffer, target, offset=None, name=None):
    offset = 0 if offset is None else offset
    length = data.nbytes
    stride = None
    result = gltf.BufferView(buffer, offset, length, stride, target, name=name)
    return result


def byteLength(buffers):
    for buffer in buffers:
        return sum(map(lambda buffer: buffer.nbytes, buffers))


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector


def cross_product(v1, v2):
    return (v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0])


def calculate_normals(vertices, indices):
    normals = np.zeros_like(vertices)

    for i in range(0, len(indices), 3):
        idx1, idx2, idx3 = indices[i], indices[i + 1], indices[i + 2]
        v1 = vertices[idx1]
        v2 = vertices[idx2]
        v3 = vertices[idx3]

        # Calculate vectors for two edges of the triangle
        edge1 = np.subtract(v2, v1)
        edge2 = np.subtract(v3, v1)

        # Calculate normal for the triangle
        normal = cross_product(edge1, edge2)
        normal = normalize_vector(normal)

        # Add the normal to each vertex of the triangle
        normals[idx1] += normal
        normals[idx2] += normal
        normals[idx3] += normal

    # Normalize the normals
    for i in range(len(normals)):
        normals[i] = normalize_vector(normals[i])

    return normals


def numpy_to_gltf(vertex_data, index_data, gltf_path, bin_path):
    mesh = gltf.Mesh([], name="Default Mesh")
    
    document = gltf.Document.from_mesh(mesh)
    buffers = [vertex_data, index_data]
    buffer = gltf.Buffer(byteLength(buffers), uri=os.path.relpath(bin_path, os.path.dirname(gltf_path)), name="Default Buffer")
    
    document.add_buffer(buffer)
    
    offset = 0
    vertex_buffer_views = generate_structured_array_buffer_views(vertex_data, buffer, gltf.BufferTarget.ARRAY_BUFFER, offset=offset, name="{key} Buffer View")
    offset += vertex_data.nbytes

    index_buffer_view = generate_array_buffer_view(index_data, buffer, gltf.BufferTarget.ELEMENT_ARRAY_BUFFER, offset=offset, name="Index Buffer View")
    offset += index_data.nbytes
    
    vertex_accessors = generate_structured_array_accessors(vertex_data, buffer_number=0, name="{key} Accessor")
    index_accessor = generate_array_accessor(index_data, buffer_number=1, name="Index Accessor")
    
    primitive = gltf.Primitive(vertex_accessors, index_accessor, None, gltf.PrimitiveMode.TRIANGLES)
    
    document.add_buffer_views(vertex_buffer_views.values())
    document.add_buffer_view(index_buffer_view)
    
    document.add_accessors(vertex_accessors.values())
    document.add_accessor(index_accessor)
    
    mesh.primitives.append(primitive)
    
    return document, buffers


def save(gltf_path, bin_path, document, buffers):
    data = document.togltf()
    with open(gltf_path, 'w') as f:
        json.dump(data, f, indent=2)

    with open(bin_path, 'wb') as f:
        for buffer in buffers:
            f.write(buffer.tobytes())

