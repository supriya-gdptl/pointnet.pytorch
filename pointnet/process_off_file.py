"""
Code borrowed from https://github.com/autonomousvision/occupancy_networks/blob/b57f6cce5c8aa0e7a4db73b8e23b481394aa6e6f/im2mesh/utils/io.py
"""
import os
import numpy as np


def read_off(input_off_file):
    """
    Reads vertices and faces from an off file.

    :param input_off_file: path to file to read
    :type input_off_file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(input_off_file), 'file %s not found' % input_off_file

    with open(input_off_file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces
        # are  all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', \
                   'invalid OFF file %s' % input_off_file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', \
                   'invalid OFF file %s' % input_off_file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', \
                      'found empty vertex index: %s (%s)' \
                      % (lines[start_index + num_vertices + i], input_off_file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, \
                'face should have %d vertices but as %d (%s)' \
                % (face[0], len(face) - 1, input_off_file)
            assert face[0] == 3, \
                'only triangular meshes supported (%s)' % input_off_file
            for index in face:
                assert index >= 0 and index < num_vertices, \
                    'vertex %d (of %d vertices) does not exist (%s)' \
                    % (index, num_vertices, input_off_file)

            assert len(face) > 1

            faces.append(face)
        vertices = np.array(vertices)
        faces = np.array(faces)
        return vertices, faces


def test_read_off_file():
    off_file = '../../../../modelnet40/ModelNet40/sofa/sofa_0001.off'
    verts, faces = read_off(input_off_file=off_file)
    print(verts.shape)
    print(faces.shape)
