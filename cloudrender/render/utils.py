import torch
from torch.nn import Module

class MeshNorms(Module):
    def __init__(self, faces: torch.Tensor):
        super().__init__()
        self.faces_count = faces.size(0)
        normmap = self.compute_face2verts_normmap(faces)
        self.register_buffer("normmap", normmap)
        self.register_buffer("faces", faces)

    @staticmethod
    def compute_face2verts_normmap(faces: torch.Tensor):
        _, faces_in_vertices_count = torch.unique(faces, sorted=True, return_counts=True)
        verts_count = len(faces_in_vertices_count)
        faces_in_vertex_max = faces_in_vertices_count.max().item()
        faces_appearance = torch.argsort(faces.view(-1)) // 3
        appearance_array_off = 0
        # print(faces.size())
        normmap = torch.ones(verts_count, faces_in_vertex_max, dtype=torch.long, device=faces.device) * faces.size(0)
        for i in range(verts_count):
            faces_in_vertex = faces_in_vertices_count[i]
            normmap[i, :faces_in_vertex] = faces_appearance[appearance_array_off:appearance_array_off + faces_in_vertex]
            appearance_array_off += faces_in_vertex
        return normmap

    def faces_norms(self, verts: torch.Tensor):
        verts_size = verts.size()
        verts_faces = verts[..., self.faces.view(-1), :].view(*verts_size[:-2], -1, 3, 3)
        vct1 = verts_faces[..., 0, :] - verts_faces[..., 1, :]
        vct2 = verts_faces[..., 0, :] - verts_faces[..., 2, :]
        cross = torch.cross(vct1, vct2, dim=-1)
        faces_norms = cross / torch.norm(cross, dim=-1, keepdim=True)
        return faces_norms

    def vertices_norms(self, verts: torch.Tensor):
        faces_norms = self.faces_norms(verts)
        faces_norms = torch.cat([faces_norms, torch.zeros(*faces_norms.size()[:-2], 1, 3, device=verts.device)], dim=-2)
        vertices_norms = faces_norms[..., self.normmap.view(-1), :].view(*faces_norms.size()[:-2], -1,
                                                                         self.normmap.size(1), 3).sum(dim=-2)
        vertices_norms = vertices_norms / torch.norm(vertices_norms, dim=-1, keepdim=True)
        return vertices_norms