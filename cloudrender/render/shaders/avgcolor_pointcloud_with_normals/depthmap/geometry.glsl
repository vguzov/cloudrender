#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform float width_mul;
uniform float splat_size;
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
//uniform float depth_offset;

in VS_OUT {
    vec3 norm;
    vec4 poseMV;
} gs_in[];

void main() {
    mat4 MVP = P*V*M;
//    vec4 depth_offset_perspective = P*vec4(0,0,depth_offset,0);
    vec4 vertexPosMV = gs_in[0].poseMV;
    vec3 norm = normalize(gs_in[0].norm);
    vec4 position = gl_in[0].gl_Position;
    float base_size = splat_size*0.015;
    vec3 starting_cross_vct = abs(norm[0]-1)<1e-5 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 splat_plane_vct1 = cross(norm, starting_cross_vct);
    vec3 splat_plane_vct2 = cross(norm, splat_plane_vct1);
    vec4 width_offset = MVP*vec4(base_size*splat_plane_vct1,0);
    vec4 height_offset = MVP*vec4(base_size*splat_plane_vct2,0);

    gl_Position = position - width_offset - height_offset; //+ depth_offset_perspective;
    EmitVertex();

    gl_Position = position + width_offset - height_offset; //+ depth_offset_perspective;
    EmitVertex();

    gl_Position = position - width_offset + height_offset; //+ depth_offset_perspective;
    EmitVertex();

    gl_Position = position + width_offset + height_offset; //+ depth_offset_perspective;
    EmitVertex();

    EndPrimitive();
}
