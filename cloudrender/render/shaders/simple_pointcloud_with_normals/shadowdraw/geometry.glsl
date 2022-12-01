#version 330 core
#define SHADOWMAPS_MAX 6
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform float width_mul;
uniform float splat_size;

in VS_OUT {
    vec4 color;
	int inst_id;
	vec3 norm;
	vec4 poseMV;
	vec4 pose_shadow[SHADOWMAPS_MAX];
} gs_in[];

out vec4 vcolor;
flat out int frag_inst_id;
out vec4 pose_shadow[SHADOWMAPS_MAX];
//out vec3 vnormal;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform bool shadowmap_enabled[SHADOWMAPS_MAX];
uniform mat4 shadowmap_MVP[SHADOWMAPS_MAX];

void main() {
    mat4 MVP = P*V*M;
    vec4 vertexPosMV = gs_in[0].poseMV;
    vec3 norm = normalize(gs_in[0].norm);
    vec4 position = gl_in[0].gl_Position;
    float base_size = splat_size*0.015;
    vec3 starting_cross_vct = abs(norm[0]-1)<1e-5 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 splat_plane_vct1 = cross(norm, starting_cross_vct);
    vec3 splat_plane_vct2 = cross(norm, splat_plane_vct1);
    vec4 width_offset = MVP*vec4(base_size*splat_plane_vct1,0);
    vec4 height_offset = MVP*vec4(base_size*splat_plane_vct2,0);
    mat4 invMVP = inverse(MVP);
    float color_mul = 1;
    vcolor = vec4(gs_in[0].color.rgb*color_mul+(1-color_mul), gs_in[0].color.a);
    frag_inst_id = gs_in[0].inst_id;

    gl_Position = position - width_offset - height_offset;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    gl_Position = position + width_offset - height_offset;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    gl_Position = position - width_offset + height_offset;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    gl_Position = position + width_offset + height_offset;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    EndPrimitive();
}
