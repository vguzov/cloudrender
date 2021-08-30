#version 330 core
#define SHADOWMAPS_MAX 6
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform float width_mul;
uniform float splat_size;

in VS_OUT {
    vec4 color;
    int inst_id;
    float depth;
//    vec3 normal;
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
    mat4 invMVP = inverse(P*V*M);
    vec4 position = gl_in[0].gl_Position;
    float size_mul = splat_size/(0.1+0.4*gs_in[0].depth)*position.w;
    float color_mul = 1;
    vcolor = vec4(gs_in[0].color.rgb*color_mul+(1-color_mul), gs_in[0].color.a);
//    vnormal = gs_in[0].normal;
    frag_inst_id = gs_in[0].inst_id;

    gl_Position = position + vec4(-0.01*width_mul, -0.01, 0.0, 0.0)*size_mul;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    gl_Position = position + vec4(0.01*width_mul, -0.01, 0.0, 0.0)*size_mul;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    gl_Position = position + vec4(-0.01*width_mul, 0.01, 0.0, 0.0)*size_mul;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    gl_Position = position + vec4(0.01*width_mul, 0.01, 0.0, 0.0)*size_mul;
    for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
        pose_shadow[i] = shadowmap_MVP[i]*invMVP*gl_Position;
    EmitVertex();

    EndPrimitive();
}
