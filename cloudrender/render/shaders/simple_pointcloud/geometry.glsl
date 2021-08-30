#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform float width_mul;
uniform float splat_size;

in VS_OUT {
    vec4 color;
    int inst_id;
    float depth;
} gs_in[];

out vec4 vcolor;
flat out int frag_inst_id;

void main() {
    vec4 position = gl_in[0].gl_Position;
    float size_mul = splat_size/(0.1+0.4*gs_in[0].depth)*position.w;
    float color_mul = 1;
    vcolor = vec4(gs_in[0].color.rgb*color_mul+(1-color_mul), gs_in[0].color.a);
    frag_inst_id = gs_in[0].inst_id;

    gl_Position = position + vec4(-0.01*width_mul, -0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    gl_Position = position + vec4(0.01*width_mul, -0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    gl_Position = position + vec4(-0.01*width_mul, 0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    gl_Position = position + vec4(0.01*width_mul, 0.01, 0.0, 0.0)*size_mul;
    EmitVertex();

    EndPrimitive();
}
