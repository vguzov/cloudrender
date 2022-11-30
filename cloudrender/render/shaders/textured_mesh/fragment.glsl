#version 330 core

struct DirLight {
    vec3 direction;
    vec3 intensity;
};

uniform DirLight dirlight;
uniform float diffuse;
uniform float ambient;
uniform float specular;
uniform float shininess;
uniform vec4 shadow_color;
uniform mat4 V;
uniform vec4 overlay_color;

uniform sampler2D meshTexture;

in VS_OUT {
	vec3 pose;
    float depth;
	vec3 normal;
	vec3 MVnormal;
    vec2 texUV;
} fs_in;

layout(location = 0) out vec4 out_color;

vec4 dirlight_calculation(DirLight light, vec4 color, vec3 normal, vec3 view_dir)
{
    vec3 light_dir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(normal, light_dir), 0.0);
    // specular shading
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = shininess > 0 ? pow(max(dot(view_dir, reflect_dir), 0.0), shininess) : 1.0;
    // combine results
    vec3 frag_ambient  = vec3(ambient);
    vec3 frag_diffuse  = vec3(light.intensity * diff * diffuse);
    vec3 frag_specular = vec3(light.intensity * spec * specular);
	vec3 color_sum = frag_ambient + frag_diffuse;
    return vec4(color_sum, 1.)*color + vec4(frag_specular, 0.);
}

vec4 alpha_blending(vec4 orig_color, vec4 overlay_color)
{
    float res_alpha = overlay_color.a + orig_color.a*(1-overlay_color.a);
	return vec4((res_alpha==0.0)?orig_color.rgb:((overlay_color.rgb*overlay_color.a+orig_color.rgb*(1-overlay_color.a))/res_alpha), res_alpha);
}

void main() {
	vec3 camera_position = transpose(V)[3].xyz;
	vec3 view_dir = normalize(camera_position - fs_in.pose);
    vec4 tex_color = texture(meshTexture, fs_in.texUV);
    tex_color = alpha_blending(tex_color, overlay_color);
	vec4 color = dirlight_calculation(dirlight, tex_color, fs_in.normal, view_dir);
	out_color = color;
}