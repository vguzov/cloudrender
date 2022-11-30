#version 400
#define SHADOWMAPS_MAX 6
#define SHADOWDEPTH_EPS 5e-3

struct DirLight {
    vec3 direction;
    vec3 intensity;
};

uniform DirLight dirlight;
uniform float diffuse;
uniform float ambient;
uniform float specular;
uniform float shininess;
uniform sampler2D shadowmaps[SHADOWMAPS_MAX];
uniform bool shadowmap_enabled[SHADOWMAPS_MAX];
uniform vec4 shadow_color;
uniform mat4 V;
uniform vec4 overlay_color;

uniform sampler2D meshTexture;

in VS_OUT {
	vec3 pose;
    float depth;
	vec3 normal;
	vec3 MVnormal;
	vec4 pose_shadow[SHADOWMAPS_MAX];
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
    vec3 frag_diffuse  = vec3(light.intensity  * diff * diffuse);
    vec3 frag_specular = vec3(light.intensity * spec * specular);
	vec3 color_sum = frag_ambient + frag_diffuse;
    return vec4(color_sum, 1.)*color + vec4(frag_specular, 0.);
}

bool shadow_calculation(sampler2D shadowmap, vec4 pose_shadow) {
	vec3 projected_shadow = pose_shadow.xyz/pose_shadow.w;
	projected_shadow = projected_shadow*0.5+0.5;
	float closest_depth = texture(shadowmap, projected_shadow.xy).r;
	float current_depth = projected_shadow.z;
	bool shadow = (projected_shadow.x>=0 && projected_shadow.x<=1 &&projected_shadow.y >=0 && projected_shadow.y<=1 &&
	projected_shadow.z<1) && (current_depth-SHADOWDEPTH_EPS > closest_depth) && (current_depth < closest_depth+0.5);
	return shadow;
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

	for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i) {
		bool is_shadow = shadow_calculation(shadowmaps[i], fs_in.pose_shadow[i]);
		color = is_shadow ? vec4(shadow_color.rgb*shadow_color.a+color.rgb*(1-shadow_color.a), shadow_color.a + color.a*(1-shadow_color.a)):color;
	}
	out_color = color;
}