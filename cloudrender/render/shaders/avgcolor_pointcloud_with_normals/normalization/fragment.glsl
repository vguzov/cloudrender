#version 330 core

layout(location = 0) out vec4 color;

uniform sampler2D pixColors;
uniform vec2 resolution;
//uniform sampler2D pixWeights;

void main() {
	vec4 sumcolor = texture(pixColors, gl_FragCoord.xy/resolution);
	color = vec4(sumcolor.rgb/max(1., sumcolor.a), 1.);
//	float cval = sumcolor.a/10.;
//	color = vec4(cval,cval,cval,1.);
//	color = texture(pixColors, gl_FragCoord.xy)/texture(pixWeights, gl_FragCoord.xy);
}