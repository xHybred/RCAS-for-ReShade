// Copyright (C)2023 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files(the “Software”), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE 
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
// THE SOFTWARE.

//Tweaked by Hybred & originally implementation with additions by RdenBlaauwen
// Repository: https://github.com/xHybred/Hybred-Shaders
// Original https://github.com/RdenBlaauwen/RCAS-for-ReShade

#include "ReShadeUI.fxh"

#ifndef Experimental_Features
  #define ENABLE_NON_STANDARD_FEATURES 1
#endif

uniform float Sharpness <
  ui_type = "slider";
  ui_step = 0.05;
  ui_min = 0.0; ui_step = 0.01;
  #if ENABLE_NON_STANDARD_FEATURES
    ui_max = 1.30; 
  #else
    ui_max = 1.3;
  #endif
  ui_label = "Sharpness";
  ui_tooltip = "Sharpening strength.";
  ui_category = "RCAS";
> = 1.0;

#if ENABLE_NON_STANDARD_FEATURES == 1
  uniform float RCASLimit <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.1875; ui_step = 0.001;
    ui_label = "Limit";
    ui_tooltip = 
      "Limits how much pixels can be sharpened.\n"
      "Lower values reduce artifacts, but reduce sharpening.\n"
      "It's recommended to lower this value when using a\n"
      "very high (> 1.2) sharpness values.";
  > = 0.1875;

  uniform bool GreenAsLuma <
    ui_type = "slider";
    ui_label = "Use green as luma.";
    ui_tooltip =
      "Lets luma calculations use the green channel only.\n"
      "Improves performance, but may cause worse graphical fidelity.";
  > = false;
#endif

// RCAS also supports a define to enable a more expensive path to avoid some sharpening of noise.
// Would suggest it is better to apply film grain after RCAS sharpening (and after scaling) instead of using this define,
#ifndef Denoise
  #define Denoise 1
#endif

// RCAS sharpening supports a CAS-like pass-through alpha via the following
#ifndef RCAS_PASSTHROUGH_ALPHA
  #define RCAS_PASSTHROUGH_ALPHA 0
#endif

#include "ReShade.fxh"

#define RCAS_LUMA_WEIGHTS float3(0.5, 1.0, 0.5) // TODO: consider using float3(0.598, 1.174, 0.228)

#if ENABLE_NON_STANDARD_FEATURES == 1
  #define RCAS_LIMIT (RCASLimit)
#else
  // This is set at the limit of providing unnatural results for sharpening.
  #define RCAS_LIMIT (0.25 - (1.0 / 16.0))
#endif

texture ColorTex : COLOR;
sampler colorBufferLinear {
  Texture = ColorTex;
  SRGBTexture = true;
};

float getRCASLuma(float3 rgb)
{  
  #if ENABLE_NON_STANDARD_FEATURES
    if(GreenAsLuma){
      return rgb.g * 2.0;
    }

    return dot(rgb, RCAS_LUMA_WEIGHTS);
  #else
    return dot(rgb, RCAS_LUMA_WEIGHTS);
  #endif
}

// Based on https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/sdk/include/FidelityFX/gpu/fsr1/ffx_fsr1.h#L684
//==============================================================================================================================
//
//                                      FSR - [RCAS] ROBUST CONTRAST ADAPTIVE SHARPENING
//
//------------------------------------------------------------------------------------------------------------------------------
// CAS uses a simplified mechanism to convert local contrast into a variable amount of sharpness.
// RCAS uses a more exact mechanism, solving for the maximum local sharpness possible before clipping.
// RCAS also has a built in process to limit sharpening of what it detects as possible noise.
// RCAS sharper does not support scaling, as it should be applied after EASU scaling.
// Pass EASU output straight into RCAS, no color conversions necessary.
//------------------------------------------------------------------------------------------------------------------------------
// RCAS is based on the following logic.
// RCAS uses a 5 tap filter in a cross pattern (same as CAS),
//    w                n
//  w 1 w  for taps  w m e 
//    w                s
// Where 'w' is the negative lobe weight.
//  output = (w*(n+e+w+s)+m)/(4*w+1)
// RCAS solves for 'w' by seeing where the signal might clip out of the {0 to 1} input range,
//  0 == (w*(n+e+w+s)+m)/(4*w+1) -> w = -m/(n+e+w+s)
//  1 == (w*(n+e+w+s)+m)/(4*w+1) -> w = (1-m)/(n+e+w+s-4*1)
// Then chooses the 'w' which results in no clipping, limits 'w', and multiplies by the 'sharp' amount.
// This solution above has issues with MSAA input as the steps along the gradient cause edge detection issues.
// So RCAS uses 4x the maximum and 4x the minimum (depending on equation)in place of the individual taps.
// As well as switching from 'm' to either the minimum or maximum (depending on side), to help in energy conservation.
// This stabilizes RCAS.
// RCAS does a simple highpass which is normalized against the local contrast then shaped,
//       0.25
//  0.25  -1  0.25
//       0.25
// This is used as a noise detection filter, to reduce the effect of RCAS on grain, and focus on real edges.
float3 rcasPS(float4 vpos : SV_Position, float2 texcoord : TexCoord) : SV_Target
{
  // Algorithm uses minimal 3x3 pixel neighborhood.
  //    b
  //  d e f
  //    h
  #if RCAS_PASSTHROUGH_ALPHA
    float4 curr = tex2D(colorBufferLinear, texcoord).rgba;
    float3 e = curr.rgb;
    float alpha = curr.a;
  #else
    float3 e = tex2D(colorBufferLinear, texcoord).rgb;
  #endif

  float3 b = tex2Doffset(colorBufferLinear, texcoord, int2(0,-1)).rgb;
  float3 d = tex2Doffset(colorBufferLinear, texcoord, int2(-1,0)).rgb;
  float3 f = tex2Doffset(colorBufferLinear, texcoord, int2(1,0)).rgb;
  float3 h = tex2Doffset(colorBufferLinear, texcoord, int2(0,1)).rgb;

  // Get lumas times 2. Should use luma weights that are twice as large as normal.
  float bL = getRCASLuma(b);
  float dL = getRCASLuma(d);
  float eL = getRCASLuma(e);
  float fL = getRCASLuma(f);
  float hL = getRCASLuma(h);

  // Noise detection.
  #if RCAS_DENOISE == 1
    float nz = (bL + dL + fL + hL) * 0.25 - eL;
    float range = max(max(max(bL, dL), max(hL, fL)), eL) - min(min(min(bL, dL), min(eL, fL)), hL);
    nz = saturate(abs(nz) * rcp(range));
    nz = -0.5 * nz + 1.0;
  #endif

  // Min and max of ring.
  float3 minRGB = min(min(b, d), min(f, h));
  float3 maxRGB = max(max(b, d), max(f, h));
  // Immediate constants for peak range.
  float2 peakC = float2(1.0, -4.0);

  // Limiters, these need to use high precision reciprocal operations.
  // Decided to use standard rcp for now in hopes of optimizing it
  float3 hitMin = minRGB * rcp(4.0 * maxRGB);
  float3 hitMax = (peakC.xxx - maxRGB) * rcp(4.0 * minRGB + peakC.yyy);
  float3 lobeRGB = max(-hitMin, hitMax);
  float lobe = max(-RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * Sharpness;

  #if RCAS_DENOISE == 1
    // Apply noise removal.
    lobe *= nz;
  #endif

  // Resolve, which needs medium precision rcp approximation to avoid visible tonality changes.
  float rcpL = rcp(4.0 * lobe + 1.0);
  float3 output = ((b + d + f + h) * lobe + e) * rcpL;

  #if RCAS_PASSTHROUGH_ALPHA
    return float4(output.r, output.g, output.b, alpha);
  #else
    return output;
  #endif
}

technique RCAS 
  <
    ui_label = "RobustCAS";
    ui_tooltip = 
      "RCAS is a light-weight adaptive sharpening shader included in AMD FidelityFX FSR.\n"
      "It is a derivative of AMD FidelityFX CAS, but it \"uses a more exact mechanism, \n"
      "solving for the maximum local sharpness possible before clipping.\"\n"
      "It also lacks the support for scaling that AMD CAS has.\n"
      "\n"
      "The algorithm applies less sharpening to areas that are already sharp, while more\n"
      "featureless areas are sharpened more. This prevents artifacts like ugly contours.\n";
  >
{
  pass
  {
    VertexShader = PostProcessVS;
    PixelShader = rcasPS;
    SRGBWriteEnable = true; // TODO: test this on or off
  } 
}
