
// Generated by Cadence Genus(TM) Synthesis Solution 16.22-s033_1
// Generated on: May  7 2020 12:14:14 EDT (May  7 2020 16:14:14 UTC)

// Verification Directory fv/c880

module c880g(N1, N8, N13, N17, N26, N29, N36, N42, N51, N55, N59, N68,
     N72, N73, N74, N75, N80, N85, N86, N87, N88, N89, N90, N91, N96,
     N101, N106, N111, N116, N121, N126, N130, N135, N138, N143, N146,
     N149, N152, N153, N156, N159, N165, N171, N177, N183, N189, N195,
     N201, N207, N210, N219, N228, N237, N246, N255, N259, N260, N261,
     N267, N268, N388, N389, N390, N391, N418, N419, N420, N421, N422,
     N423, N446, N447, N448, N449, N450, N767, N768, N850, N863, N864,
     N865, N866, N874, N878, N879, N880);
  input N1, N8, N13, N17, N26, N29, N36, N42, N51, N55, N59, N68, N72,
       N73, N74, N75, N80, N85, N86, N87, N88, N89, N90, N91, N96,
       N101, N106, N111, N116, N121, N126, N130, N135, N138, N143,
       N146, N149, N152, N153, N156, N159, N165, N171, N177, N183,
       N189, N195, N201, N207, N210, N219, N228, N237, N246, N255,
       N259, N260, N261, N267, N268;
  output N388, N389, N390, N391, N418, N419, N420, N421, N422, N423,
       N446, N447, N448, N449, N450, N767, N768, N850, N863, N864,
       N865, N866, N874, N878, N879, N880;
  wire N1, N8, N13, N17, N26, N29, N36, N42, N51, N55, N59, N68, N72,
       N73, N74, N75, N80, N85, N86, N87, N88, N89, N90, N91, N96,
       N101, N106, N111, N116, N121, N126, N130, N135, N138, N143,
       N146, N149, N152, N153, N156, N159, N165, N171, N177, N183,
       N189, N195, N201, N207, N210, N219, N228, N237, N246, N255,
       N259, N260, N261, N267, N268;
  wire N388, N389, N390, N391, N418, N419, N420, N421, N422, N423,
       N446, N447, N448, N449, N450, N767, N768, N850, N863, N864,
       N865, N866, N874, N878, N879, N880;
  wire n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7;
  wire n_8, n_9, n_10, n_11, n_12, n_13, n_14, n_15;
  wire n_16, n_17, n_18, n_19, n_20, n_21, n_22, n_23;
  wire n_24, n_25, n_26, n_27, n_28, n_29, n_30, n_31;
  wire n_32, n_33, n_34, n_35, n_36, n_37, n_38, n_39;
  wire n_40, n_41, n_42, n_43, n_44, n_45, n_46, n_47;
  wire n_48, n_50, n_51, n_52, n_53, n_54, n_55, n_56;
  wire n_57, n_58, n_59, n_60, n_61, n_62, n_65, n_66;
  wire n_67, n_68, n_70, n_71, n_72, n_73, n_74, n_75;
  wire n_77, n_78, n_80, n_81, n_83, n_84, n_85, n_86;
  wire n_98, n_99, n_100, n_101, n_102, n_103, n_104, n_105;
  wire n_107, n_108, n_109, n_110, n_111, n_113, n_115, n_116;
  wire n_117, n_118, n_119, n_120, n_122, n_124, n_125, n_126;
  wire n_127, n_128, n_129, n_130, n_131, n_132, n_133, n_134;
  wire n_135, n_136, n_137, n_138, n_139, n_141, n_142, n_143;
  wire n_146, n_147, n_148, n_149, n_150, n_151, n_152, n_153;
  wire n_154, n_155, n_156, n_157, n_158, n_159, n_160, n_161;
  wire n_163, n_164, n_165, n_166, n_167, n_168, n_169, n_170;
  wire n_171, n_172, n_173, n_174, n_175, n_176, n_177, n_178;
  wire n_179, n_180, n_181, n_182, n_183, n_184, n_185, n_186;
  wire n_187, n_188, n_189, n_190, n_191, n_192, n_193, n_194;
  wire n_195, n_196, n_197, n_198, n_199, n_200, n_201, n_202;
  wire n_203, n_204, n_205, n_206, n_207, n_208, n_209, n_210;
  wire n_211, n_212, n_213, n_214, n_215, n_216, n_217, n_218;
  wire n_219, n_220, n_221, n_222, n_223, n_224, n_225, n_226;
  wire n_227, n_228, n_229, n_230, n_231, n_232, n_233, n_234;
  wire n_235, n_236, n_237, n_238, n_239, n_240, n_241, n_242;
  wire n_243, n_244, n_245, n_246, n_247, n_248, n_249, n_250;
  wire n_251, n_252, n_253, n_254, n_255, n_256, n_257, n_258;
  wire n_259, n_260, n_261, n_262, n_263, n_264, n_265, n_266;
  wire n_267, n_268, n_269, n_270, n_271, n_272, n_273, n_275;
  wire n_276, n_277, n_278, n_279, n_280, n_281, n_282, n_283;
  wire n_284, n_285, n_286, n_287, n_288, n_289, n_290, n_291;
  wire n_293, n_294, n_295, n_296, n_297, n_298, n_299, n_300;
  wire n_301, n_302, n_304, n_305, n_306, n_307, n_308, n_309;
  wire n_310, n_311, n_312, n_313, n_314, n_316, n_317, n_318;
  wire n_319, n_320, n_321, n_322, n_323, n_324, n_325, n_326;
  wire n_327, n_328, n_329, n_331, n_332, n_333, n_334, n_335;
  wire n_336, n_337, n_338, n_339, n_340, n_341, n_342, n_343;
  wire n_346, n_347, n_348, n_349, n_350, n_351, n_352, n_354;
  wire n_355, n_356;
  nand g2541__7837 (N878, n_355, n_356);
  nor g2542__7557 (n_356, n_241, n_354);
  not g2544 (n_355, n_352);
  not g2545 (n_354, n_351);
  nand g2543__7654 (N879, n_346, n_350);
  nand g2546__8867 (n_352, n_37, n_348);
  nand g2547__1377 (n_351, n_340, n_349);
  nor g2548__3717 (n_350, n_237, n_347);
  nand g2549__4599 (n_349, n_333, n_339);
  nand g2550__3779 (n_348, N219, n_341);
  not g2556 (n_347, n_342);
  not g2555 (n_346, n_343);
  nand g2551__2007 (N866, n_220, n_337);
  nand g2552__1237 (N880, n_329, n_336);
  nand g2558__1297 (n_343, n_27, n_335);
  nand g2559__2006 (n_342, n_325, n_334);
  nor g2553__2833 (n_341, n_340, n_338);
  nand g2554__7547 (n_339, N219, n_338);
  nand g2557__7765 (n_337, n_226, n_332);
  nor g2560__9867 (n_336, n_235, n_331);
  nand g2561__3377 (n_335, N219, n_326);
  nand g2563__9719 (n_334, n_333, n_324);
  not g2562 (n_338, n_332);
  not g2569 (n_331, n_327);
  nand g2564__1591 (n_332, n_218, n_322);
  nand g2565__6789 (N874, n_317, n_321);
  not g2568 (n_329, n_328);
  nand g2571__5927 (n_328, n_32, n_320);
  nand g2572__2001 (n_327, n_313, n_319);
  nor g2566__1122 (n_326, n_325, n_323);
  nand g2567__2005 (n_324, N219, n_323);
  nand g2570__9771 (n_322, n_225, n_318);
  nor g2573__3457 (n_321, n_236, n_316);
  nand g2574__1279 (n_320, N219, n_314);
  nand g2576__6179 (n_319, n_333, n_312);
  not g2575 (n_323, n_318);
  not g2581 (n_317, n_310);
  nand g2577__7837 (n_318, n_216, n_308);
  not g2582 (n_316, n_309);
  nand g2578__7557 (N863, n_302, n_307);
  nor g2579__7654 (n_314, n_313, n_311);
  nand g2580__8867 (n_312, N219, n_311);
  nand g2584__1377 (n_310, n_33, n_306);
  nand g2585__3717 (n_309, n_297, n_304);
  nand g2583__4599 (n_308, n_224, n_305);
  nor g2586__3779 (n_307, n_223, n_301);
  nand g2589__2007 (n_306, N219, n_298);
  not g2588 (n_311, n_305);
  nand g2590__1237 (n_304, n_333, n_296);
  nand g2587__1297 (N864, n_262, n_294);
  not g2595 (n_302, n_300);
  not g2596 (n_301, n_299);
  nand g2591__2006 (n_305, n_214, n_293);
  nand g2597__2833 (n_300, n_25, n_291);
  nand g2599__7547 (n_299, n_285, n_290);
  nor g2592__7765 (n_298, n_297, n_295);
  nand g2593__9867 (n_296, N219, n_295);
  nor g2594__3377 (n_294, n_277, n_289);
  nand g2598__9719 (n_293, n_230, n_288);
  nand g2600__1591 (N865, n_263, n_282);
  nand g2601__6789 (n_291, N219, n_286);
  nand g2603__5927 (n_290, n_333, n_284);
  nand g2604__2001 (n_289, n_160, n_287);
  not g2602 (n_295, n_288);
  nand g2605__1122 (n_288, n_201, n_281);
  nand g2606__2005 (n_287, n_246, n_280);
  nor g2607__9771 (n_286, n_285, n_283);
  nand g2608__3457 (n_284, N219, n_283);
  nor g2609__1279 (n_282, n_267, n_279);
  nand g2610__6179 (n_281, n_210, n_278);
  nand g2612__7837 (n_280, n_333, n_275);
  nand g2614__7557 (n_279, n_154, n_276);
  not g2613 (n_283, n_278);
  nor g2611__7654 (n_277, n_266, n_272);
  nand g2615__8867 (n_278, n_233, n_273);
  nand g2616__1377 (n_276, n_248, n_270);
  nand g2618__3717 (n_275, N219, n_268);
  nand g2617__4599 (N850, n_261, n_269);
  nand g2619__3779 (n_273, n_234, n_271);
  nand g2620__2007 (n_272, n_245, n_271);
  nand g2621__1237 (n_270, n_333, n_265);
  nor g2622__1297 (n_269, n_264, n_259);
  not g2623 (n_268, n_271);
  nor g2624__2006 (n_267, n_266, n_258);
  nand g2625__2833 (n_271, n_231, n_260);
  nand g2629__7547 (n_265, N219, n_255);
  nand g2631__7765 (n_264, n_156, n_256);
  nor g2626__9867 (n_263, n_227, n_253);
  nor g2627__3377 (n_262, n_229, n_254);
  nor g2628__9719 (n_261, n_228, n_251);
  nand g2630__1591 (n_260, n_232, n_257);
  nor g2632__6789 (n_259, n_266, n_252);
  nand g2633__5927 (n_258, n_247, n_257);
  nand g2640__2001 (n_256, n_77, n_250);
  not g2641 (n_255, n_257);
  nand g2642__1122 (n_254, n_21, n_242);
  nand g2643__2005 (n_253, n_20, n_243);
  nand g2635__9771 (n_252, N261, n_249);
  nand g2638__3457 (n_251, n_30, n_244);
  nand g2648__1279 (n_257, n_240, n_239);
  not g2649 (n_250, n_249);
  not g2662 (n_248, n_247);
  not g2663 (n_246, n_245);
  nand g2656__6179 (n_244, N237, n_209);
  nand g2652__7837 (n_243, N237, n_207);
  nand g2651__7557 (n_242, N237, n_208);
  nand g2637__7654 (n_241, n_155, n_212);
  nand g2661__8867 (n_249, n_240, n_238);
  nand g2668__1377 (n_239, N261, n_238);
  nand g2639__3717 (n_237, n_149, n_211);
  nand g2636__4599 (n_236, n_157, n_222);
  nand g2634__3779 (n_235, n_158, n_213);
  nand g2674__2007 (n_245, n_233, n_234);
  nand g2673__1237 (n_247, n_231, n_232);
  nand g2664__1297 (n_230, n_0, n_205);
  nand g2665__2006 (n_229, n_40, n_198);
  wire w, w0, w1, w2;
  nand g2646__2833 (n_340, w0, w2);
  nand g2 (w2, w1, N159);
  not g1 (w1, n_219);
  nand g0 (w0, w, n_219);
  not g (w, N159);
  wire w3, w4, w5, w6;
  nand g2647__7547 (n_325, w4, w6);
  nand g6 (w6, w5, N165);
  not g5 (w5, n_217);
  nand g4 (w4, w3, n_217);
  not g3 (w3, N165);
  wire w7, w8, w9, w10;
  nand g2644__7765 (n_313, w8, w10);
  nand g10 (w10, w9, N171);
  not g9 (w9, n_215);
  nand g8 (w8, w7, n_215);
  not g7 (w7, N171);
  wire w11, w12, w13, w14;
  nand g2645__9867 (n_297, w12, w14);
  nand g14 (w14, w13, N177);
  not g13 (w13, n_221);
  nand g12 (w12, w11, n_221);
  not g11 (w11, N177);
  nand g2666__3377 (n_228, n_38, n_202);
  nand g2667__9719 (n_227, n_15, n_197);
  nand g2669__1591 (n_226, n_12, n_204);
  nand g2670__6789 (n_225, n_3, n_203);
  nand g2671__5927 (n_224, n_13, n_206);
  nand g2672__2001 (n_223, n_161, n_199);
  nand g2657__1122 (n_222, n_65, n_221);
  nand g2653__2005 (n_220, N159, n_219);
  nand g2654__9771 (n_218, N165, n_217);
  nand g2655__3457 (n_216, N171, n_215);
  nand g2650__1279 (n_214, N177, n_221);
  nand g2658__6179 (n_213, n_75, n_215);
  nand g2659__7837 (n_212, n_73, n_219);
  nand g2660__7557 (n_211, n_71, n_217);
  nand g2691__7654 (n_210, n_11, n_189);
  not g2676 (n_209, n_240);
  not g2677 (n_208, n_233);
  not g2690 (n_207, n_231);
  nand g2697__8867 (n_238, n_5, n_192);
  nand g2696__1377 (n_234, n_7, n_190);
  nand g2695__3717 (n_232, n_9, n_191);
  wire w15, w16, w17, w18;
  nand g2675__4599 (n_285, w16, w18);
  nand g18 (w18, w17, N183);
  not g17 (w17, n_200);
  nand g16 (w16, w15, n_200);
  not g15 (w15, N183);
  not g2678 (n_206, n_215);
  not g2679 (n_205, n_221);
  not g2680 (n_204, n_219);
  not g2681 (n_203, n_217);
  nand g2682__3779 (n_202, N246, n_195);
  nand g2683__2007 (n_201, N183, n_200);
  nand g2692__1237 (n_199, n_70, n_200);
  nand g2693__1297 (n_198, N246, n_196);
  nand g2694__2006 (n_197, N246, n_194);
  nand g2685__2833 (n_233, N189, n_196);
  nand g2684__7547 (n_240, N201, n_195);
  nand g2698__7765 (n_231, N195, n_194);
  nand g2688__9867 (n_219, n_193, n_185);
  nand g2687__3377 (n_221, n_193, n_184);
  nand g2686__9719 (n_215, n_193, n_187);
  nand g2689__1591 (n_217, n_193, n_186);
  not g2705 (n_192, n_195);
  not g2704 (n_191, n_194);
  not g2703 (n_190, n_196);
  not g2706 (n_189, n_200);
  nand g2707__6789 (n_196, n_188, n_182);
  nand g2708__5927 (n_194, n_188, n_180);
  nand g2709__2001 (n_195, n_188, n_183);
  nand g2710__1122 (n_200, n_188, n_181);
  nor g2699__2005 (n_187, n_136, n_176);
  nor g2700__9771 (n_186, n_134, n_177);
  nor g2701__3457 (n_185, n_133, n_178);
  nor g2702__1279 (n_184, n_135, n_175);
  not g2711 (n_183, n_179);
  not g2716 (n_182, n_173);
  not g2717 (n_181, n_174);
  not g2718 (n_180, n_172);
  nand g2714__6179 (n_179, n_147, n_171);
  nand g2713__7837 (n_178, n_24, n_164);
  nand g2712__7557 (n_177, n_16, n_167);
  nand g2715__7654 (n_176, n_14, n_163);
  nand g2721__8867 (n_175, n_36, n_165);
  nand g2720__1377 (n_174, n_153, n_166);
  nand g2719__3717 (n_173, n_152, n_168);
  nand g2722__4599 (n_172, n_150, n_170);
  nand g2723__3779 (n_171, N126, n_169);
  nand g2729__2007 (n_170, N121, n_169);
  nand g2730__1237 (n_168, N116, n_169);
  nand g2724__1297 (n_167, N96, n_169);
  nand g2727__2006 (n_166, N111, n_169);
  nand g2726__2833 (n_165, N106, n_169);
  nand g2725__7547 (n_164, N91, n_169);
  nand g2728__7765 (n_163, N101, n_169);
  nand g2733__9867 (n_169, n_141, n_148);
  nor g2735__3377 (N449, n_2, n_146);
  nand g2744__9719 (n_161, N183, n_159);
  nand g2739__1591 (n_160, N189, n_159);
  nand g2740__6789 (n_158, N171, n_159);
  nand g2741__5927 (n_157, N177, n_159);
  nand g2738__2001 (n_156, N201, n_159);
  nand g2743__1122 (n_155, N159, n_159);
  nand g2742__2005 (n_193, N75, n_143);
  nand g2737__9771 (n_154, N195, n_159);
  nand g2746__3457 (n_153, N143, n_151);
  nand g2747__1279 (n_152, N146, n_151);
  nand g2748__6179 (n_150, N149, n_151);
  nand g2734__7837 (n_149, N165, n_159);
  nand g2736__7557 (n_148, n_46, n_142);
  nand g2745__7654 (n_147, N153, n_151);
  nand g2753__8867 (n_146, N74, n_137);
  nand g2760__1377 (n_188, N75, n_138);
  wire w19, w20, w21, w22;
  nand g2731__3717 (N768, w20, w22);
  nand g22 (w22, w21, n_128);
  not g21 (w21, n_131);
  nand g20 (w20, w19, n_131);
  not g19 (w19, n_128);
  wire w23, w24, w25, w26;
  nand g2732__4599 (N767, w24, w26);
  nand g26 (w26, w25, n_129);
  not g25 (w25, n_130);
  nand g24 (w24, w23, n_130);
  not g23 (w23, n_129);
  nor g2754__3779 (n_143, N268, n_126);
  nand g2757__2007 (n_142, n_120, n_124);
  nor g2756__1237 (n_159, n_81, n_139);
  nand g2758__1297 (n_141, N17, n_132);
  nor g2755__2006 (N448, n_8, n_139);
  nand g2759__2833 (n_151, N1, n_127);
  not g2762 (n_138, n_122);
  not g2763 (n_137, n_139);
  not g2772 (n_136, n_119);
  not g2773 (n_135, n_118);
  not g2774 (n_134, n_116);
  not g2775 (n_133, n_115);
  not g2761 (n_132, n_111);
  wire w27, w28, w29, w30;
  nand g2750__7547 (n_131, w28, w30);
  nand g30 (w30, w29, N207);
  not g29 (w29, n_102);
  nand g28 (w28, w27, n_102);
  not g27 (w27, N207);
  wire w31, w32, w33, w34;
  nand g2751__7765 (n_130, w32, w34);
  nand g34 (w34, w33, N135);
  not g33 (w33, n_99);
  nand g32 (w32, w31, n_99);
  not g31 (w31, N135);
  wire w35, w36, w37, w38;
  nand g2752__9867 (n_129, w36, w38);
  nand g38 (w38, w37, N130);
  not g37 (w37, n_101);
  nand g36 (w36, w35, n_101);
  not g35 (w35, N130);
  wire w39, w40, w41, w42;
  nand g2749__3377 (n_128, w40, w42);
  nand g42 (w42, w41, N130);
  not g41 (w41, n_100);
  nand g40 (w40, w39, n_100);
  not g39 (w39, N130);
  nand g2765__9719 (n_127, n_66, n_125);
  nand g2766__1591 (n_126, n_83, n_125);
  nand g2767__6789 (n_124, n_109, n_125);
  nand g2770__5927 (N419, n_86, n_113);
  nand g2769__2001 (n_122, N447, n_107);
  nand g2768__1122 (n_120, n_104, n_110);
  nand g2776__2005 (n_119, N149, n_117);
  nand g2777__9771 (n_118, N153, n_117);
  nand g2778__3457 (n_116, N146, n_117);
  nand g2779__1279 (n_115, N143, n_117);
  nand g2780__6179 (N446, N390, n_113);
  nand g2771__7837 (n_139, N55, n_108);
  nand g2764__7557 (n_111, N51, n_98);
  nor g2785__7654 (n_110, n_109, n_103);
  nor g2786__8867 (n_108, n_10, n_105);
  nor g2787__1377 (n_107, N268, n_84);
  nor g2789__3717 (N418, n_104, n_105);
  nor g2791__4599 (n_113, n_104, n_85);
  nor g2792__3779 (n_125, n_104, n_103);
  nor g2788__2007 (n_117, n_67, n_103);
  wire w43, w44, w45, w46;
  nand g2783__1237 (n_102, w44, w46);
  nand g46 (w46, w45, n_54);
  not g45 (w45, n_45);
  nand g44 (w44, w43, n_45);
  not g43 (w43, n_54);
  wire w47, w48, w49, w50;
  nand g2782__1297 (n_101, w48, w50);
  nand g50 (w50, w49, n_59);
  not g49 (w49, n_58);
  nand g48 (w48, w47, n_58);
  not g47 (w47, n_59);
  wire w51, w52, w53, w54;
  nand g2781__2006 (n_100, w52, w54);
  nand g54 (w54, w53, n_55);
  not g53 (w53, n_56);
  nand g52 (w52, w51, n_56);
  not g51 (w51, n_55);
  wire w55, w56, w57, w58;
  nand g2784__2833 (n_99, w56, w58);
  nand g58 (w58, w57, n_60);
  not g57 (w57, n_57);
  nand g56 (w56, w55, n_57);
  not g55 (w55, n_60);
  nor g2790__7547 (n_98, n_50, n_74);
  not g2793 (N389, n_80);
  not g2795 (N447, n_103);
  not g2794 (N390, n_86);
  nand g2798__7765 (n_85, N1, n_47);
  nand g2797__9867 (n_84, N55, n_83);
  nand g2806__3377 (N420, N59, n_53);
  nand g2801__9719 (n_81, n_52, n_78);
  nand g2804__1591 (n_80, N36, n_83);
  nand g2805__6789 (N422, N36, n_78);
  nand g2814__5927 (n_77, n_333, n_44);
  nand g2802__2001 (N421, N59, n_61);
  nand g2808__1122 (n_86, N36, n_43);
  nand g2807__2005 (n_105, N13, n_51);
  nand g2809__9771 (n_103, N1, n_48);
  nand g2810__3457 (n_75, n_72, n_39);
  nor g2799__1279 (n_74, n_68, n_41);
  nand g2816__6179 (n_73, n_72, n_28);
  nand g2811__7837 (n_71, n_72, n_22);
  nand g2812__7557 (n_70, n_72, n_31);
  nor g2796__7654 (N388, n_68, n_42);
  nand g2815__8867 (n_67, N55, n_66);
  nand g2813__1377 (n_65, n_72, n_34);
  nor g2803__3717 (N450, n_4, n_62);
  nor g2800__4599 (N423, n_1, n_62);
  not g2848 (n_61, n_17);
  wire w59, w60, w61, w62;
  nand g2817__3779 (n_60, w60, w62);
  nand g62 (w62, w61, N116);
  not g61 (w61, N111);
  nand g60 (w60, w59, N111);
  not g59 (w59, N116);
  wire w63, w64, w65, w66;
  nand g2818__2007 (n_59, w64, w66);
  nand g66 (w66, w65, N96);
  not g65 (w65, N91);
  nand g64 (w64, w63, N91);
  not g63 (w63, N96);
  wire w67, w68, w69, w70;
  nand g2819__1237 (n_58, w68, w70);
  nand g70 (w70, w69, N106);
  not g69 (w69, N101);
  nand g68 (w68, w67, N101);
  not g67 (w67, N106);
  wire w71, w72, w73, w74;
  nand g2821__1297 (n_57, w72, w74);
  nand g74 (w74, w73, N126);
  not g73 (w73, N121);
  nand g72 (w72, w71, N121);
  not g71 (w71, N126);
  wire w75, w76, w77, w78;
  nand g2822__2006 (n_56, w76, w78);
  nand g78 (w78, w77, N177);
  not g77 (w77, N171);
  nand g76 (w76, w75, N171);
  not g75 (w75, N177);
  wire w79, w80, w81, w82;
  nand g2823__2833 (n_55, w80, w82);
  nand g82 (w82, w81, N165);
  not g81 (w81, N159);
  nand g80 (w80, w79, N159);
  not g79 (w79, N165);
  wire w83, w84, w85, w86;
  nand g2824__7547 (n_54, w84, w86);
  nand g86 (w86, w85, N189);
  not g85 (w85, N183);
  nand g84 (w84, w83, N183);
  not g83 (w83, N189);
  not g2825 (n_53, n_29);
  not g2826 (n_52, n_35);
  not g2827 (n_51, n_50);
  not g2846 (N391, n_23);
  not g2845 (n_48, n_18);
  not g2847 (n_47, n_19);
  not g2851 (n_46, n_66);
  wire w87, w88, w89, w90;
  nand g2820__7765 (n_45, w88, w90);
  nand g90 (w90, w89, N201);
  not g89 (w89, N195);
  nand g88 (w88, w87, N195);
  not g87 (w87, N201);
  nand g2830__9867 (n_44, N219, n_6);
  not g2849 (n_43, n_42);
  not g2850 (n_78, n_41);
  not g2828 (n_83, n_26);
  nand g2841__3377 (n_40, N259, N255);
  nand g2864__9719 (n_39, N237, N171);
  nand g2839__1591 (n_38, N267, N255);
  nand g2865__6789 (n_37, N268, N210);
  nand g2860__5927 (n_36, N152, N138);
  nand g2840__2001 (n_35, N73, N72);
  nand g2832__1122 (n_34, N237, N177);
  nand g2833__2005 (n_33, N210, N101);
  nand g2834__9771 (n_32, N210, N96);
  nand g2835__3457 (n_31, N237, N183);
  nand g2859__1279 (n_30, N210, N121);
  nand g2836__6179 (n_29, N80, N75);
  nand g2838__7837 (n_28, N237, N159);
  nand g2831__7557 (n_27, N210, N91);
  nand g2867__7654 (n_41, N59, N42);
  nand g2866__8867 (n_42, N42, N29);
  nand g2844__1377 (n_26, N80, N29);
  nand g2853__3717 (n_25, N210, N106);
  nand g2855__4599 (n_24, N138, N8);
  nand g2856__3779 (n_23, N86, N85);
  nand g2854__2007 (n_22, N237, N165);
  nand g2862__1237 (n_21, N210, N111);
  nand g2863__1297 (n_20, N210, N116);
  nand g2857__2006 (n_19, N26, N13);
  nand g2852__2833 (n_18, N51, N26);
  nand g2861__7547 (n_17, N80, N36);
  nand g2837__7765 (n_16, N138, N51);
  nand g2858__9867 (n_15, N260, N255);
  nand g2829__3377 (n_14, N138, N17);
  nor g2843__9719 (n_62, N88, N87);
  nand g2842__1591 (n_50, N8, N1);
  nand g2868__6789 (n_66, N156, N59);
  not g2887 (n_13, N171);
  not g2877 (n_12, N159);
  not g2874 (n_11, N183);
  not g2879 (n_10, N68);
  not g2884 (n_9, N195);
  not g2880 (n_8, N29);
  not g2883 (n_7, N189);
  not g2870 (n_109, N42);
  not g2888 (n_266, N219);
  not g2886 (n_333, N228);
  not g2881 (n_6, N261);
  not g2872 (n_5, N201);
  not g2869 (n_4, N89);
  not g2873 (n_3, N165);
  not g2871 (n_2, N59);
  not g2878 (n_1, N90);
  not g2875 (n_0, N177);
  not g2882 (n_68, N75);
  not g2885 (n_104, N17);
  not g2876 (n_72, N246);
endmodule
