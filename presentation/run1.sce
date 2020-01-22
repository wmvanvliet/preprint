write_codes = true;
active_buttons = 2;
begin;

picture {
  default_code = "fixation";
  bitmap {
		 filename = "fixation_cross.png";
  };
  x = 0; y = 0;
} fixation;

TEMPLATE "stimulus.tem" {
word file code;
"svsod" "symbols_svsod.png" 30;
"cnwh" "consonants_cnwh.png" 20;
"v^v^" "symbols_v^v^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ ^ _ _" "symbols_v^v^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^o^^v" "symbols_s^o^^v.png" 30;
"dvsvos" "symbols_dvsvos.png" 30;
"orpo" "word_orpo.png" 10;
"vodv^^" "symbols_vodv^^.png" 30;
"scgxmk" "consonants_scgxmk.png" 20;
"karies" "word_karies.png" 10;
"tenä" "word_tenä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ e _ _" "word_tenä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ampuja" "word_ampuja.png" 10;
"ässä" "word_ässä.png" 10;
"vssov" "symbols_vssov.png" 30;
"kyhmy" "word_kyhmy.png" 10;
"jänne" "word_jänne.png" 10;
"hormi" "word_hormi.png" 10;
"bfkx" "consonants_bfkx.png" 20;
"sdsosv" "symbols_sdsosv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "symbols_sdsosv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"d^^do" "symbols_d^^do.png" 30;
"kihu" "word_kihu.png" 10;
"äänne" "word_äänne.png" 10;
"pidot" "word_pidot.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _ _" "word_pidot_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uima" "word_uima.png" 10;
"ääriin" "word_ääriin.png" 10;
"^dso" "symbols_^dso.png" 30;
"rkgpjs" "consonants_rkgpjs.png" 20;
"^vdovd" "symbols_^vdovd.png" 30;
"gxqtx" "consonants_gxqtx.png" 20;
"äänes" "word_äänes.png" 10;
"älytä" "word_älytä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ä" "word_älytä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sodo" "symbols_sodo.png" 30;
"wvxw" "consonants_wvxw.png" 20;
"kustos" "word_kustos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ü _ _ _" "word_kustos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"köli" "word_köli.png" 10;
"tämä" "word_tämä.png" 10;
"vvovov" "symbols_vvovov.png" 30;
"solmio" "word_solmio.png" 10;
"tyvi" "word_tyvi.png" 10;
"estyä" "word_estyä.png" 10;
"os^ood" "symbols_os^ood.png" 30;
"salaus" "word_salaus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_salaus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^vvsvv" "symbols_^vvsvv.png" 30;
"odsoo^" "symbols_odsoo^.png" 30;
"vipu" "word_vipu.png" 10;
"^d^oos" "symbols_^d^oos.png" 30;
"tkcqd" "consonants_tkcqd.png" 20;
"txhs" "consonants_txhs.png" 20;
"vvvd" "symbols_vvvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _" "symbols_vvvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"isyys" "word_isyys.png" 10;
"shiia" "word_shiia.png" 10;
"itää" "word_itää.png" 10;
"piip" "word_piip.png" 10;
"osdod^" "symbols_osdod^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s _" "symbols_osdod^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"emätin" "word_emätin.png" 10;
"vwrptk" "consonants_vwrptk.png" 20;
"ositus" "word_ositus.png" 10;
"rtsgdh" "consonants_rtsgdh.png" 20;
"korren" "word_korren.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ w _ _" "word_korren_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jqdsmj" "consonants_jqdsmj.png" 20;
"juhta" "word_juhta.png" 10;
"poru" "word_poru.png" 10;
"vodds" "symbols_vodds.png" 30;
"o^v^" "symbols_o^v^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s" "symbols_o^v^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gchqcz" "consonants_gchqcz.png" 20;
"uuni" "word_uuni.png" 10;
"qpvbs" "consonants_qpvbs.png" 20;
"tcfwdr" "consonants_tcfwdr.png" 20;
"zsgj" "consonants_zsgj.png" 20;
"hioa" "word_hioa.png" 10;
"nieriä" "word_nieriä.png" 10;
"vvosod" "symbols_vvosod.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_vvosod_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gnxd" "consonants_gnxd.png" 20;
"spmb" "consonants_spmb.png" 20;
"osuma" "word_osuma.png" 10;
"afasia" "word_afasia.png" 10;
"zkwnr" "consonants_zkwnr.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "consonants_zkwnr_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svvd" "symbols_svvd.png" 30;
"rulla" "word_rulla.png" 10;
"^o^so" "symbols_^o^so.png" 30;
"itiö" "word_itiö.png" 10;
"kytkin" "word_kytkin.png" 10;
"czkdbs" "consonants_czkdbs.png" 20;
"luusto" "word_luusto.png" 10;
"uuma" "word_uuma.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ u _ _" "word_uuma_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vsv^" "symbols_vsv^.png" 30;
"urut" "word_urut.png" 10;
"silaus" "word_silaus.png" 10;
"jdfjs" "consonants_jdfjs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ k _" "consonants_jdfjs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"myyty" "word_myyty.png" 10;
"lblxm" "consonants_lblxm.png" 20;
"s^dv^" "symbols_s^dv^.png" 30;
"ovo^s" "symbols_ovo^s.png" 30;
"eliö" "word_eliö.png" 10;
"arpoa" "word_arpoa.png" 10;
"d^v^^" "symbols_d^v^^.png" 30;
"sarake" "word_sarake.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"ä _ _ _ _ _" "word_sarake_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bgrzd" "consonants_bgrzd.png" 20;
"zpcqc" "consonants_zpcqc.png" 20;
"gnbtn" "consonants_gnbtn.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ n _ _ _" "consonants_gnbtn_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sovo" "symbols_sovo.png" 30;
"uoma" "word_uoma.png" 10;
"xgtt" "consonants_xgtt.png" 20;
"ssods" "symbols_ssods.png" 30;
"vvv^v" "symbols_vvv^v.png" 30;
"sclc" "consonants_sclc.png" 20;
"dwmdd" "consonants_dwmdd.png" 20;
"ratamo" "word_ratamo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _ _ _" "word_ratamo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ilmi" "word_ilmi.png" 10;
"ajos" "word_ajos.png" 10;
"qmncn" "consonants_qmncn.png" 20;
"ltqrr" "consonants_ltqrr.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"l _ _ _ _" "consonants_ltqrr_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gkzpg" "consonants_gkzpg.png" 20;
"nisä" "word_nisä.png" 10;
"lohi" "word_lohi.png" 10;
"vovvvv" "symbols_vovvvv.png" 30;
"zkxtf" "consonants_zkxtf.png" 20;
"wcjrjq" "consonants_wcjrjq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"w _ _ _ _ _" "consonants_wcjrjq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"syaani" "word_syaani.png" 10;
"rjfcdx" "consonants_rjfcdx.png" 20;
"^s^vds" "symbols_^s^vds.png" 30;
"hjqc" "consonants_hjqc.png" 20;
"^vod" "symbols_^vod.png" 30;
"rmmrh" "consonants_rmmrh.png" 20;
"salvaa" "word_salvaa.png" 10;
"kiuas" "word_kiuas.png" 10;
"vdod^" "symbols_vdod^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_vdod^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vtkdtx" "consonants_vtkdtx.png" 20;
"dod^os" "symbols_dod^os.png" 30;
"torium" "word_torium.png" 10;
"someen" "word_someen.png" 10;
"häkä" "word_häkä.png" 10;
"^s^^" "symbols_^s^^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ^ _" "symbols_^s^^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ssdvo^" "symbols_ssdvo^.png" 30;
"pamppu" "word_pamppu.png" 10;
"stoola" "word_stoola.png" 10;
"erkani" "word_erkani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ a _ _" "word_erkani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tdgw" "consonants_tdgw.png" 20;
"xggtw" "consonants_xggtw.png" 20;
"jäte" "word_jäte.png" 10;
"räntä" "word_räntä.png" 10;
"sddo" "symbols_sddo.png" 30;
"lgtjb" "consonants_lgtjb.png" 20;
"räme" "word_räme.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ i _ _" "word_räme_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vv^^" "symbols_vv^^.png" 30;
"rukki" "word_rukki.png" 10;
"kolvi" "word_kolvi.png" 10;
"dfdzj" "consonants_dfdzj.png" 20;
"vvv^" "symbols_vvv^.png" 30;
"näkö" "word_näkö.png" 10;
"sikhi" "word_sikhi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"h _ _ _ _" "word_sikhi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"katodi" "word_katodi.png" 10;
"tyrä" "word_tyrä.png" 10;
"pesula" "word_pesula.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_pesula_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nirso" "word_nirso.png" 10;
"seimi" "word_seimi.png" 10;
"jaos" "word_jaos.png" 10;
"qwjvhz" "consonants_qwjvhz.png" 20;
"hqzll" "consonants_hqzll.png" 20;
"kaapia" "word_kaapia.png" 10;
"^osd" "symbols_^osd.png" 30;
"sqdq" "consonants_sqdq.png" 20;
"häät" "word_häät.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ l" "word_häät_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^s^d" "symbols_^s^d.png" 30;
"o^ss" "symbols_o^ss.png" 30;
"grgdd" "consonants_grgdd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ x _ _ _" "consonants_grgdd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"säle" "word_säle.png" 10;
"vuoka" "word_vuoka.png" 10;
"tykö" "word_tykö.png" 10;
"vs^oo" "symbols_vs^oo.png" 30;
"vvsod" "symbols_vvsod.png" 30;
"^ooso" "symbols_^ooso.png" 30;
"nurin" "word_nurin.png" 10;
"wqghh" "consonants_wqghh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ h" "consonants_wqghh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vlptrn" "consonants_vlptrn.png" 20;
"harjus" "word_harjus.png" 10;
"voov^v" "symbols_voov^v.png" 30;
"känsä" "word_känsä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"a _ _ _ _" "word_känsä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lcrzjb" "consonants_lcrzjb.png" 20;
"mths" "consonants_mths.png" 20;
"diodi" "word_diodi.png" 10;
"dv^o" "symbols_dv^o.png" 30;
"ovvds" "symbols_ovvds.png" 30;
"^d^vs" "symbols_^d^vs.png" 30;
"faksi" "word_faksi.png" 10;
"kortti" "word_kortti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ t _ _" "word_kortti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"duuri" "word_duuri.png" 10;
"fotoni" "word_fotoni.png" 10;
"häpy" "word_häpy.png" 10;
"menijä" "word_menijä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _ _" "word_menijä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oo^v" "symbols_oo^v.png" 30;
"särkyä" "word_särkyä.png" 10;
"oinas" "word_oinas.png" 10;
"motata" "word_motata.png" 10;
"silsa" "word_silsa.png" 10;
"dzvxq" "consonants_dzvxq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q" "consonants_dzvxq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdsdoo" "symbols_sdsdoo.png" 30;
"voo^o" "symbols_voo^o.png" 30;
"jymy" "word_jymy.png" 10;
"gqpkfs" "consonants_gqpkfs.png" 20;
"sdosd" "symbols_sdosd.png" 30;
"ddvsd" "symbols_ddvsd.png" 30;
"nczkbj" "consonants_nczkbj.png" 20;
"ripeä" "word_ripeä.png" 10;
"hihna" "word_hihna.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ü _" "word_hihna_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^s^s" "symbols_^s^s.png" 30;
"vd^s" "symbols_vd^s.png" 30;
"dsdoo^" "symbols_dsdoo^.png" 30;
"qckq" "consonants_qckq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ q" "consonants_qckq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qvkjt" "consonants_qvkjt.png" 20;
"xnbrh" "consonants_xnbrh.png" 20;
"kussa" "word_kussa.png" 10;
"v^s^vd" "symbols_v^s^vd.png" 30;
"sävy" "word_sävy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ e _" "word_sävy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zkbwm" "consonants_zkbwm.png" 20;
"blkxwz" "consonants_blkxwz.png" 20;
"psalmi" "word_psalmi.png" 10;
"kysta" "word_kysta.png" 10;
"terska" "word_terska.png" 10;
"oo^vdv" "symbols_oo^vdv.png" 30;
"druidi" "word_druidi.png" 10;
"lldk" "consonants_lldk.png" 20;
"siitos" "word_siitos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"i _ _ _ _ _" "word_siitos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fwgntw" "consonants_fwgntw.png" 20;
"vovso" "symbols_vovso.png" 30;
"dlgt" "consonants_dlgt.png" 20;
"äimä" "word_äimä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"e _ _ _" "word_äimä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"huorin" "word_huorin.png" 10;
"uute" "word_uute.png" 10;
"sopu" "word_sopu.png" 10;
"hnxf" "consonants_hnxf.png" 20;
"tyköä" "word_tyköä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "word_tyköä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vsd^o" "symbols_vsd^o.png" 30;
"dovv^" "symbols_dovv^.png" 30;
"ndsvw" "consonants_ndsvw.png" 20;
"pyörö" "word_pyörö.png" 10;
"gsdx" "consonants_gsdx.png" 20;
"ylkä" "word_ylkä.png" 10;
"^vvdv" "symbols_^vvdv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_^vvdv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tuohus" "word_tuohus.png" 10;
"kohu" "word_kohu.png" 10;
"riimu" "word_riimu.png" 10;
"zfjxqk" "consonants_zfjxqk.png" 20;
"xmpt" "consonants_xmpt.png" 20;
"dodv" "symbols_dodv.png" 30;
"loraus" "word_loraus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ o _ _ _ _" "word_loraus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^vsov" "symbols_v^vsov.png" 30;
"laukka" "word_laukka.png" 10;
"syöpyä" "word_syöpyä.png" 10;
"pokeri" "word_pokeri.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
"klaava" "word_klaava.png" 10;
"kyteä" "word_kyteä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ä" "word_kyteä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^s^sso" "symbols_^s^sso.png" 30;
"uuhi" "word_uuhi.png" 10;
"kuje" "word_kuje.png" 10;
"jaardi" "word_jaardi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ e" "word_jaardi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdsvs" "symbols_sdsvs.png" 30;
"hxjlgq" "consonants_hxjlgq.png" 20;
"gmrq" "consonants_gmrq.png" 20;
"pitäen" "word_pitäen.png" 10;
"tcftcg" "consonants_tcftcg.png" 20;
"ldtg" "consonants_ldtg.png" 20;
"sovssd" "symbols_sovssd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ^ _" "symbols_sovssd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zqshvj" "consonants_zqshvj.png" 20;
"gcfj" "consonants_gcfj.png" 20;
"säiky" "word_säiky.png" 10;
"mpvl" "consonants_mpvl.png" 20;
"ds^^o" "symbols_ds^^o.png" 30;
"dsvvd" "symbols_dsvvd.png" 30;
"hautua" "word_hautua.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _ _" "word_hautua_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mxnhh" "consonants_mxnhh.png" 20;
"^dss" "symbols_^dss.png" 30;
"säie" "word_säie.png" 10;
"ddjs" "consonants_ddjs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ j _" "consonants_ddjs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bzhkd" "consonants_bzhkd.png" 20;
"almu" "word_almu.png" 10;
"uivelo" "word_uivelo.png" 10;
"nuha" "word_nuha.png" 10;
"sysi" "word_sysi.png" 10;
"^svs^" "symbols_^svs^.png" 30;
"ksrwvk" "consonants_ksrwvk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"r _ _ _ _ _" "consonants_ksrwvk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kopina" "word_kopina.png" 10;
"s^od" "symbols_s^od.png" 30;
"rfrqfp" "consonants_rfrqfp.png" 20;
"osdoo^" "symbols_osdoo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ o _" "symbols_osdoo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"täällä" "word_täällä.png" 10;
"hamaan" "word_hamaan.png" 10;
"soov" "symbols_soov.png" 30;
"ähky" "word_ähky.png" 10;
"gongi" "word_gongi.png" 10;
"mäntä" "word_mäntä.png" 10;
"pöty" "word_pöty.png" 10;
"suti" "word_suti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_suti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zdvv" "consonants_zdvv.png" 20;
"noppa" "word_noppa.png" 10;
"viipyä" "word_viipyä.png" 10;
"osinko" "word_osinko.png" 10;
"brsjjh" "consonants_brsjjh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ l" "consonants_brsjjh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"do^s" "symbols_do^s.png" 30;
"lhqt" "consonants_lhqt.png" 20;
"xxqhnz" "consonants_xxqhnz.png" 20;
"lähi" "word_lähi.png" 10;
"sv^vvs" "symbols_sv^vvs.png" 30;
"uiva" "word_uiva.png" 10;
"puida" "word_puida.png" 10;
"pesin" "word_pesin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_pesin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hajan" "word_hajan.png" 10;
"kerubi" "word_kerubi.png" 10;
"rämä" "word_rämä.png" 10;
"s^v^s" "symbols_s^v^s.png" 30;
"nide" "word_nide.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m" "word_nide_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lakea" "word_lakea.png" 10;
"jxfm" "consonants_jxfm.png" 20;
"akti" "word_akti.png" 10;
"lempo" "word_lempo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ e _ _ _" "word_lempo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lemu" "word_lemu.png" 10;
"v^^^^" "symbols_v^^^^.png" 30;
"zjwhs" "consonants_zjwhs.png" 20;
"opaali" "word_opaali.png" 10;
"sovdvo" "symbols_sovdvo.png" 30;
"rapsi" "word_rapsi.png" 10;
"iäti" "word_iäti.png" 10;
"jakaja" "word_jakaja.png" 10;
"vvo^" "symbols_vvo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ^" "symbols_vvo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svds" "symbols_svds.png" 30;
"möly" "word_möly.png" 10;
"lotja" "word_lotja.png" 10;
"sdvd^" "symbols_sdvd^.png" 30;
"qmff" "consonants_qmff.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _" "consonants_qmff_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"säkä" "word_säkä.png" 10;
"^o^o^v" "symbols_^o^o^v.png" 30;
"zlln" "consonants_zlln.png" 20;
"otsoni" "word_otsoni.png" 10;
"jtltfj" "consonants_jtltfj.png" 20;
"miten" "word_miten.png" 10;
"kaste" "word_kaste.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _" "word_kaste_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ahjo" "word_ahjo.png" 10;
"^vd^vd" "symbols_^vd^vd.png" 30;
"fuksi" "word_fuksi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ z _" "word_fuksi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vahaus" "word_vahaus.png" 10;
"rwfcq" "consonants_rwfcq.png" 20;
"kolhia" "word_kolhia.png" 10;
"näppy" "word_näppy.png" 10;
"sodvds" "symbols_sodvds.png" 30;
"^ddv" "symbols_^ddv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v" "symbols_^ddv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svks" "consonants_svks.png" 20;
"xkfvmb" "consonants_xkfvmb.png" 20;
"suippo" "word_suippo.png" 10;
"nrxlzx" "consonants_nrxlzx.png" 20;
"ripsi" "word_ripsi.png" 10;
"kopio" "word_kopio.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ o" "word_kopio_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vaje" "word_vaje.png" 10;
"nurja" "word_nurja.png" 10;
"ilkiö" "word_ilkiö.png" 10;
"sovs" "symbols_sovs.png" 30;
"yöpyä" "word_yöpyä.png" 10;
"vsds" "symbols_vsds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _" "symbols_vsds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"päkiä" "word_päkiä.png" 10;
"odote" "word_odote.png" 10;
"kuskus" "word_kuskus.png" 10;
};
