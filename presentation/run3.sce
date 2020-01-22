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
"ositus" "word_ositus.png" 10;
"salaus" "word_salaus.png" 10;
"ääriin" "word_ääriin.png" 10;
"d^^do" "symbols_d^^do.png" 30;
"^ddv" "symbols_^ddv.png" 30;
"hioa" "word_hioa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _" "word_hioa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ajos" "word_ajos.png" 10;
"xnbrh" "consonants_xnbrh.png" 20;
"^vod" "symbols_^vod.png" 30;
"^ooso" "symbols_^ooso.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _ _" "symbols_^ooso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pöty" "word_pöty.png" 10;
"ilkiö" "word_ilkiö.png" 10;
"otsoni" "word_otsoni.png" 10;
"sdvd^" "symbols_sdvd^.png" 30;
"nisä" "word_nisä.png" 10;
"brsjjh" "consonants_brsjjh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ l" "consonants_brsjjh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nuha" "word_nuha.png" 10;
"kihu" "word_kihu.png" 10;
"säkä" "word_säkä.png" 10;
"duuri" "word_duuri.png" 10;
"rjfcdx" "consonants_rjfcdx.png" 20;
"s^o^^v" "symbols_s^o^^v.png" 30;
"shiia" "word_shiia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "word_shiia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gxqtx" "consonants_gxqtx.png" 20;
"tyrä" "word_tyrä.png" 10;
"säiky" "word_säiky.png" 10;
"vovvvv" "symbols_vovvvv.png" 30;
"^svs^" "symbols_^svs^.png" 30;
"vdod^" "symbols_vdod^.png" 30;
"ddjs" "consonants_ddjs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ j _" "consonants_ddjs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ndsvw" "consonants_ndsvw.png" 20;
"o^ss" "symbols_o^ss.png" 30;
"fuksi" "word_fuksi.png" 10;
"äänne" "word_äänne.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ n _" "word_äänne_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"someen" "word_someen.png" 10;
"jymy" "word_jymy.png" 10;
"^o^so" "symbols_^o^so.png" 30;
"sddo" "symbols_sddo.png" 30;
"vtkdtx" "consonants_vtkdtx.png" 20;
"dzvxq" "consonants_dzvxq.png" 20;
"kysta" "word_kysta.png" 10;
"tykö" "word_tykö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ h _" "word_tykö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lempo" "word_lempo.png" 10;
"kolvi" "word_kolvi.png" 10;
"rämä" "word_rämä.png" 10;
"qmff" "consonants_qmff.png" 20;
"uuma" "word_uuma.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ u _ _" "word_uuma_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rukki" "word_rukki.png" 10;
"vvovov" "symbols_vvovov.png" 30;
"fotoni" "word_fotoni.png" 10;
"salvaa" "word_salvaa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _ _ _" "word_salvaa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ähky" "word_ähky.png" 10;
"nurja" "word_nurja.png" 10;
"svks" "consonants_svks.png" 20;
"lotja" "word_lotja.png" 10;
"scgxmk" "consonants_scgxmk.png" 20;
"räme" "word_räme.png" 10;
"sdsvs" "symbols_sdsvs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _ _" "symbols_sdsvs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lgtjb" "consonants_lgtjb.png" 20;
"stoola" "word_stoola.png" 10;
"lähi" "word_lähi.png" 10;
"xkfvmb" "consonants_xkfvmb.png" 20;
"sdosd" "symbols_sdosd.png" 30;
"qpvbs" "consonants_qpvbs.png" 20;
"kohu" "word_kohu.png" 10;
"isyys" "word_isyys.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ x _ _" "word_isyys_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gcfj" "consonants_gcfj.png" 20;
"motata" "word_motata.png" 10;
"yöpyä" "word_yöpyä.png" 10;
"sdsosv" "symbols_sdsosv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "symbols_sdsosv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"suippo" "word_suippo.png" 10;
"kuskus" "word_kuskus.png" 10;
"tkcqd" "consonants_tkcqd.png" 20;
"uuni" "word_uuni.png" 10;
"kyteä" "word_kyteä.png" 10;
"xggtw" "consonants_xggtw.png" 20;
"häkä" "word_häkä.png" 10;
"dvsvos" "symbols_dvsvos.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _ _" "symbols_dvsvos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^s^sso" "symbols_^s^sso.png" 30;
"nurin" "word_nurin.png" 10;
"qmncn" "consonants_qmncn.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _ _" "consonants_qmncn_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sikhi" "word_sikhi.png" 10;
"känsä" "word_känsä.png" 10;
"tyvi" "word_tyvi.png" 10;
"vlptrn" "consonants_vlptrn.png" 20;
"osinko" "word_osinko.png" 10;
"suti" "word_suti.png" 10;
"nieriä" "word_nieriä.png" 10;
"vvsod" "symbols_vvsod.png" 30;
"ksrwvk" "consonants_ksrwvk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"r _ _ _ _ _" "consonants_ksrwvk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oo^v" "symbols_oo^v.png" 30;
"siitos" "word_siitos.png" 10;
"vsd^o" "symbols_vsd^o.png" 30;
"osuma" "word_osuma.png" 10;
"qckq" "consonants_qckq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ q" "consonants_qckq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^vsov" "symbols_v^vsov.png" 30;
"kopio" "word_kopio.png" 10;
"vsv^" "symbols_vsv^.png" 30;
"solmio" "word_solmio.png" 10;
"myyty" "word_myyty.png" 10;
"uivelo" "word_uivelo.png" 10;
"rtsgdh" "consonants_rtsgdh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"r _ _ _ _ _" "consonants_rtsgdh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"opaali" "word_opaali.png" 10;
"hihna" "word_hihna.png" 10;
"kopina" "word_kopina.png" 10;
"wcjrjq" "consonants_wcjrjq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"w _ _ _ _ _" "consonants_wcjrjq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"näppy" "word_näppy.png" 10;
"ddvsd" "symbols_ddvsd.png" 30;
"jakaja" "word_jakaja.png" 10;
"rfrqfp" "consonants_rfrqfp.png" 20;
"fwgntw" "consonants_fwgntw.png" 20;
"o^v^" "symbols_o^v^.png" 30;
"pidot" "word_pidot.png" 10;
"ripeä" "word_ripeä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _ _" "word_ripeä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ratamo" "word_ratamo.png" 10;
"kaapia" "word_kaapia.png" 10;
"hqzll" "consonants_hqzll.png" 20;
"rapsi" "word_rapsi.png" 10;
"almu" "word_almu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"a _ _ _" "word_almu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jaardi" "word_jaardi.png" 10;
"silsa" "word_silsa.png" 10;
"svds" "symbols_svds.png" 30;
"torium" "word_torium.png" 10;
"d^v^^" "symbols_d^v^^.png" 30;
"osdod^" "symbols_osdod^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s _" "symbols_osdod^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"voo^o" "symbols_voo^o.png" 30;
"czkdbs" "consonants_czkdbs.png" 20;
"dlgt" "consonants_dlgt.png" 20;
"xxqhnz" "consonants_xxqhnz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ x" "consonants_xxqhnz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"emätin" "word_emätin.png" 10;
"pitäen" "word_pitäen.png" 10;
"arpoa" "word_arpoa.png" 10;
"rulla" "word_rulla.png" 10;
"vs^oo" "symbols_vs^oo.png" 30;
"pesin" "word_pesin.png" 10;
"^o^o^v" "symbols_^o^o^v.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_^o^o^v_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"cnwh" "consonants_cnwh.png" 20;
"zkxtf" "consonants_zkxtf.png" 20;
"lemu" "word_lemu.png" 10;
"bgrzd" "consonants_bgrzd.png" 20;
"sovo" "symbols_sovo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ v _" "symbols_sovo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^dss" "symbols_^dss.png" 30;
"vv^^" "symbols_vv^^.png" 30;
"psalmi" "word_psalmi.png" 10;
"hxjlgq" "consonants_hxjlgq.png" 20;
"vvvd" "symbols_vvvd.png" 30;
"grgdd" "consonants_grgdd.png" 20;
"gchqcz" "consonants_gchqcz.png" 20;
"jqdsmj" "consonants_jqdsmj.png" 20;
"jxfm" "consonants_jxfm.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ x" "consonants_jxfm_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oinas" "word_oinas.png" 10;
"gongi" "word_gongi.png" 10;
"dsvvd" "symbols_dsvvd.png" 30;
"tcftcg" "consonants_tcftcg.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ n _ _" "consonants_tcftcg_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vwrptk" "consonants_vwrptk.png" 20;
"mths" "consonants_mths.png" 20;
"qwjvhz" "consonants_qwjvhz.png" 20;
"ovo^s" "symbols_ovo^s.png" 30;
"vvv^" "symbols_vvv^.png" 30;
"^s^d" "symbols_^s^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _" "symbols_^s^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uoma" "word_uoma.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
"mpvl" "consonants_mpvl.png" 20;
"rmmrh" "consonants_rmmrh.png" 20;
"laukka" "word_laukka.png" 10;
"vovso" "symbols_vovso.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_vovso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zpcqc" "consonants_zpcqc.png" 20;
"ltqrr" "consonants_ltqrr.png" 20;
"lakea" "word_lakea.png" 10;
"qvkjt" "consonants_qvkjt.png" 20;
"ssdvo^" "symbols_ssdvo^.png" 30;
"afasia" "word_afasia.png" 10;
"estyä" "word_estyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"e _ _ _ _" "word_estyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hormi" "word_hormi.png" 10;
"^s^^" "symbols_^s^^.png" 30;
"urut" "word_urut.png" 10;
"häpy" "word_häpy.png" 10;
"tdgw" "consonants_tdgw.png" 20;
"vahaus" "word_vahaus.png" 10;
"lblxm" "consonants_lblxm.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ x _ _" "consonants_lblxm_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyköä" "word_tyköä.png" 10;
"ds^^o" "symbols_ds^^o.png" 30;
"juhta" "word_juhta.png" 10;
"menijä" "word_menijä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _ _" "word_menijä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"do^s" "symbols_do^s.png" 30;
"nide" "word_nide.png" 10;
"sdsdoo" "symbols_sdsdoo.png" 30;
"lhqt" "consonants_lhqt.png" 20;
"häät" "word_häät.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ l" "word_häät_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mxnhh" "consonants_mxnhh.png" 20;
"ässä" "word_ässä.png" 10;
"pyörö" "word_pyörö.png" 10;
"kuje" "word_kuje.png" 10;
"vsds" "symbols_vsds.png" 30;
"blkxwz" "consonants_blkxwz.png" 20;
"svsod" "symbols_svsod.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ s _ _" "symbols_svsod_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ovvds" "symbols_ovvds.png" 30;
"sävy" "word_sävy.png" 10;
"kiuas" "word_kiuas.png" 10;
"zjwhs" "consonants_zjwhs.png" 20;
"vvv^v" "symbols_vvv^v.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _" "symbols_vvv^v_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tämä" "word_tämä.png" 10;
"kortti" "word_kortti.png" 10;
"spmb" "consonants_spmb.png" 20;
"sv^vvs" "symbols_sv^vvs.png" 30;
"huorin" "word_huorin.png" 10;
"soov" "symbols_soov.png" 30;
"zsgj" "consonants_zsgj.png" 20;
"täällä" "word_täällä.png" 10;
"vuoka" "word_vuoka.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _" "word_vuoka_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvo^" "symbols_vvo^.png" 30;
"uiva" "word_uiva.png" 10;
"dv^o" "symbols_dv^o.png" 30;
"jtltfj" "consonants_jtltfj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ f _" "consonants_jtltfj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"silaus" "word_silaus.png" 10;
"klaava" "word_klaava.png" 10;
"jäte" "word_jäte.png" 10;
"hajan" "word_hajan.png" 10;
"zqshvj" "consonants_zqshvj.png" 20;
"v^s^vd" "symbols_v^s^vd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ s _ _" "symbols_v^s^vd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"itää" "word_itää.png" 10;
"puida" "word_puida.png" 10;
"räntä" "word_räntä.png" 10;
"voov^v" "symbols_voov^v.png" 30;
"gmrq" "consonants_gmrq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "consonants_gmrq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jänne" "word_jänne.png" 10;
"itiö" "word_itiö.png" 10;
"katodi" "word_katodi.png" 10;
"tenä" "word_tenä.png" 10;
"loraus" "word_loraus.png" 10;
"gkzpg" "consonants_gkzpg.png" 20;
"ilmi" "word_ilmi.png" 10;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"odsoo^" "symbols_odsoo^.png" 30;
"ahjo" "word_ahjo.png" 10;
"s^v^s" "symbols_s^v^s.png" 30;
"zlln" "consonants_zlln.png" 20;
"akti" "word_akti.png" 10;
"terska" "word_terska.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ c _" "word_terska_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"säle" "word_säle.png" 10;
"riimu" "word_riimu.png" 10;
"poru" "word_poru.png" 10;
"kolhia" "word_kolhia.png" 10;
"dodv" "symbols_dodv.png" 30;
"^d^oos" "symbols_^d^oos.png" 30;
"bzhkd" "consonants_bzhkd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ d" "consonants_bzhkd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ldtg" "consonants_ldtg.png" 20;
"druidi" "word_druidi.png" 10;
"ssods" "symbols_ssods.png" 30;
"pesula" "word_pesula.png" 10;
"korren" "word_korren.png" 10;
"sarake" "word_sarake.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"ä _ _ _ _ _" "word_sarake_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sovs" "symbols_sovs.png" 30;
"seimi" "word_seimi.png" 10;
"uuhi" "word_uuhi.png" 10;
"lcrzjb" "consonants_lcrzjb.png" 20;
"uute" "word_uute.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _" "word_uute_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"faksi" "word_faksi.png" 10;
"vipu" "word_vipu.png" 10;
"jaos" "word_jaos.png" 10;
"txhs" "consonants_txhs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ h _" "consonants_txhs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sclc" "consonants_sclc.png" 20;
"hamaan" "word_hamaan.png" 10;
"lohi" "word_lohi.png" 10;
"päkiä" "word_päkiä.png" 10;
"diodi" "word_diodi.png" 10;
"rwfcq" "consonants_rwfcq.png" 20;
"vodds" "symbols_vodds.png" 30;
"gsdx" "consonants_gsdx.png" 20;
"älytä" "word_älytä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ä" "word_älytä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svvd" "symbols_svvd.png" 30;
"mäntä" "word_mäntä.png" 10;
"xgtt" "consonants_xgtt.png" 20;
"zfjxqk" "consonants_zfjxqk.png" 20;
"s^dv^" "symbols_s^dv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _ _" "symbols_s^dv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dwmdd" "consonants_dwmdd.png" 20;
"syöpyä" "word_syöpyä.png" 10;
"erkani" "word_erkani.png" 10;
"zkbwm" "consonants_zkbwm.png" 20;
"säie" "word_säie.png" 10;
"zdvv" "consonants_zdvv.png" 20;
"^s^s" "symbols_^s^s.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ^ _" "symbols_^s^s_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hjqc" "consonants_hjqc.png" 20;
"kyhmy" "word_kyhmy.png" 10;
"^dso" "symbols_^dso.png" 30;
"uima" "word_uima.png" 10;
"sodvds" "symbols_sodvds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_sodvds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tcfwdr" "consonants_tcfwdr.png" 20;
"äimä" "word_äimä.png" 10;
"iäti" "word_iäti.png" 10;
"näkö" "word_näkö.png" 10;
"gnbtn" "consonants_gnbtn.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ n _ _ _" "consonants_gnbtn_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^vdovd" "symbols_^vdovd.png" 30;
"nrxlzx" "consonants_nrxlzx.png" 20;
"kussa" "word_kussa.png" 10;
"^vvsvv" "symbols_^vvsvv.png" 30;
"ampuja" "word_ampuja.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ u _ _" "word_ampuja_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^^^^" "symbols_v^^^^.png" 30;
"gqpkfs" "consonants_gqpkfs.png" 20;
"vodv^^" "symbols_vodv^^.png" 30;
"vaje" "word_vaje.png" 10;
"dfdzj" "consonants_dfdzj.png" 20;
"xmpt" "consonants_xmpt.png" 20;
"vssov" "symbols_vssov.png" 30;
"äänes" "word_äänes.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ä _ _ _" "word_äänes_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^vvdv" "symbols_^vvdv.png" 30;
"lldk" "consonants_lldk.png" 20;
"s^od" "symbols_s^od.png" 30;
"kaste" "word_kaste.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _" "word_kaste_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"noppa" "word_noppa.png" 10;
"dovv^" "symbols_dovv^.png" 30;
"wqghh" "consonants_wqghh.png" 20;
"harjus" "word_harjus.png" 10;
"dsdoo^" "symbols_dsdoo^.png" 30;
"sodo" "symbols_sodo.png" 30;
"wvxw" "consonants_wvxw.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ k _" "consonants_wvxw_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"syaani" "word_syaani.png" 10;
"kustos" "word_kustos.png" 10;
"^s^vds" "symbols_^s^vds.png" 30;
"hautua" "word_hautua.png" 10;
"hnxf" "consonants_hnxf.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ k _ _" "consonants_hnxf_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nczkbj" "consonants_nczkbj.png" 20;
"oo^vdv" "symbols_oo^vdv.png" 30;
"kytkin" "word_kytkin.png" 10;
"vvosod" "symbols_vvosod.png" 30;
"osdoo^" "symbols_osdoo^.png" 30;
"sopu" "word_sopu.png" 10;
"pokeri" "word_pokeri.png" 10;
"ripsi" "word_ripsi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"r _ _ _ _" "word_ripsi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kerubi" "word_kerubi.png" 10;
"sqdq" "consonants_sqdq.png" 20;
"^d^vs" "symbols_^d^vs.png" 30;
"köli" "word_köli.png" 10;
"os^ood" "symbols_os^ood.png" 30;
"ylkä" "word_ylkä.png" 10;
"jdfjs" "consonants_jdfjs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ k _" "consonants_jdfjs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"piip" "word_piip.png" 10;
"bfkx" "consonants_bfkx.png" 20;
"nirso" "word_nirso.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _ _" "word_nirso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tuohus" "word_tuohus.png" 10;
"luusto" "word_luusto.png" 10;
"viipyä" "word_viipyä.png" 10;
"sovssd" "symbols_sovssd.png" 30;
"sysi" "word_sysi.png" 10;
"miten" "word_miten.png" 10;
"orpo" "word_orpo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ö _ _" "word_orpo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vd^s" "symbols_vd^s.png" 30;
"gnxd" "consonants_gnxd.png" 20;
"zkwnr" "consonants_zkwnr.png" 20;
"eliö" "word_eliö.png" 10;
"v^v^" "symbols_v^v^.png" 30;
"rkgpjs" "consonants_rkgpjs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m _ _" "consonants_rkgpjs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^vd^vd" "symbols_^vd^vd.png" 30;
"dod^os" "symbols_dod^os.png" 30;
"särkyä" "word_särkyä.png" 10;
"sovdvo" "symbols_sovdvo.png" 30;
"karies" "word_karies.png" 10;
"^osd" "symbols_^osd.png" 30;
"odote" "word_odote.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ j" "word_odote_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"möly" "word_möly.png" 10;
};
